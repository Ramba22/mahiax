"""
Meta-Controller Reinforcement Loop for MAHIA-X
Implements self-learning LR/Precision adjustment using reinforcement learning
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
from typing import Dict, Any, Optional, List, Tuple
from collections import deque
import time
from datetime import datetime

# Conditional imports
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    
try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False

class MetaPolicyNetwork(nn.Module):
    """Policy network for meta-controller decisions"""
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 128):
        """
        Initialize policy network
        
        Args:
            state_dim: Dimension of state space
            action_dim: Dimension of action space
            hidden_dim: Hidden layer dimension
        """
        super().__init__()
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # Policy network
        self.policy_net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )
        
        # Value network for baseline
        self.value_net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
        self._init_weights()
        
    def _init_weights(self):
        """Initialize network weights"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0.0)
                
    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass
        
        Args:
            state: State tensor
            
        Returns:
            Tuple of (action logits, state value)
        """
        action_logits = self.policy_net(state)
        state_value = self.value_net(state)
        return action_logits, state_value
        
    def get_action(self, state: torch.Tensor, deterministic: bool = False) -> Tuple[int, float]:
        """
        Get action from policy
        
        Args:
            state: State tensor
            deterministic: Whether to select deterministic action
            
        Returns:
            Tuple of (action index, log probability)
        """
        action_logits, _ = self.forward(state)
        action_probs = F.softmax(action_logits, dim=-1)
        
        if deterministic:
            action = torch.argmax(action_probs, dim=-1).item()
        else:
            action = torch.multinomial(action_probs, 1).item()
            
        log_prob = torch.log(action_probs[action] + 1e-8)
        return action, log_prob.item()


class MetaControllerRL:
    """Meta-controller using reinforcement learning for self-learning adjustments"""
    
    def __init__(self, 
                 state_dim: int = 16,
                 action_dim: int = 8,
                 learning_rate: float = 1e-3,
                 gamma: float = 0.99,
                 epsilon: float = 0.1,
                 buffer_size: int = 10000):
        """
        Initialize meta-controller
        
        Args:
            state_dim: Dimension of state space
            action_dim: Dimension of action space
            learning_rate: Learning rate for policy updates
            gamma: Discount factor
            epsilon: Exploration rate
            buffer_size: Size of experience replay buffer
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon
        
        # Initialize policy network
        self.policy_net = MetaPolicyNetwork(state_dim, action_dim)
        
        # Optimizer
        self.optimizer = torch.optim.Adam(
            self.policy_net.parameters(), 
            lr=learning_rate
        )
        
        # Experience replay buffer
        self.replay_buffer = deque(maxlen=buffer_size)
        
        # Training statistics
        self.total_rewards = []
        self.losses = []
        self.step_count = 0
        
        # Action space definition
        self._define_action_space()
        
        print(f"✅ MetaControllerRL initialized")
        print(f"   State dim: {state_dim}, Action dim: {action_dim}")
        print(f"   Parameters: {sum(p.numel() for p in self.policy_net.parameters()):,}")
        
    def _define_action_space(self):
        """Define the action space for learning rate and precision adjustments"""
        self.action_descriptions = [
            "Increase learning rate significantly",
            "Increase learning rate moderately",
            "Slightly increase learning rate",
            "Keep learning rate unchanged",
            "Slightly decrease learning rate",
            "Decrease learning rate moderately",
            "Decrease learning rate significantly",
            "Switch to lower precision (FP32->FP16->FP8)"
        ]
        
    def get_state(self, training_metrics: Dict[str, Any]) -> torch.Tensor:
        """
        Extract state from training metrics
        
        Args:
            training_metrics: Current training metrics
            
        Returns:
            State tensor
        """
        # Extract relevant metrics
        loss = training_metrics.get("loss", 0.0)
        accuracy = training_metrics.get("accuracy", 0.0)
        gradient_norm = training_metrics.get("gradient_norm", 0.0)
        learning_rate = training_metrics.get("learning_rate", 1e-3)
        batch_time = training_metrics.get("batch_time", 0.1)
        memory_usage = training_metrics.get("memory_usage", 0.5)
        entropy = training_metrics.get("entropy", 1.0)
        confidence = training_metrics.get("confidence", 0.5)
        
        # Normalize metrics to [0, 1] range
        normalized_metrics = [
            min(loss / 10.0, 1.0),  # Normalize loss
            accuracy,  # Already in [0, 1]
            min(gradient_norm / 100.0, 1.0),  # Normalize gradient norm
            min(learning_rate / 1e-2, 1.0),  # Normalize learning rate
            min(batch_time / 1.0, 1.0),  # Normalize batch time
            memory_usage,  # Already in [0, 1]
            entropy,  # Already in [0, 1]
            confidence  # Already in [0, 1]
        ]
        
        # Add derived metrics
        loss_improvement = training_metrics.get("loss_improvement", 0.0)
        accuracy_improvement = training_metrics.get("accuracy_improvement", 0.0)
        
        derived_metrics = [
            max(min(loss_improvement * 100, 1.0), -1.0),  # Normalize improvement
            max(min(accuracy_improvement * 100, 1.0), -1.0)
        ]
        
        # Combine all metrics
        state_vector = normalized_metrics + derived_metrics
        
        # Pad or truncate to state_dim
        if len(state_vector) < self.state_dim:
            state_vector.extend([0.0] * (self.state_dim - len(state_vector)))
        elif len(state_vector) > self.state_dim:
            state_vector = state_vector[:self.state_dim]
            
        return torch.tensor(state_vector, dtype=torch.float32)
        
    def select_action(self, state: torch.Tensor, deterministic: bool = False) -> Tuple[int, float]:
        """
        Select action based on current state
        
        Args:
            state: Current state
            deterministic: Whether to select deterministic action
            
        Returns:
            Tuple of (action index, log probability)
        """
        # Epsilon-greedy exploration
        if not deterministic and random.random() < self.epsilon:
            action = random.randint(0, self.action_dim - 1)
            log_prob = torch.log(torch.tensor(1.0 / self.action_dim))
            return action, log_prob.item()
            
        # Policy-based action selection
        return self.policy_net.get_action(state, deterministic)
        
    def compute_reward(self, current_metrics: Dict[str, Any], 
                      previous_metrics: Dict[str, Any]) -> float:
        """
        Compute reward based on metric improvements
        
        Args:
            current_metrics: Current training metrics
            previous_metrics: Previous training metrics
            
        Returns:
            Reward value
        """
        # Extract metrics
        current_loss = current_metrics.get("loss", 0.0)
        previous_loss = previous_metrics.get("loss", current_loss)
        
        current_accuracy = current_metrics.get("accuracy", 0.0)
        previous_accuracy = previous_metrics.get("accuracy", current_accuracy)
        
        current_time = current_metrics.get("batch_time", 0.1)
        previous_time = previous_metrics.get("batch_time", current_time)
        
        current_memory = current_metrics.get("memory_usage", 0.5)
        previous_memory = previous_metrics.get("memory_usage", current_memory)
        
        # Compute improvements
        loss_improvement = previous_loss - current_loss
        accuracy_improvement = current_accuracy - previous_accuracy
        time_improvement = previous_time - current_time  # Negative is better
        memory_improvement = previous_memory - current_memory  # Negative is better
        
        # Compute reward components
        loss_reward = loss_improvement * 10.0  # Scale loss improvement
        accuracy_reward = accuracy_improvement * 5.0  # Scale accuracy improvement
        efficiency_reward = (time_improvement * 2.0) + (memory_improvement * 3.0)
        
        # Combine rewards
        total_reward = loss_reward + accuracy_reward + efficiency_reward
        
        # Penalty for extreme values
        if current_loss > 10.0:
            total_reward -= 5.0  # Penalty for high loss
        if current_memory > 0.9:
            total_reward -= 3.0  # Penalty for high memory usage
            
        return total_reward
        
    def update_policy(self, states: List[torch.Tensor], 
                     actions: List[int], 
                     rewards: List[float], 
                     next_states: List[torch.Tensor],
                     dones: List[bool]) -> float:
        """
        Update policy using REINFORCE with baseline
        
        Args:
            states: List of state tensors
            actions: List of actions
            rewards: List of rewards
            next_states: List of next state tensors
            dones: List of done flags
            
        Returns:
            Loss value
        """
        if len(states) == 0:
            return 0.0
            
        # Convert to tensors
        states_tensor = torch.stack(states)
        actions_tensor = torch.tensor(actions, dtype=torch.long)
        rewards_tensor = torch.tensor(rewards, dtype=torch.float32)
        
        # Compute action probabilities and state values
        action_logits, state_values = self.policy_net(states_tensor)
        action_probs = F.softmax(action_logits, dim=-1)
        selected_action_probs = action_probs.gather(1, actions_tensor.unsqueeze(1)).squeeze()
        log_probs = torch.log(selected_action_probs + 1e-8)
        
        # Compute returns (simplified for online learning)
        returns = rewards_tensor
        
        # Compute advantage (return - baseline)
        advantages = returns - state_values.squeeze()
        
        # Policy loss with baseline
        policy_loss = (-log_probs * advantages.detach()).mean()
        
        # Value loss
        value_loss = F.mse_loss(state_values.squeeze(), returns)
        
        # Total loss
        total_loss = policy_loss + 0.5 * value_loss
        
        # Update policy
        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=1.0)
        self.optimizer.step()
        
        loss_value = total_loss.item()
        self.losses.append(loss_value)
        
        return loss_value
        
    def step(self, current_metrics: Dict[str, Any], 
             previous_metrics: Dict[str, Any]) -> Tuple[int, float, Dict[str, Any]]:
        """
        Perform one step of meta-control
        
        Args:
            current_metrics: Current training metrics
            previous_metrics: Previous training metrics
            
        Returns:
            Tuple of (action, reward, action_info)
        """
        # Get current state
        state = self.get_state(current_metrics)
        
        # Select action
        action, log_prob = self.select_action(state)
        
        # Compute reward
        reward = self.compute_reward(current_metrics, previous_metrics)
        self.total_rewards.append(reward)
        
        # Store experience
        experience = {
            "state": state,
            "action": action,
            "reward": reward,
            "log_prob": log_prob,
            "timestamp": time.time()
        }
        
        # Update policy periodically
        if self.step_count % 10 == 0 and len(self.replay_buffer) > 32:
            # Sample batch from replay buffer
            batch_size = min(32, len(self.replay_buffer))
            batch = random.sample(self.replay_buffer, batch_size)
            
            states = [exp["state"] for exp in batch]
            actions = [exp["action"] for exp in batch]
            rewards = [exp["reward"] for exp in batch]
            # For simplicity, we're not using next_states and dones in this implementation
            next_states = [exp["state"] for exp in batch]  # Placeholder
            dones = [False] * len(batch)  # Placeholder
            
            loss = self.update_policy(states, actions, rewards, next_states, dones)
            
        # Store experience in replay buffer
        self.replay_buffer.append(experience)
        self.step_count += 1
        
        # Generate action info
        action_info = {
            "action_description": self.action_descriptions[action],
            "action_index": action,
            "log_probability": log_prob,
            "state_metrics": {
                "loss": current_metrics.get("loss", 0.0),
                "accuracy": current_metrics.get("accuracy", 0.0),
                "gradient_norm": current_metrics.get("gradient_norm", 0.0),
                "learning_rate": current_metrics.get("learning_rate", 1e-3)
            }
        }
        
        return action, reward, action_info
        
    def adjust_learning_rate(self, current_lr: float, action: int) -> float:
        """
        Adjust learning rate based on action
        
        Args:
            current_lr: Current learning rate
            action: Action index
            
        Returns:
            Adjusted learning rate
        """
        lr_adjustments = {
            0: current_lr * 2.0,      # Increase significantly
            1: current_lr * 1.5,      # Increase moderately
            2: current_lr * 1.1,      # Slightly increase
            3: current_lr,            # Keep unchanged
            4: current_lr * 0.9,      # Slightly decrease
            5: current_lr * 0.7,      # Decrease moderately
            6: current_lr * 0.5,      # Decrease significantly
            7: current_lr             # Precision change action (handled separately)
        }
        
        return lr_adjustments.get(action, current_lr)
        
    def adjust_precision(self, current_precision: str, action: int) -> str:
        """
        Adjust precision based on action
        
        Args:
            current_precision: Current precision level
            action: Action index
            
        Returns:
            Adjusted precision level
        """
        if action != 7:  # Not precision change action
            return current_precision
            
        # Precision levels in order of decreasing precision
        precision_levels = ["FP32", "FP16", "BF16", "FP8", "INT8", "INT4"]
        
        try:
            current_index = precision_levels.index(current_precision)
            # Move to next lower precision (higher index)
            new_index = min(current_index + 1, len(precision_levels) - 1)
            return precision_levels[new_index]
        except ValueError:
            # Current precision not in list, return FP16 as default
            return "FP16"
            
    def get_training_stats(self) -> Dict[str, Any]:
        """
        Get training statistics
        
        Returns:
            Dictionary of training statistics
        """
        if not self.total_rewards:
            return {"status": "no_data"}
            
        return {
            "total_steps": self.step_count,
            "average_reward": np.mean(self.total_rewards) if NUMPY_AVAILABLE else sum(self.total_rewards) / len(self.total_rewards),
            "recent_rewards": self.total_rewards[-100:],  # Last 100 rewards
            "average_loss": np.mean(self.losses) if self.losses and NUMPY_AVAILABLE else (sum(self.losses) / len(self.losses) if self.losses else 0.0),
            "buffer_size": len(self.replay_buffer),
            "exploration_rate": self.epsilon
        }
        
    def save_model(self, filepath: str):
        """
        Save model to file
        
        Args:
            filepath: Path to save model
        """
        torch.save({
            'policy_net_state_dict': self.policy_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'training_stats': self.get_training_stats()
        }, filepath)
        print(f"✅ Meta-controller model saved to {filepath}")
        
    def load_model(self, filepath: str):
        """
        Load model from file
        
        Args:
            filepath: Path to load model from
        """
        checkpoint = torch.load(filepath)
        self.policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print(f"✅ Meta-controller model loaded from {filepath}")


def demo_meta_controller_rl():
    """Demonstrate meta-controller RL functionality"""
    print("🚀 Demonstrating Meta-Controller Reinforcement Learning...")
    print("=" * 60)
    
    # Create meta-controller
    meta_controller = MetaControllerRL(
        state_dim=16,
        action_dim=8,
        learning_rate=1e-3
    )
    print("✅ Created meta-controller RL")
    
    # Simulate training loop
    print("\n🔄 Simulating training with meta-control...")
    
    # Initial metrics
    previous_metrics = {
        "loss": 1.0,
        "accuracy": 0.5,
        "gradient_norm": 5.0,
        "learning_rate": 1e-3,
        "batch_time": 0.5,
        "memory_usage": 0.3,
        "entropy": 0.8,
        "confidence": 0.6
    }
    
    current_lr = 1e-3
    current_precision = "FP16"
    
    # Simulate 20 training steps
    for step in range(20):
        # Simulate changing metrics
        current_metrics = {
            "loss": max(0.1, previous_metrics["loss"] - random.uniform(0.01, 0.1)),
            "accuracy": min(0.99, previous_metrics["accuracy"] + random.uniform(0.001, 0.02)),
            "gradient_norm": max(0.1, previous_metrics["gradient_norm"] + random.uniform(-0.5, 0.5)),
            "learning_rate": current_lr,
            "batch_time": max(0.1, previous_metrics["batch_time"] + random.uniform(-0.05, 0.05)),
            "memory_usage": min(0.99, previous_metrics["memory_usage"] + random.uniform(-0.01, 0.02)),
            "entropy": max(0.01, previous_metrics["entropy"] - random.uniform(0.01, 0.05)),
            "confidence": min(0.99, previous_metrics["confidence"] + random.uniform(-0.02, 0.03))
        }
        
        # Add improvement metrics
        current_metrics["loss_improvement"] = previous_metrics["loss"] - current_metrics["loss"]
        current_metrics["accuracy_improvement"] = current_metrics["accuracy"] - previous_metrics["accuracy"]
        
        # Perform meta-control step
        action, reward, action_info = meta_controller.step(current_metrics, previous_metrics)
        
        # Apply action
        if action < 7:  # Learning rate adjustment
            new_lr = meta_controller.adjust_learning_rate(current_lr, action)
            if new_lr != current_lr:
                print(f"   Step {step+1}: LR adjusted from {current_lr:.6f} to {new_lr:.6f}")
                current_lr = new_lr
        else:  # Precision adjustment
            new_precision = meta_controller.adjust_precision(current_precision, action)
            if new_precision != current_precision:
                print(f"   Step {step+1}: Precision changed from {current_precision} to {new_precision}")
                current_precision = new_precision
                
        # Print step info
        if step % 5 == 0:
            print(f"   Step {step+1}: Action='{action_info['action_description']}', "
                  f"Reward={reward:.3f}, Loss={current_metrics['loss']:.4f}, "
                  f"Accuracy={current_metrics['accuracy']:.4f}")
        
        # Update previous metrics
        previous_metrics = current_metrics.copy()
        
    # Show training statistics
    stats = meta_controller.get_training_stats()
    print(f"\n📊 Training Statistics:")
    print(f"   Total Steps: {stats['total_steps']}")
    print(f"   Average Reward: {stats['average_reward']:.3f}")
    print(f"   Average Loss: {stats['average_loss']:.4f}")
    print(f"   Replay Buffer Size: {stats['buffer_size']}")
    
    # Save model
    meta_controller.save_model("demo_meta_controller.pth")
    
    print("\n" + "=" * 60)
    print("META-CONTROLLER RL DEMO SUMMARY")
    print("=" * 60)
    print("Key Features Implemented:")
    print("  1. Reinforcement learning policy network")
    print("  2. State representation from training metrics")
    print("  3. Action space for LR and precision adjustments")
    print("  4. Reward function based on metric improvements")
    print("  5. Experience replay for stable learning")
    print("  6. Online policy updates")
    print("\nBenefits:")
    print("  - Self-learning optimization policies")
    print("  - Adaptive learning rate adjustment")
    print("  - Dynamic precision management")
    print("  - Automated hyperparameter tuning")
    print("  - Continuous improvement through RL")
    
    print("\n✅ Meta-controller RL demonstration completed!")


if __name__ == "__main__":
    demo_meta_controller_rl()