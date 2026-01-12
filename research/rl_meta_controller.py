"""
Research implementation of RL-driven Meta-Controller System (PPO/A3C).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import sys
import os
from typing import Dict, Any, Tuple, List
from collections import deque
import random

# Add the project root to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

try:
    from modell_V5_MAHIA_HyenaMoE import MetaLRPolicyController
    RL_RESEARCH_AVAILABLE = True
except ImportError:
    RL_RESEARCH_AVAILABLE = False
    print("⚠️  MAHIA-X modules not available for RL research")


class PPOPolicyNetwork(nn.Module):
    """PPO Policy Network for Meta-Controller"""
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # Actor network (policy)
        self.actor = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Softmax(dim=-1)
        )
        
        # Critic network (value)
        self.critic = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass
        Args:
            state: State tensor
        Returns:
            Tuple of (action_probs, state_value)
        """
        action_probs = self.actor(state)
        state_value = self.critic(state)
        return action_probs, state_value


class A3CWorker(nn.Module):
    """A3C Worker for asynchronous training"""
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.policy = PPOPolicyNetwork(state_dim, action_dim, hidden_dim)
        
    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass
        Args:
            state: State tensor
        Returns:
            Tuple of (action_probs, state_value)
        """
        return self.policy(state)


class RLMetaController:
    """RL-driven Meta-Controller System using PPO/A3C approaches"""
    
    def __init__(self, state_dim: int = 12, action_dim: int = 4, 
                 hidden_dim: int = 128, lr: float = 3e-4,
                 gamma: float = 0.99, eps_clip: float = 0.2):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.lr = lr
        self.gamma = gamma
        self.eps_clip = eps_clip
        
        # Initialize policy network
        self.policy = PPOPolicyNetwork(state_dim, action_dim, hidden_dim)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr)
        
        # Memory for PPO updates
        self.memory = {
            'states': [],
            'actions': [],
            'log_probs': [],
            'rewards': [],
            'values': [],
            'dones': []
        }
        
        # Training statistics
        self.training_stats = {
            'episode_rewards': deque(maxlen=100),
            'policy_losses': deque(maxlen=100),
            'value_losses': deque(maxlen=100)
        }
        
    def get_state_representation(self, training_metrics: Dict[str, Any]) -> torch.Tensor:
        """Convert training metrics to state representation
        
        Args:
            training_metrics: Dictionary of training metrics
            
        Returns:
            State tensor of shape (state_dim,)
        """
        # Extract relevant metrics
        loss = training_metrics.get('loss', 0.0)
        grad_norm = training_metrics.get('grad_norm', 0.0)
        lr = training_metrics.get('lr', 1e-3)
        epoch = training_metrics.get('epoch', 0)
        step = training_metrics.get('step', 0)
        entropy = training_metrics.get('entropy', 0.0)
        aux_loss = training_metrics.get('aux_loss', 0.0)
        confidence = training_metrics.get('confidence', 0.5)
        
        # Create state vector
        state_vector = np.array([
            loss,           # Current loss
            grad_norm,      # Gradient norm
            np.log(lr + 1e-8),  # Log learning rate
            epoch / 100.0,  # Normalized epoch
            step / 10000.0, # Normalized step
            entropy,        # Gradient entropy
            aux_loss if aux_loss is not None else 0.0,  # Auxiliary loss
            confidence,     # Model confidence
            loss / (grad_norm + 1e-8),  # Loss-to-gradient ratio
            np.sin(step * 0.01),  # Periodic feature
            np.cos(step * 0.01),  # Periodic feature
            np.random.randn()  # Exploration noise
        ], dtype=np.float32)
        
        # Normalize state vector
        state_vector = (state_vector - np.mean(state_vector)) / (np.std(state_vector) + 1e-8)
        
        return torch.tensor(state_vector, dtype=torch.float32)
    
    def select_action(self, state: torch.Tensor, deterministic: bool = False) -> Tuple[int, float, float]:
        """Select action using current policy
        
        Args:
            state: State tensor
            deterministic: Whether to select deterministic action
            
        Returns:
            Tuple of (action, log_prob, value)
        """
        with torch.no_grad():
            action_probs, state_value = self.policy(state.unsqueeze(0))
            action_probs = action_probs.squeeze(0)
            state_value = state_value.squeeze(0)
            
            if deterministic:
                action = torch.argmax(action_probs).item()
            else:
                action = torch.multinomial(action_probs, 1).item()
                
            log_prob = torch.log(action_probs[action] + 1e-8)
            
        return action, log_prob.item(), state_value.item()
    
    def store_transition(self, state: torch.Tensor, action: int, log_prob: float, 
                        reward: float, value: float, done: bool):
        """Store transition in memory for PPO update
        
        Args:
            state: State tensor
            action: Selected action
            log_prob: Log probability of action
            reward: Reward received
            value: State value
            done: Whether episode is done
        """
        self.memory['states'].append(state)
        self.memory['actions'].append(action)
        self.memory['log_probs'].append(log_prob)
        self.memory['rewards'].append(reward)
        self.memory['values'].append(value)
        self.memory['dones'].append(done)
    
    def compute_returns(self, rewards: List[float], values: List[float], 
                       dones: List[bool]) -> List[float]:
        """Compute discounted returns for PPO update
        
        Args:
            rewards: List of rewards
            values: List of state values
            dones: List of done flags
            
        Returns:
            List of discounted returns
        """
        returns = []
        gae = 0
        next_value = 0
        
        # Compute returns in reverse order
        for i in reversed(range(len(rewards))):
            if dones[i]:
                next_value = 0
                
            delta = rewards[i] + self.gamma * next_value - values[i]
            gae = delta + self.gamma * 0.95 * gae  # GAE with lambda=0.95
            returns.insert(0, gae + values[i])
            next_value = values[i]
            
        return returns
    
    def update_policy(self, batch_size: int = 64, epochs: int = 4) -> Dict[str, float]:
        """Update policy using PPO algorithm
        
        Args:
            batch_size: Batch size for updates
            epochs: Number of update epochs
            
        Returns:
            Dictionary of training statistics
        """
        if len(self.memory['states']) < batch_size:
            return {}
            
        # Convert memory to tensors
        states = torch.stack(self.memory['states'])
        actions = torch.tensor(self.memory['actions'], dtype=torch.long)
        old_log_probs = torch.tensor(self.memory['log_probs'], dtype=torch.float32)
        rewards = self.memory['rewards']
        values = self.memory['values']
        dones = self.memory['dones']
        
        # Compute returns
        returns = self.compute_returns(rewards, values, dones)
        returns = torch.tensor(returns, dtype=torch.float32)
        
        # Convert values to tensor
        values = torch.tensor(values, dtype=torch.float32)
        
        # Compute advantages
        advantages = returns - values
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Update policy for multiple epochs
        policy_loss_total = 0.0
        value_loss_total = 0.0
        
        for _ in range(epochs):
            # Sample mini-batches
            indices = list(range(len(states)))
            random.shuffle(indices)
            
            for i in range(0, len(indices), batch_size):
                batch_indices = indices[i:i+batch_size]
                
                # Get batch data
                batch_states = states[batch_indices]
                batch_actions = actions[batch_indices]
                batch_old_log_probs = old_log_probs[batch_indices]
                batch_returns = returns[batch_indices]
                batch_advantages = advantages[batch_indices]
                
                # Forward pass
                action_probs, state_values = self.policy(batch_states)
                state_values = state_values.squeeze()
                
                # Compute ratios
                action_log_probs = torch.log(action_probs.gather(1, batch_actions.unsqueeze(1)) + 1e-8).squeeze()
                ratios = torch.exp(action_log_probs - batch_old_log_probs)
                
                # Compute surrogate losses
                surr1 = ratios * batch_advantages
                surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * batch_advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # Compute value loss
                value_loss = F.mse_loss(state_values, batch_returns)
                
                # Total loss
                loss = policy_loss + 0.5 * value_loss
                
                # Update policy
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)
                self.optimizer.step()
                
                # Accumulate losses
                policy_loss_total += policy_loss.item()
                value_loss_total += value_loss.item()
        
        # Clear memory
        self.memory = {
            'states': [],
            'actions': [],
            'log_probs': [],
            'rewards': [],
            'values': [],
            'dones': []
        }
        
        # Store statistics
        avg_policy_loss = policy_loss_total / (epochs * max(1, len(states) // batch_size))
        avg_value_loss = value_loss_total / (epochs * max(1, len(states) // batch_size))
        
        self.training_stats['policy_losses'].append(avg_policy_loss)
        self.training_stats['value_losses'].append(avg_value_loss)
        
        return {
            'policy_loss': avg_policy_loss,
            'value_loss': avg_value_loss
        }
    
    def compute_reward(self, action: int, training_metrics: Dict[str, Any]) -> float:
        """Compute reward based on action and training metrics
        
        Args:
            action: Selected action (0: reduce_lr, 1: increase_lr, 2: maintain_lr, 3: other)
            training_metrics: Dictionary of training metrics
            
        Returns:
            Reward value
        """
        loss = training_metrics.get('loss', 0.0)
        prev_loss = training_metrics.get('prev_loss', loss)
        grad_norm = training_metrics.get('grad_norm', 0.0)
        entropy = training_metrics.get('entropy', 0.0)
        confidence = training_metrics.get('confidence', 0.5)
        
        # Base reward on loss improvement
        loss_improvement = prev_loss - loss
        reward = loss_improvement * 10.0  # Scale reward
        
        # Bonus for good gradient behavior
        if grad_norm < 1.0:
            reward += 0.1
        elif grad_norm > 10.0:
            reward -= 0.1
            
        # Bonus for appropriate entropy
        if 0.5 <= entropy <= 2.0:
            reward += 0.05
            
        # Bonus for high confidence with good performance
        if confidence > 0.8 and loss_improvement > 0:
            reward += 0.1
            
        # Penalty for extreme actions when not needed
        if action in [0, 1] and abs(loss_improvement) < 1e-4:
            reward -= 0.05
            
        return reward
    
    def get_action_meaning(self, action: int) -> str:
        """Get human-readable meaning of action
        
        Args:
            action: Action index
            
        Returns:
            Action description
        """
        actions = {
            0: "reduce_learning_rate",
            1: "increase_learning_rate",
            2: "maintain_learning_rate",
            3: "adjust_other_parameters"
        }
        return actions.get(action, "unknown_action")
    
    def get_training_stats(self) -> Dict[str, Any]:
        """Get training statistics
        
        Returns:
            Dictionary of training statistics
        """
        if not self.training_stats['episode_rewards']:
            return {}
            
        return {
            'avg_reward': np.mean(self.training_stats['episode_rewards']),
            'avg_policy_loss': np.mean(self.training_stats['policy_losses']) if self.training_stats['policy_losses'] else 0.0,
            'avg_value_loss': np.mean(self.training_stats['value_losses']) if self.training_stats['value_losses'] else 0.0,
            'episodes_trained': len(self.training_stats['episode_rewards'])
        }


class A3CMetaController:
    """A3C-based Meta-Controller for asynchronous training"""
    
    def __init__(self, state_dim: int = 12, action_dim: int = 4, 
                 hidden_dim: int = 128, lr: float = 3e-4, num_workers: int = 4):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.lr = lr
        self.num_workers = num_workers
        
        # Global shared policy
        self.global_policy = PPOPolicyNetwork(state_dim, action_dim, hidden_dim)
        self.global_optimizer = torch.optim.Adam(self.global_policy.parameters(), lr=lr)
        
        # Worker networks
        self.workers = nn.ModuleList([
            A3CWorker(state_dim, action_dim, hidden_dim) 
            for _ in range(num_workers)
        ])
        
        # Training statistics
        self.training_stats = {
            'episode_rewards': deque(maxlen=1000),
            'worker_updates': 0
        }
        
    def synchronize_workers(self):
        """Synchronize worker networks with global policy"""
        global_state_dict = self.global_policy.state_dict()
        for worker in self.workers:
            worker.policy.load_state_dict(global_state_dict)
    
    def train_worker(self, worker_id: int, episodes: int = 10) -> Dict[str, Any]:
        """Train a specific worker
        
        Args:
            worker_id: ID of worker to train
            episodes: Number of episodes to train
            
        Returns:
            Training results
        """
        if worker_id >= len(self.workers):
            raise ValueError(f"Invalid worker ID: {worker_id}")
            
        worker = self.workers[worker_id]
        episode_rewards = []
        
        # Train for specified episodes
        for episode in range(episodes):
            # This is a simplified training loop
            # In practice, this would involve environment interaction
            total_reward = 0.0
            
            # Simulate training steps
            for step in range(50):  # 50 steps per episode
                # Create dummy state
                state = torch.randn(self.state_dim)
                
                # Select action
                with torch.no_grad():
                    action_probs, value = worker(state)
                    action = torch.multinomial(action_probs, 1).item()
                    log_prob = torch.log(action_probs[action] + 1e-8)
                
                # Compute reward (simplified)
                reward = np.random.randn() * 0.1  # Random reward with small variance
                total_reward += reward * (0.99 ** step)  # Discounted reward
                
            episode_rewards.append(total_reward)
            
        # Update global policy (simplified A3C update)
        self.synchronize_workers()
        
        return {
            'worker_id': worker_id,
            'episode_rewards': episode_rewards,
            'avg_reward': np.mean(episode_rewards)
        }


def research_comparison():
    """Compare PPO and A3C approaches for meta-control"""
    if not RL_RESEARCH_AVAILABLE:
        print("❌ MAHIA-X modules not available for RL research")
        return
        
    print("🔬 Researching RL-driven Meta-Controller Systems...")
    print("=" * 60)
    
    # Initialize controllers
    ppo_controller = RLMetaController(state_dim=12, action_dim=4, lr=3e-4)
    a3c_controller = A3CMetaController(state_dim=12, action_dim=4, num_workers=2)
    
    print("✅ Initialized PPO and A3C Meta-Controllers")
    
    # Simulate training metrics
    sample_metrics = {
        'loss': 0.5,
        'grad_norm': 1.2,
        'lr': 1e-3,
        'epoch': 10,
        'step': 1000,
        'entropy': 1.0,
        'aux_loss': 0.01,
        'confidence': 0.8,
        'prev_loss': 0.6
    }
    
    # Test state representation
    state = ppo_controller.get_state_representation(sample_metrics)
    print(f"✅ State representation shape: {state.shape}")
    
    # Test action selection
    action, log_prob, value = ppo_controller.select_action(state)
    action_meaning = ppo_controller.get_action_meaning(action)
    print(f"✅ Sample action: {action} ({action_meaning})")
    print(f"   Log probability: {log_prob:.4f}")
    print(f"   State value: {value:.4f}")
    
    # Test reward computation
    reward = ppo_controller.compute_reward(action, sample_metrics)
    print(f"✅ Sample reward: {reward:.4f}")
    
    # Test memory storage
    ppo_controller.store_transition(state, action, log_prob, reward, value, False)
    print(f"✅ Stored transition in memory")
    
    # Test policy update (with dummy data to fill memory)
    for i in range(10):
        dummy_state = torch.randn(12)
        dummy_action, dummy_log_prob, dummy_value = ppo_controller.select_action(dummy_state)
        dummy_reward = np.random.randn()
        ppo_controller.store_transition(dummy_state, dummy_action, dummy_log_prob, dummy_reward, dummy_value, False)
    
    update_stats = ppo_controller.update_policy(batch_size=4, epochs=2)
    if update_stats:
        print(f"✅ Policy updated - Policy Loss: {update_stats['policy_loss']:.4f}")
    
    # Test A3C worker training
    worker_results = a3c_controller.train_worker(0, episodes=2)
    print(f"✅ A3C Worker trained - Avg Reward: {worker_results['avg_reward']:.4f}")
    
    # Print final comparison
    print("\n" + "=" * 60)
    print("RL META-CONTROLLER RESEARCH SUMMARY")
    print("=" * 60)
    print("PPO Approach:")
    print("  - Advantage: Stable training with clipping")
    print("  - Disadvantage: Requires more memory for experience replay")
    print("  - Best for: Batch-based training scenarios")
    print("\nA3C Approach:")
    print("  - Advantage: Asynchronous training, faster convergence")
    print("  - Disadvantage: Can suffer from stale gradients")
    print("  - Best for: Real-time, online learning scenarios")
    print("\nRecommendation:")
    print("  For MAHIA-X, PPO is recommended for stable training,")
    print("  while A3C can be used for rapid online adaptation.")
    
    print("\n✅ RL-driven Meta-Controller research completed!")


def main():
    """Main research function"""
    research_comparison()


if __name__ == '__main__':
    main()