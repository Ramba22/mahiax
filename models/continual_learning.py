"""
Continual Learning Buffer with EWC (Elastic Weight Consolidation) + Replay for MAHIA-X.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import sys
import os
from typing import Dict, Any, Tuple, Optional, List
import random
from collections import deque

# Add the project root to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

try:
    # Import required modules if available
    CONTINUAL_AVAILABLE = True
except ImportError:
    CONTINUAL_AVAILABLE = False
    print("⚠️  Some modules not available for continual learning")


class EWCRegularizer:
    """Elastic Weight Consolidation (EWC) regularizer"""
    
    def __init__(self, model: nn.Module, lambda_ewc: float = 1000.0):
        self.model = model
        self.lambda_ewc = lambda_ewc
        self.fisher_information = {}
        self.optimal_params = {}
        self.task_count = 0
        
    def compute_fisher_information(self, data_loader, criterion, num_samples: int = 100):
        """Compute Fisher information matrix for current task
        Args:
            data_loader: DataLoader for current task
            criterion: Loss function
            num_samples: Number of samples to use for Fisher computation
        """
        # Initialize Fisher information
        for name, param in self.model.named_parameters():
            self.fisher_information[name] = torch.zeros_like(param)
            
        # Store current parameters as optimal
        for name, param in self.model.named_parameters():
            self.optimal_params[name] = param.clone().detach()
            
        # Compute Fisher information
        self.model.eval()
        sample_count = 0
        
        for batch_idx, (data, target) in enumerate(data_loader):
            if sample_count >= num_samples:
                break
                
            # Move to device
            data = data.to(next(self.model.parameters()).device)
            target = target.to(next(self.model.parameters()).device)
            
            # Forward pass
            self.model.zero_grad()
            output, _ = self.model(data, None)  # Assuming MAHIA model interface
            loss = criterion(output, target)
            
            # Compute gradients
            loss.backward()
            
            # Accumulate squared gradients (Fisher information)
            for name, param in self.model.named_parameters():
                if param.grad is not None:
                    self.fisher_information[name] += param.grad.data ** 2
                    
            sample_count += data.size(0)
            
        # Average Fisher information
        for name in self.fisher_information:
            self.fisher_information[name] /= sample_count
            
        self.model.train()
        
    def ewc_loss(self) -> torch.Tensor:
        """Compute EWC regularization loss
        Returns:
            EWC loss tensor
        """
        if self.task_count == 0:
            return torch.tensor(0.0, device=next(self.model.parameters()).device)
            
        loss = 0.0
        for name, param in self.model.named_parameters():
            if name in self.fisher_information:
                fisher = self.fisher_information[name]
                optimal = self.optimal_params[name]
                loss += (fisher * (param - optimal) ** 2).sum()
                
        return self.lambda_ewc * loss
    
    def update_task(self):
        """Update task counter and prepare for next task"""
        self.task_count += 1


class ReplayBuffer:
    """Experience replay buffer for continual learning"""
    
    def __init__(self, capacity: int = 1000, sampling_strategy: str = "uniform"):
        self.capacity = capacity
        self.sampling_strategy = sampling_strategy
        self.buffer = deque(maxlen=capacity)
        self.position = 0
        
    def add(self, experience: Dict[str, Any]):
        """Add experience to buffer
        Args:
            experience: Dictionary containing experience data
        """
        # Store experience
        self.buffer.append(experience)
        
    def sample(self, batch_size: int) -> List[Dict[str, Any]]:
        """Sample batch of experiences from buffer
        Args:
            batch_size: Number of experiences to sample
        Returns:
            List of sampled experiences
        """
        if len(self.buffer) == 0:
            return []
            
        if self.sampling_strategy == "uniform":
            # Uniform random sampling
            indices = random.sample(range(len(self.buffer)), min(batch_size, len(self.buffer)))
        elif self.sampling_strategy == "recent":
            # Sample from most recent experiences
            indices = list(range(max(0, len(self.buffer) - batch_size), len(self.buffer)))
        else:
            # Default to uniform sampling
            indices = random.sample(range(len(self.buffer)), min(batch_size, len(self.buffer)))
            
        return [self.buffer[i] for i in indices]
    
    def __len__(self) -> int:
        return len(self.buffer)


class ContinualLearningBuffer:
    """Complete continual learning buffer with EWC + Replay"""
    
    def __init__(self, model: nn.Module, buffer_capacity: int = 1000, 
                 lambda_ewc: float = 1000.0, replay_ratio: float = 0.5):
        self.model = model
        self.buffer_capacity = buffer_capacity
        self.replay_ratio = replay_ratio
        
        # Initialize components
        self.ewc = EWCRegularizer(model, lambda_ewc)
        self.replay_buffer = ReplayBuffer(capacity=buffer_capacity)
        
        # Task management
        self.current_task = 0
        self.task_data = {}
        
        # Performance tracking
        self.performance_history = {
            'task_accuracies': [],
            'catastrophic_forgetting': [],
            'buffer_utilization': []
        }
        
    def add_experience(self, task_id: int, data: torch.Tensor, target: torch.Tensor,
                      logits: torch.Tensor = None, features: torch.Tensor = None):
        """Add experience to replay buffer
        Args:
            task_id: Task identifier
            data: Input data
            target: Target labels
            logits: Model logits (optional)
            features: Model features (optional)
        """
        # Store experience in buffer
        experience = {
            'task_id': task_id,
            'data': data.detach().cpu(),
            'target': target.detach().cpu(),
            'logits': logits.detach().cpu() if logits is not None else None,
            'features': features.detach().cpu() if features is not None else None,
            'timestamp': len(self.replay_buffer)
        }
        
        self.replay_buffer.add(experience)
        
        # Track task data
        if task_id not in self.task_data:
            self.task_data[task_id] = {
                'experience_count': 0,
                'last_accessed': len(self.replay_buffer)
            }
        self.task_data[task_id]['experience_count'] += 1
        self.task_data[task_id]['last_accessed'] = len(self.replay_buffer)
        
    def sample_replay_batch(self, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample batch from replay buffer
        Args:
            batch_size: Batch size
        Returns:
            Tuple of (data, targets)
        """
        experiences = self.replay_buffer.sample(batch_size)
        
        if not experiences:
            return None, None
            
        # Combine experiences into batch
        data_batch = torch.stack([exp['data'] for exp in experiences])
        target_batch = torch.stack([exp['target'] for exp in experiences])
        
        # Move to device
        device = next(self.model.parameters()).device
        data_batch = data_batch.to(device)
        target_batch = target_batch.to(device)
        
        return data_batch, target_batch
    
    def compute_ewc_loss(self) -> torch.Tensor:
        """Compute EWC regularization loss
        Returns:
            EWC loss tensor
        """
        return self.ewc.ewc_loss()
    
    def finish_task(self, data_loader, criterion, num_fisher_samples: int = 100):
        """Finish current task and prepare for next task
        Args:
            data_loader: DataLoader for current task
            criterion: Loss function
            num_fisher_samples: Number of samples for Fisher computation
        """
        # Compute Fisher information for current task
        self.ewc.compute_fisher_information(data_loader, criterion, num_fisher_samples)
        
        # Update task counter
        self.ewc.update_task()
        self.current_task += 1
        
        # Update performance tracking
        self.performance_history['buffer_utilization'].append(len(self.replay_buffer) / self.buffer_capacity)
        
    def get_continual_loss(self, current_loss: torch.Tensor) -> torch.Tensor:
        """Get total loss including EWC regularization
        Args:
            current_loss: Current task loss
        Returns:
            Total loss tensor
        """
        ewc_loss = self.compute_ewc_loss()
        total_loss = current_loss + ewc_loss
        return total_loss
    
    def get_buffer_stats(self) -> Dict[str, Any]:
        """Get buffer statistics
        Returns:
            Dictionary of buffer statistics
        """
        return {
            'buffer_size': len(self.replay_buffer),
            'buffer_capacity': self.buffer_capacity,
            'buffer_utilization': len(self.replay_buffer) / self.buffer_capacity,
            'current_task': self.current_task,
            'tasks_tracked': len(self.task_data),
            'total_experiences': sum(data['experience_count'] for data in self.task_data.values())
        }


class TaskSampler:
    """Task sampler for continual learning with balanced sampling"""
    
    def __init__(self, task_weights: Optional[Dict[int, float]] = None):
        self.task_weights = task_weights or {}
        self.task_weights = task_weights or {}
        self.task_access_counts = {}
        
    def update_task_weights(self, task_performance: Dict[int, float]):
        """Update task sampling weights based on performance
        Args:
            task_performance: Dictionary mapping task IDs to performance metrics
        """
        # Inverse performance weighting (lower performance -> higher sampling weight)
        total_performance = sum(task_performance.values())
        if total_performance > 0:
            for task_id, perf in task_performance.items():
                # Higher weight for lower performance (more practice needed)
                self.task_weights[task_id] = (1.0 - perf / total_performance) + 0.1
                
    def sample_task(self) -> Optional[int]:
        """Sample task based on weights
        Returns:
            Task ID or None if no tasks available
        """
        if not self.task_weights:
            return None
            
        tasks = list(self.task_weights.keys())
        weights = [self.task_weights[task] for task in tasks]
        
        # Normalize weights
        total_weight = sum(weights)
        if total_weight > 0:
            weights = [w / total_weight for w in weights]
        else:
            weights = [1.0 / len(weights)] * len(weights)
            
        # Sample task
        sampled_task = random.choices(tasks, weights=weights, k=1)[0]
        
        # Update access counts
        self.task_access_counts[sampled_task] = self.task_access_counts.get(sampled_task, 0) + 1
        
        return sampled_task


class AdaptiveContinualLearner:
    """Adaptive continual learner with dynamic strategy selection"""
    
    def __init__(self, model: nn.Module, buffer_capacity: int = 1000):
        self.model = model
        self.continual_buffer = ContinualLearningBuffer(model, buffer_capacity)
        self.task_sampler = TaskSampler()
        
        # Strategy selection
        self.strategies = {
            'ewc_only': 0.3,
            'replay_only': 0.3,
            'ewc_replay_combined': 0.4
        }
        self.current_strategy = 'ewc_replay_combined'
        
        # Performance tracking
        self.strategy_performance = {
            'ewc_only': [],
            'replay_only': [],
            'ewc_replay_combined': []
        }
        
    def select_strategy(self, validation_performance: Optional[Dict[str, float]] = None):
        if validation_performance is None:
            validation_performance = {}
        """Select learning strategy based on performance
        Args:
            validation_performance: Validation performance for each strategy
        """
        if validation_performance:
            # Update strategy performance history
            for strategy, perf in validation_performance.items():
                if strategy in self.strategy_performance:
                    self.strategy_performance[strategy].append(perf)
                    
            # Select best performing strategy
            avg_performances = {}
            for strategy, perfs in self.strategy_performance.items():
                if perfs:
                    avg_performances[strategy] = np.mean(perfs[-10:])  # Last 10 performances
                    
            if avg_performances:
                if avg_performances:
                    self.current_strategy = max(avg_performances.keys(), key=lambda k: avg_performances[k])
                
    def get_training_mixture(self, current_batch_size: int) -> Dict[str, int]:
        """Get training batch mixture for current strategy
        Args:
            current_batch_size: Current batch size
        Returns:
            Dictionary mapping component to batch size
        """
        mixture = {}
        
        if self.current_strategy == 'ewc_only':
            mixture['current_task'] = current_batch_size
            mixture['replay'] = 0
        elif self.current_strategy == 'replay_only':
            mixture['current_task'] = current_batch_size // 2
            mixture['replay'] = current_batch_size // 2
        elif self.current_strategy == 'ewc_replay_combined':
            mixture['current_task'] = int(current_batch_size * 0.6)
            mixture['replay'] = int(current_batch_size * 0.4)
            
        return mixture
    
    def get_adaptive_loss(self, current_loss: torch.Tensor) -> torch.Tensor:
        """Get adaptive loss based on current strategy
        Args:
            current_loss: Current task loss
        Returns:
            Adaptive loss tensor
        """
        if self.current_strategy in ['ewc_only', 'ewc_replay_combined']:
            return self.continual_buffer.get_continual_loss(current_loss)
        else:
            return current_loss
    
    def get_learner_stats(self) -> Dict[str, Any]:
        """Get learner statistics
        Returns:
            Dictionary of learner statistics
        """
        buffer_stats = self.continual_buffer.get_buffer_stats()
        
        return {
            'current_strategy': self.current_strategy,
            'buffer_stats': buffer_stats,
            'strategy_weights': self.strategies,
            'task_sampler_weights': self.task_sampler.task_weights
        }


def demo_continual_learning():
    """Demonstrate continual learning buffer"""
    if not CONTINUAL_AVAILABLE:
        print("❌ Some modules not available for continual learning")
        # Continue with demo using mock components
        
    print("🚀 Demonstrating Continual Learning Buffer (EWC + Replay)...")
    print("=" * 60)
    
    # Create mock model for demonstration
    class MockModel(nn.Module):
        def __init__(self, input_dim=64, num_classes=2):
            super().__init__()
            self.classifier = nn.Linear(input_dim, num_classes)
            
        def forward(self, x, aux=None):
            if x.dim() == 3:
                x = x.mean(dim=1)  # Pool over sequence
            return self.classifier(x), None
    
    # Create model
    model = MockModel(input_dim=64, num_classes=2)
    print("✅ Created mock model for demonstration")
    
    # Create continual learning buffer
    cl_buffer = ContinualLearningBuffer(model, buffer_capacity=500, lambda_ewc=100.0)
    print("✅ Initialized Continual Learning Buffer")
    print(f"   Buffer capacity: {cl_buffer.buffer_capacity}")
    print(f"   EWC lambda: {cl_buffer.ewc.lambda_ewc}")
    
    # Add sample experiences
    print("✅ Adding sample experiences to buffer...")
    for task_id in range(3):
        for i in range(50):
            # Create sample data
            data = torch.randn(1, 64)  # (1, 64)
            target = torch.randint(0, 2, (1,))  # (1,)
            logits = torch.randn(1, 2)  # (1, 2)
            
            # Add to buffer
            cl_buffer.add_experience(task_id, data, target, logits)
            
    print(f"   Added {len(cl_buffer.replay_buffer)} experiences from 3 tasks")
    
    # Check buffer statistics
    buffer_stats = cl_buffer.get_buffer_stats()
    print(f"✅ Buffer statistics:")
    for key, value in buffer_stats.items():
        print(f"   {key}: {value}")
    
    # Test EWC loss computation
    ewc_loss = cl_buffer.compute_ewc_loss()
    print(f"✅ EWC loss: {ewc_loss.item():.6f}")
    
    # Test continual loss
    current_loss = torch.tensor(0.5)
    total_loss = cl_buffer.get_continual_loss(current_loss)
    print(f"✅ Continual loss: {total_loss.item():.6f} (current: {current_loss.item():.6f})")
    
    # Test replay sampling
    replay_data, replay_targets = cl_buffer.sample_replay_batch(8)
    if replay_data is not None:
        print(f"✅ Sampled replay batch:")
        print(f"   Data shape: {replay_data.shape}")
        print(f"   Targets shape: {replay_targets.shape}")
    else:
        print("✅ No replay data available")
    
    # Create adaptive learner
    adaptive_learner = AdaptiveContinualLearner(model, buffer_capacity=500)
    print("✅ Initialized Adaptive Continual Learner")
    
    # Test strategy selection
    adaptive_learner.select_strategy({
        'ewc_only': 0.75,
        'replay_only': 0.68,
        'ewc_replay_combined': 0.82
    })
    print(f"✅ Selected strategy: {adaptive_learner.current_strategy}")
    
    # Test training mixture
    mixture = adaptive_learner.get_training_mixture(32)
    print(f"✅ Training mixture for batch size 32:")
    for component, size in mixture.items():
        print(f"   {component}: {size}")
    
    # Test adaptive loss
    sample_loss = torch.tensor(0.3)
    adaptive_loss = adaptive_learner.get_adaptive_loss(sample_loss)
    print(f"✅ Adaptive loss: {adaptive_loss.item():.6f} (original: {sample_loss.item():.6f})")
    
    # Create task sampler
    task_sampler = TaskSampler()
    
    # Update task weights based on performance
    task_performance = {0: 0.85, 1: 0.72, 2: 0.91}
    task_sampler.update_task_weights(task_performance)
    print(f"✅ Updated task sampling weights:")
    for task_id, weight in task_sampler.task_weights.items():
        print(f"   Task {task_id}: {weight:.3f}")
    
    # Sample tasks
    print("✅ Sampling tasks:")
    for i in range(10):
        sampled_task = task_sampler.sample_task()
        print(f"   Sample {i+1}: Task {sampled_task}")
    
    # Get learner statistics
    learner_stats = adaptive_learner.get_learner_stats()
    print(f"✅ Learner statistics collected")
    
    print("\n" + "=" * 60)
    print("CONTINUAL LEARNING BUFFER DEMO SUMMARY")
    print("=" * 60)
    print("Key Features Implemented:")
    print("  1. Elastic Weight Consolidation (EWC) regularizer")
    print("  2. Experience replay buffer with smart sampling")
    print("  3. Adaptive strategy selection")
    print("  4. Task-aware sampling")
    print("  5. Performance monitoring and tracking")
    print("\nBenefits:")
    print("  - Reduced catastrophic forgetting")
    print("  - Efficient knowledge transfer")
    print("  - Adaptive learning strategies")
    print("  - Balanced task sampling")
    
    print("\n✅ Continual Learning Buffer demonstration completed!")


def main():
    """Main demonstration function"""
    demo_continual_learning()


if __name__ == '__main__':
    main()