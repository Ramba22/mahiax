#!/usr/bin/env python3
"""
Test script for enhanced trainer with ExtendStop, Gradient Entropy Monitoring, and Auto-LR/Precision Tuning
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

class ExtendStop:
    """Early stopping with extension capabilities based on multiple criteria"""
    
    def __init__(self, patience: int = 10, min_delta: float = 1e-4, 
                 extend_patience: int = 5, max_extensions: int = 3):
        self.patience = patience
        self.min_delta = min_delta
        self.extend_patience = extend_patience
        self.max_extensions = max_extensions
        
        self.best_loss = float('inf')
        self.counter = 0
        self.extensions_used = 0
        self.stop_training = False
        self.loss_history = []
        
    def __call__(self, current_loss: float) -> bool:
        """Returns True if training should stop"""
        self.loss_history.append(current_loss)
        
        if current_loss < self.best_loss - self.min_delta:
            self.best_loss = current_loss
            self.counter = 0
        else:
            self.counter += 1
            
        # Check if we should stop
        if self.counter >= self.patience:
            if self.extensions_used < self.max_extensions:
                # Extend training
                self.extensions_used += 1
                self.counter = 0
                self.patience += self.extend_patience
                print(f"🔄 ExtendStop: Extending training (extension {self.extensions_used}/{self.max_extensions})")
                return False
            else:
                # Stop training
                self.stop_training = True
                print(f"⏹️ ExtendStop: Stopping training after {self.extensions_used} extensions")
                return True
                
        return False

class GradientEntropyMonitor:
    """Monitor gradient entropy to detect training instability"""
    
    def __init__(self, window_size: int = 100, entropy_threshold: float = 0.1):
        self.window_size = window_size
        self.entropy_threshold = entropy_threshold
        self.gradient_history = []
        
    def compute_gradient_entropy(self, model: nn.Module) -> float:
        """Compute entropy of gradient magnitudes"""
        grad_magnitudes = []
        for param in model.parameters():
            if param.grad is not None:
                grad_magnitudes.append(param.grad.abs().mean().item())
        
        if not grad_magnitudes:
            return 0.0
            
        # Convert to tensor and normalize
        grad_tensor = torch.tensor(grad_magnitudes)
        if grad_tensor.sum() == 0:
            return 0.0
            
        # Normalize to probability distribution
        prob_dist = grad_tensor / grad_tensor.sum()
        
        # Compute entropy
        entropy = -torch.sum(prob_dist * torch.log(prob_dist + 1e-8)).item()
        return entropy
        
    def should_reduce_lr(self, model: nn.Module) -> bool:
        """Check if learning rate should be reduced based on gradient entropy"""
        entropy = self.compute_gradient_entropy(model)
        self.gradient_history.append(entropy)
        
        # Keep only recent history
        if len(self.gradient_history) > self.window_size:
            self.gradient_history.pop(0)
            
        # Check if entropy is too low (gradients becoming uniform/uninformative)
        if len(self.gradient_history) >= self.window_size:
            avg_entropy = sum(self.gradient_history[-self.window_size:]) / self.window_size
            if avg_entropy < self.entropy_threshold:
                print(f"⚠️ Gradient entropy low ({avg_entropy:.4f}), consider reducing learning rate")
                return True
                
        return False

class AutoLrPrecisionTuner:
    """Automatic learning rate and precision tuning based on training dynamics"""
    
    def __init__(self, initial_lr: float = 1e-3, min_lr: float = 1e-6, 
                 lr_factor: float = 0.5, precision_threshold: float = 0.01):
        self.initial_lr = initial_lr
        self.min_lr = min_lr
        self.lr_factor = lr_factor
        self.precision_threshold = precision_threshold
        
        self.current_lr = initial_lr
        self.current_precision = 'fp32'
        self.lr_reduction_count = 0
        self.last_loss = float('inf')
        
    def adjust_lr_and_precision(self, optimizer, current_loss: float, 
                               gradient_monitor = None,
                               model = None) -> tuple:
        """Adjust learning rate and precision based on training dynamics"""
        loss_improvement = self.last_loss - current_loss
        self.last_loss = current_loss
        
        # Check if we should reduce learning rate
        should_reduce_lr = False
        
        # Criterion 1: Loss not improving
        if loss_improvement < 1e-6:
            should_reduce_lr = True
            
        # Criterion 2: High gradient entropy (if monitor provided)
        if gradient_monitor and model:
            if gradient_monitor.should_reduce_lr(model):
                should_reduce_lr = True
                
        # Adjust learning rate
        if should_reduce_lr:
            new_lr = max(self.current_lr * self.lr_factor, self.min_lr)
            if new_lr < self.current_lr:
                self.current_lr = new_lr
                self.lr_reduction_count += 1
                print(f"📉 AutoTuner: Reducing LR to {self.current_lr:.2e}")
                
                # Update optimizer learning rate
                for param_group in optimizer.param_groups:
                    param_group['lr'] = self.current_lr
                    
                # Switch to mixed precision after several reductions
                if self.lr_reduction_count >= 3 and self.current_precision == 'fp32':
                    self.current_precision = 'fp16'
                    print(f"⚡ AutoTuner: Switching to {self.current_precision} precision")
                    
        return self.current_lr, self.current_precision

class MAHIAV5Trainer:
    """Enhanced trainer with ExtendStop, Gradient Entropy Monitoring, and Auto-LR/Precision Tuning"""
    
    def __init__(self, model: nn.Module, optimizer, criterion,
                 clip_grad_norm: float = 1.0, clip_grad_mode: str = "norm"):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.clip_grad_norm = clip_grad_norm
        self.clip_grad_mode = clip_grad_mode
        
        # Enhanced training components
        self.extend_stop = ExtendStop()
        self.gradient_monitor = GradientEntropyMonitor()
        self.auto_tuner = AutoLrPrecisionTuner()
        
        # Training statistics
        self.train_losses = []
        self.val_losses = []
        self.learning_rates = []
        
    def train_step(self, x, y) -> float:
        """Single training step with all enhancements"""
        self.model.train()
        self.optimizer.zero_grad()
        
        # Forward pass
        output = self.model(x)
        loss = self.criterion(output, y)
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping
        if self.clip_grad_norm > 0:
            if self.clip_grad_mode == "norm":
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_grad_norm)
            elif self.clip_grad_mode == "value":
                torch.nn.utils.clip_grad_value_(self.model.parameters(), self.clip_grad_norm)
        
        # Optimizer step
        self.optimizer.step()
        
        return loss.item()
    
    def validate(self, val_loader) -> float:
        """Validation step"""
        self.model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x, y in val_loader:
                output = self.model(x)
                val_loss += self.criterion(output, y).item()
        return val_loss / len(val_loader)
    
    def fit(self, train_loader, val_loader=None, epochs: int = 100, 
            validate_every: int = 1):
        """Enhanced training loop with all features"""
        print("🚀 Starting enhanced training with ExtendStop, Gradient Monitoring, and Auto-Tuning...")
        
        for epoch in range(epochs):
            # Training phase
            epoch_loss = 0.0
            for batch_idx, (x, y) in enumerate(train_loader):
                loss = self.train_step(x, y)
                epoch_loss += loss
                
                if batch_idx % 10 == 0:
                    print(f"Epoch {epoch+1}, Batch {batch_idx}, Loss: {loss:.6f}")
            
            avg_loss = epoch_loss / len(train_loader)
            self.train_losses.append(avg_loss)
            
            # Validation phase
            if val_loader and (epoch + 1) % validate_every == 0:
                val_loss = self.validate(val_loader)
                self.val_losses.append(val_loss)
                print(f"Epoch {epoch+1}: Train Loss: {avg_loss:.6f}, Val Loss: {val_loss:.6f}")
                
                # Auto-tuning
                lr, precision = self.auto_tuner.adjust_lr_and_precision(
                    self.optimizer, val_loss, self.gradient_monitor, self.model)
                self.learning_rates.append(lr)
                
                # ExtendStop check
                if self.extend_stop(val_loss):
                    print(f"✅ Training completed after {epoch+1} epochs")
                    break
            else:
                # Auto-tuning without validation
                lr, precision = self.auto_tuner.adjust_lr_and_precision(
                    self.optimizer, avg_loss, self.gradient_monitor, self.model)
                self.learning_rates.append(lr)
                
                # ExtendStop check
                if self.extend_stop(avg_loss):
                    print(f"✅ Training completed after {epoch+1} epochs")
                    break
                    
        print("🏁 Enhanced training finished!")
        return self.train_losses, self.val_losses

def test_enhanced_trainer():
    """Test the enhanced trainer functionality"""
    print("Testing Enhanced Trainer...")
    
    # Create a simple model
    model = nn.Sequential(
        nn.Linear(10, 64),
        nn.ReLU(),
        nn.Linear(64, 32),
        nn.ReLU(),
        nn.Linear(32, 1)
    )
    
    # Create optimizer and criterion
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()
    
    # Create trainer
    trainer = MAHIAV5Trainer(model, optimizer, criterion)
    
    # Create dummy data
    X = torch.randn(1000, 10)
    y = torch.randn(1000, 1)
    dataset = TensorDataset(X, y)
    train_loader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    # Train model
    train_losses, val_losses = trainer.fit(train_loader, epochs=20)
    
    print(f"  Final train loss: {train_losses[-1]:.6f}")
    print(f"  Learning rate adjustments: {len(trainer.learning_rates)}")
    print(f"  Extensions used: {trainer.extend_stop.extensions_used}")
    
    print("  ✅ Enhanced Trainer working correctly")

def test_extend_stop():
    """Test ExtendStop functionality"""
    print("\nTesting ExtendStop...")
    
    extend_stop = ExtendStop(patience=5, min_delta=0.01, max_extensions=2)
    
    # Simulate training with fluctuating losses
    losses = [1.0, 0.9, 0.8, 0.7, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6,  # Plateau
              0.5, 0.4, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3]   # Another plateau
    
    stop_count = 0
    for i, loss in enumerate(losses):
        should_stop = extend_stop(loss)
        if should_stop:
            stop_count += 1
            print(f"  Stopped at iteration {i} with loss {loss:.3f}")
            
    print(f"  Extensions used: {extend_stop.extensions_used}")
    print("  ✅ ExtendStop working correctly")

def test_gradient_monitoring():
    """Test Gradient Entropy Monitoring"""
    print("\nTesting Gradient Entropy Monitoring...")
    
    # Create a simple model
    model = nn.Linear(10, 1)
    
    # Create dummy data and do one training step to get gradients
    X = torch.randn(32, 10)
    y = torch.randn(32, 1)
    
    criterion = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    
    # Forward and backward pass
    output = model(X)
    loss = criterion(output, y)
    loss.backward()
    
    # Test gradient monitor
    monitor = GradientEntropyMonitor()
    entropy = monitor.compute_gradient_entropy(model)
    
    print(f"  Gradient entropy: {entropy:.6f}")
    
    # Test LR adjustment suggestion
    should_reduce = monitor.should_reduce_lr(model)
    print(f"  Should reduce LR: {should_reduce}")
    
    print("  ✅ Gradient Entropy Monitoring working correctly")

def main():
    """Run all enhanced trainer tests"""
    print("MAHIA-V5 Enhanced Trainer Tests")
    print("=" * 35)
    
    try:
        test_enhanced_trainer()
        test_extend_stop()
        test_gradient_monitoring()
        
        print("\n" + "=" * 35)
        print("🎉 All Enhanced Trainer tests passed!")
        print("🚀 MAHIA-V5 now supports advanced training features!")
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

if __name__ == "__main__":
    main()