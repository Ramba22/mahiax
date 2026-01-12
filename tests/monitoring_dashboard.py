#!/usr/bin/env python3
"""
Monitoring dashboard for the enhanced training controllers
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict

# Import the classes we want to monitor
from modell_V5_MAHIA_HyenaMoE import ExtendStop, GradientEntropyMonitor, AutoLrPrecisionTuner

class TrainingMonitor:
    """Monitor and visualize training metrics"""
    
    def __init__(self):
        self.metrics = defaultdict(list)
        self.controller_states = defaultdict(list)
        
    def log_metric(self, name: str, value: float, step = None):
        """Log a metric value"""
        if step is None:
            step = len(self.metrics[name])
        self.metrics[name].append((step, value))
        
    def log_controller_state(self, controller_name: str, state: dict, step = None):
        """Log controller state"""
        if step is None:
            step = len(self.controller_states[controller_name])
        self.controller_states[controller_name].append((step, state))
        
    def plot_dashboard(self):
        """Create a dashboard visualization"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('MAHIA-V5 Training Dashboard', fontsize=16)
        
        # Plot 1: Loss curves
        if 'train_loss' in self.metrics:
            steps, values = zip(*self.metrics['train_loss'])
            axes[0, 0].plot(steps, values, label='Train Loss', color='blue')
            axes[0, 0].set_title('Training Loss')
            axes[0, 0].set_xlabel('Steps')
            axes[0, 0].set_ylabel('Loss')
            axes[0, 0].legend()
            axes[0, 0].grid(True)
            
        if 'val_loss' in self.metrics:
            steps, values = zip(*self.metrics['val_loss'])
            axes[0, 0].plot(steps, values, label='Validation Loss', color='red')
            axes[0, 0].legend()
            
        # Plot 2: Gradient metrics
        if 'grad_norm' in self.metrics:
            steps, values = zip(*self.metrics['grad_norm'])
            axes[0, 1].plot(steps, values, label='Gradient Norm', color='green')
            axes[0, 1].set_title('Gradient Norm')
            axes[0, 1].set_xlabel('Steps')
            axes[0, 1].set_ylabel('Norm')
            axes[0, 1].legend()
            axes[0, 1].grid(True)
            
        if 'grad_entropy' in self.metrics:
            steps, values = zip(*self.metrics['grad_entropy'])
            axes[0, 1].plot(steps, values, label='Gradient Entropy', color='orange')
            axes[0, 1].legend()
            
        # Plot 3: Learning rate and precision
        if 'learning_rate' in self.metrics:
            steps, values = zip(*self.metrics['learning_rate'])
            ax3 = axes[1, 0]
            ax3.plot(steps, values, label='Learning Rate', color='purple')
            ax3.set_title('Learning Rate')
            ax3.set_xlabel('Steps')
            ax3.set_ylabel('LR')
            ax3.set_yscale('log')
            ax3.legend()
            ax3.grid(True)
            
        # Plot 4: Controller actions
        if 'extendstop_action' in self.metrics:
            steps, actions = zip(*self.metrics['extendstop_action'])
            action_values = [1 if a == 'extend' else 2 if a == 'stop' else 0 for a in actions]
            axes[1, 1].scatter(steps, action_values, c=action_values, cmap='viridis', alpha=0.7)
            axes[1, 1].set_title('ExtendStop Actions')
            axes[1, 1].set_xlabel('Steps')
            axes[1, 1].set_ylabel('Action (0=wait, 1=extend, 2=stop)')
            axes[1, 1].set_yticks([0, 1, 2])
            axes[1, 1].set_yticklabels(['Wait', 'Extend', 'Stop'])
            axes[1, 1].grid(True)
            
        plt.tight_layout()
        plt.savefig('training_dashboard.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("Dashboard saved as 'training_dashboard.png'")
        
    def generate_report(self):
        """Generate a summary report"""
        print("\n" + "="*50)
        print("TRAINING MONITORING REPORT")
        print("="*50)
        
        # Training metrics summary
        if 'train_loss' in self.metrics:
            final_train_loss = self.metrics['train_loss'][-1][1]
            print(f"Final Train Loss: {final_train_loss:.6f}")
            
        if 'val_loss' in self.metrics:
            final_val_loss = self.metrics['val_loss'][-1][1]
            print(f"Final Validation Loss: {final_val_loss:.6f}")
            
        # Controller actions summary
        if 'extendstop_action' in self.metrics:
            actions = [a for _, a in self.metrics['extendstop_action']]
            extend_count = actions.count('extend')
            stop_count = actions.count('stop')
            print(f"ExtendStop Extensions Used: {extend_count}")
            print(f"ExtendStop Stops: {stop_count}")
            
        if 'learning_rate' in self.metrics:
            initial_lr = self.metrics['learning_rate'][0][1]
            final_lr = self.metrics['learning_rate'][-1][1]
            print(f"Learning Rate: {initial_lr:.2e} → {final_lr:.2e}")
            
        print("="*50)

def demo_monitoring():
    """Demonstrate monitoring with simulated training"""
    print("Training Monitoring Dashboard Demo")
    print("=" * 35)
    
    # Initialize monitor and controllers
    monitor = TrainingMonitor()
    extend_stop = ExtendStop(patience=5, min_delta=1e-4)
    gradient_monitor = GradientEntropyMonitor(window_size=3, entropy_drop_threshold=0.25)
    auto_tuner = AutoLrPrecisionTuner(initial_lr=1e-3, min_lr=1e-7)
    
    # Simulate training
    losses = [1.0, 0.9, 0.8, 0.7, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 
              0.5, 0.4, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3]
    
    for step, loss in enumerate(losses):
        # Log loss
        monitor.log_metric('train_loss', loss, step)
        
        # ExtendStop check
        extend_action = extend_stop(loss)
        monitor.log_metric('extendstop_action', extend_action['action'], step)
        monitor.log_controller_state('extendstop', extend_action, step)
        
        # Simulate gradient entropy (in real training, this would be computed from actual gradients)
        entropy = 1.0 - (step / len(losses)) * 0.5  # Simulate decreasing entropy
        monitor.log_metric('grad_entropy', entropy, step)
        
        # Auto tuning
        tuner_result = auto_tuner.adjust_lr_and_precision(None, loss, loss_improving=(step < 5))
        monitor.log_metric('learning_rate', tuner_result['lr'], step)
        monitor.log_controller_state('autotuner', tuner_result, step)
        
        if step % 5 == 0:
            print(f"Step {step}: Loss={loss:.4f}, LR={tuner_result['lr']:.2e}, "
                  f"Action={extend_action['action']}")
    
    # Generate dashboard and report
    try:
        monitor.plot_dashboard()
    except Exception as e:
        print(f"Note: Could not generate dashboard plot: {e}")
        print("This is expected if matplotlib is not fully configured in this environment.")
    
    monitor.generate_report()
    print("\n✅ Monitoring demo completed!")

def main():
    """Main function"""
    try:
        demo_monitoring()
        print("\n" + "=" * 35)
        print("🎉 Monitoring Dashboard Ready!")
        print("🚀 Use this to track your MAHIA-V5 training!")
    except Exception as e:
        print(f"\n❌ Demo failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

if __name__ == "__main__":
    main()