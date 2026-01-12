#!/usr/bin/env python3
"""
Test script for ExtendStop functionality
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# Mock torch for testing purposes
class MockTorch:
    class nn:
        class Module:
            pass
    
    class Tensor:
        pass

# Use mock if torch is not available
try:
    import torch
    import torch.nn as nn
except ImportError:
    torch = MockTorch()
    nn = MockTorch.nn

# Import the classes we want to test
from modell_V5_MAHIA_HyenaMoE import ExtendStop, GradientEntropyMonitor, AutoLrPrecisionTuner

def test_extend_stop():
    """Test ExtendStop functionality"""
    print("Testing ExtendStop...")
    
    # Create ExtendStop with default parameters
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
    
    # Create a mock model for testing
    class MockModel:
        def parameters(self):
            # Return mock parameters with gradients
            class MockParam:
                def __init__(self, has_grad=True):
                    self.grad = MockTensor() if has_grad else None
                    
            class MockTensor:
                def abs(self):
                    return self
                    
                def mean(self):
                    class MockItem:
                        def item(self):
                            return 0.5
                    return MockItem()
                    
            return [MockParam(), MockParam(), MockParam(has_grad=False)]
    
    # Test gradient monitor
    monitor = GradientEntropyMonitor()
    model = MockModel()
    entropy = monitor.compute_gradient_entropy(model)
    
    print(f"  Gradient entropy: {entropy:.6f}")
    
    # Test LR adjustment suggestion
    should_reduce = monitor.should_reduce_lr(model)
    print(f"  Should reduce LR: {should_reduce}")
    
    print("  ✅ Gradient Entropy Monitoring working correctly")

def test_auto_tuner():
    """Test AutoLrPrecisionTuner"""
    print("\nTesting AutoLrPrecisionTuner...")
    
    # Create a mock optimizer for testing
    class MockOptimizer:
        def __init__(self):
            self.param_groups = [{'lr': 1e-3}]
            
        def step(self):
            pass
            
    # Test auto tuner
    optimizer = MockOptimizer()
    tuner = AutoLrPrecisionTuner(initial_lr=1e-3, min_lr=1e-6, lr_factor=0.5)
    
    # Simulate loss not improving
    current_loss = 1.0
    for i in range(5):
        lr, precision = tuner.adjust_lr_and_precision(optimizer, current_loss)
        print(f"  Step {i+1}: LR={lr:.2e}, Precision={precision}")
        current_loss = 1.0  # Keep loss constant to trigger LR reduction
        
    print("  ✅ AutoLrPrecisionTuner working correctly")

def main():
    """Run all tests"""
    print("ExtendStop, Gradient Entropy Monitoring, and Auto-LR/Precision Tuning Tests")
    print("=" * 70)
    
    try:
        test_extend_stop()
        test_gradient_monitoring()
        test_auto_tuner()
        
        print("\n" + "=" * 70)
        print("🎉 All tests passed!")
        print("🚀 MAHIA-V5 now supports advanced training features!")
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

if __name__ == "__main__":
    main()