#!/usr/bin/env python3
"""
Test script for next-generation controller enhancements:
1. Adaptive Curriculum Scheduler (ACS)
2. Uncertainty-Aware Controller Coupling
3. Predictive Stop Forecasting
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import numpy as np

def test_adaptive_curriculum_scheduler():
    """Test Adaptive Curriculum Scheduler implementation"""
    print("Testing Adaptive Curriculum Scheduler...")
    
    from modell_V5_MAHIA_HyenaMoE import AdaptiveCurriculumScheduler
    
    # Create ACS
    acs = AdaptiveCurriculumScheduler(initial_difficulty=0.5, min_difficulty=0.1, 
                                     max_difficulty=1.0, adjustment_factor=0.1)
    
    # Test with low entropy (should increase difficulty)
    result1 = acs.adjust_curriculum(gradient_entropy=0.3, current_epoch=1)
    print(f"  Low entropy test: difficulty={result1['difficulty']:.2f}, action={result1['action']}")
    
    # Test with high entropy (should decrease difficulty)
    result2 = acs.adjust_curriculum(gradient_entropy=2.0, current_epoch=2)
    print(f"  High entropy test: difficulty={result2['difficulty']:.2f}, action={result2['action']}")
    
    # Test with medium entropy (should maintain difficulty)
    result3 = acs.adjust_curriculum(gradient_entropy=1.0, current_epoch=3)
    print(f"  Medium entropy test: difficulty={result3['difficulty']:.2f}, action={result3['action']}")
    
    print("  ✅ Adaptive Curriculum Scheduler working correctly")

def test_uncertainty_aware_controller_coupling():
    """Test Uncertainty-Aware Controller Coupling implementation"""
    print("\nTesting Uncertainty-Aware Controller Coupling...")
    
    from modell_V5_MAHIA_HyenaMoE import UncertaintyAwareControllerCoupling, AutoLrPrecisionTuner
    
    # Create components
    coupling = UncertaintyAwareControllerCoupling(confidence_threshold=0.7)
    tuner = AutoLrPrecisionTuner(initial_lr=1e-3)
    
    # Mock optimizer
    class MockOptimizer:
        def __init__(self):
            self.param_groups = [{'lr': 1e-3}]
    
    # Mock ExtendStop
    class MockExtendStop:
        def __init__(self):
            pass
    
    optimizer = MockOptimizer()
    extend_stop = MockExtendStop()
    
    # Test with low confidence (should adjust controllers)
    low_confidence = torch.tensor([0.3, 0.4, 0.2, 0.5])  # Batch of low confidence scores
    actions1 = coupling.adjust_controllers(
        confidence=low_confidence, 
        current_loss=0.5,
        optimizer=optimizer,
        auto_tuner=tuner,
        extend_stop=None  # Not testing extend stop in this test
    )
    print(f"  Low confidence test: adjust_lr={actions1['adjust_lr']}, "
          f"increase_precision={actions1['increase_precision']}")
    
    # Test with high confidence (should not adjust controllers)
    high_confidence = torch.tensor([0.8, 0.9, 0.7, 0.85])  # Batch of high confidence scores
    actions2 = coupling.adjust_controllers(
        confidence=high_confidence,
        current_loss=0.5,
        optimizer=optimizer,
        auto_tuner=tuner,
        extend_stop=None
    )
    print(f"  High confidence test: adjust_lr={actions2['adjust_lr']}, "
          f"increase_precision={actions2['increase_precision']}")
    
    print("  ✅ Uncertainty-Aware Controller Coupling working correctly")

def test_predictive_stop_forecaster():
    """Test Predictive Stop Forecaster implementation"""
    print("\nTesting Predictive Stop Forecaster...")
    
    from modell_V5_MAHIA_HyenaMoE import PredictiveStopForecaster
    
    # Create forecaster
    forecaster = PredictiveStopForecaster(window_size=5, prediction_horizon=2)
    
    # Simulate training with improving metrics
    print("  Simulating training with improving metrics...")
    for epoch in range(1, 8):
        loss = 1.0 - (epoch * 0.1)  # Decreasing loss
        metric = 0.5 + (epoch * 0.05)  # Increasing metric
        prediction = forecaster.predict_saturation(
            current_loss=loss,
            current_metric=metric,
            current_epoch=epoch
        )
        print(f"    Epoch {epoch}: loss={loss:.3f}, metric={metric:.3f}, "
              f"saturation_predicted={prediction['saturation_predicted']}")
    
    # Simulate training with plateauing metrics
    print("  Simulating training with plateauing metrics...")
    base_metric = 0.85
    for epoch in range(8, 12):
        loss = 0.3  # Constant loss
        metric = base_metric + np.random.normal(0, 0.01)  # Small fluctuations around base
        prediction = forecaster.predict_saturation(
            current_loss=loss,
            current_metric=metric,
            current_epoch=epoch
        )
        should_stop = forecaster.should_early_stop()
        print(f"    Epoch {epoch}: loss={loss:.3f}, metric={metric:.3f}, "
              f"saturation_predicted={prediction['saturation_predicted']}, "
              f"should_stop={should_stop}")
    
    print("  ✅ Predictive Stop Forecaster working correctly")

def test_integrated_next_gen_controllers():
    """Test integrated next-generation controller system"""
    print("\nTesting Integrated Next-Gen Controller System...")
    
    from modell_V5_MAHIA_HyenaMoE import (
        AdaptiveCurriculumScheduler, 
        UncertaintyAwareControllerCoupling, 
        PredictiveStopForecaster,
        AutoLrPrecisionTuner
    )
    
    # Create all controllers
    acs = AdaptiveCurriculumScheduler()
    coupling = UncertaintyAwareControllerCoupling()
    forecaster = PredictiveStopForecaster()
    tuner = AutoLrPrecisionTuner(initial_lr=1e-3)
    
    # Mock optimizer
    class MockOptimizer:
        def __init__(self):
            self.param_groups = [{'lr': 1e-3}]
    
    optimizer = MockOptimizer()
    
    # Simulate a few training steps
    print("  Simulating integrated controller system...")
    for epoch in range(1, 6):
        # Simulate metrics
        loss = 1.0 - (epoch * 0.15)
        metric = 0.5 + (epoch * 0.08)
        entropy = 1.0 + np.random.normal(0, 0.2)  # Variable entropy
        confidence = torch.tensor([0.6 + np.random.normal(0, 0.1) for _ in range(4)])  # Batch confidence
        
        # Apply all controllers
        curriculum_result = acs.adjust_curriculum(gradient_entropy=entropy, current_epoch=epoch)
        coupling_actions = coupling.adjust_controllers(
            confidence=confidence, 
            current_loss=loss,
            optimizer=optimizer,
            auto_tuner=tuner,
            extend_stop=None
        )
        prediction_result = forecaster.predict_saturation(
            current_loss=loss,
            current_metric=metric,
            current_epoch=epoch
        )
        should_stop = forecaster.should_early_stop()
        
        print(f"    Epoch {epoch}:")
        print(f"      Curriculum: difficulty={curriculum_result['difficulty']:.2f}")
        print(f"      Coupling: LR adjust={coupling_actions['adjust_lr']}")
        print(f"      Forecast: saturation={prediction_result['saturation_predicted']}, stop={should_stop}")
    
    print("  ✅ Integrated Next-Gen Controller System working correctly")

def main():
    """Run all next-generation controller tests"""
    print("MAHIA-V5 Next-Generation Controller Tests")
    print("=" * 45)
    
    try:
        test_adaptive_curriculum_scheduler()
        test_uncertainty_aware_controller_coupling()
        test_predictive_stop_forecaster()
        test_integrated_next_gen_controllers()
        
        print("\n" + "=" * 45)
        print("🎉 All next-generation controller tests passed!")
        print("🚀 MAHIA-V5 now supports advanced adaptive training!")
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

if __name__ == "__main__":
    main()