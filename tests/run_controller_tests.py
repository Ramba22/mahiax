"""
Test runner for MAHIA-X controller tests.
"""
import sys
import os

# Add the project root to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

def run_tests():
    """Run all controller tests"""
    try:
        import torch
        import numpy as np
        print("✅ Required dependencies available")
    except ImportError as e:
        print(f"⚠️  Missing dependencies: {e}")
        print("Please install PyTorch and NumPy to run tests")
        return False
        
    try:
        # Import and run tests
        from controller_tests import (
            TestPredictiveStopForecaster,
            TestExtendStop,
            TestGradientEntropyMonitor,
            TestMetaLRPolicyController,
            TestConfidenceTrendBasedLRAdjuster,
            TestExpertLoadBalancerV2,
            TestCurriculumMemorySystem
        )
        import unittest
        
        # Create test suite
        loader = unittest.TestLoader()
        suite = unittest.TestSuite()
        
        # Add all test classes
        test_classes = [
            TestPredictiveStopForecaster,
            TestExtendStop,
            TestGradientEntropyMonitor,
            TestMetaLRPolicyController,
            TestConfidenceTrendBasedLRAdjuster,
            TestExpertLoadBalancerV2,
            TestCurriculumMemorySystem
        ]
        
        for test_class in test_classes:
            suite.addTests(loader.loadTestsFromTestCase(test_class))
            
        # Run tests
        runner = unittest.TextTestRunner(verbosity=2)
        result = runner.run(suite)
        
        # Return success status
        return result.wasSuccessful()
        
    except Exception as e:
        print(f"❌ Error running tests: {e}")
        return False

if __name__ == '__main__':
    success = run_tests()
    if success:
        print("\n✅ All controller tests completed successfully!")
        sys.exit(0)
    else:
        print("\n❌ Some tests failed!")
        sys.exit(1)