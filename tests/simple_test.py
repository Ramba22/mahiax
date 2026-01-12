"""
Simple test for MAHIA OptiCore components
"""

import sys
import os

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

def test_basic_imports():
    """Test basic imports"""
    print("Testing basic imports...")
    
    try:
        # Test importing OptiCore
        from opticore import get_opticore
        print("✅ OptiCore import successful")
        
        # Test importing individual components
        from opticore.core_manager import get_core_manager
        print("✅ CoreManager import successful")
        
        from opticore.memory_allocator import get_memory_allocator
        print("✅ MemoryAllocator import successful")
        
        from opticore.pooling_engine import get_pooling_engine
        print("✅ PoolingEngine import successful")
        
        from opticore.activation_checkpoint import get_activation_checkpoint
        print("✅ ActivationCheckpoint import successful")
        
        from opticore.precision_tuner import get_precision_tuner
        print("✅ PrecisionTuner import successful")
        
        from opticore.telemetry_layer import get_telemetry_layer
        print("✅ TelemetryLayer import successful")
        
        from opticore.energy_controller import get_energy_controller
        print("✅ EnergyController import successful")
        
        from opticore.diagnostics import get_diagnostics
        print("✅ Diagnostics import successful")
        
        from opticore.compatibility import get_compatibility_layer
        print("✅ CompatibilityLayer import successful")
        
        return True
        
    except Exception as e:
        print(f"❌ Import test failed: {e}")
        return False

def test_component_creation():
    """Test component creation"""
    print("\nTesting component creation...")
    
    try:
        # Test creating instances
        from opticore import get_opticore
        opticore = get_opticore()
        print("✅ OptiCore instance created")
        
        from opticore.core_manager import get_core_manager
        core_manager = get_core_manager()
        print("✅ CoreManager instance created")
        
        from opticore.memory_allocator import get_memory_allocator
        memory_allocator = get_memory_allocator()
        print("✅ MemoryAllocator instance created")
        
        from opticore.pooling_engine import get_pooling_engine
        pooling_engine = get_pooling_engine()
        print("✅ PoolingEngine instance created")
        
        return True
        
    except Exception as e:
        print(f"❌ Component creation test failed: {e}")
        return False

def test_basic_functionality():
    """Test basic functionality"""
    print("\nTesting basic functionality...")
    
    try:
        from opticore.memory_allocator import get_memory_allocator
        allocator = get_memory_allocator()
        
        # Test allocation
        buffer = allocator.allocate(1024)  # 1KB
        print(f"✅ Memory allocation successful: {buffer is not None}")
        
        # Test deallocation
        if buffer is not None:
            success = allocator.deallocate(buffer)
            print(f"✅ Memory deallocation successful: {success}")
        
        from opticore.precision_tuner import get_precision_tuner
        tuner = get_precision_tuner()
        
        # Test precision tuning
        gradients = [[0.1, 0.2], [0.05, 0.15]]
        precision = tuner.tune_precision(0.5, gradients)
        print(f"✅ Precision tuning successful: {precision}")
        
        return True
        
    except Exception as e:
        print(f"❌ Basic functionality test failed: {e}")
        return False

def main():
    """Main test function"""
    print("🚀 Running Simple MAHIA OptiCore Test")
    print("=" * 40)
    
    tests = [
        test_basic_imports,
        test_component_creation,
        test_basic_functionality
    ]
    
    passed = 0
    for test in tests:
        if test():
            passed += 1
        else:
            print("❌ Test failed")
    
    print("\n" + "=" * 40)
    print(f"📊 Results: {passed}/{len(tests)} tests passed")
    
    if passed == len(tests):
        print("🎉 All tests passed!")
        return True
    else:
        print("❌ Some tests failed")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)