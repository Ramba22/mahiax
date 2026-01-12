"""
Test suite for MAHIA OptiCore components
"""

import sys
import os
import time

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

def test_opticore_initialization():
    """Test OptiCore initialization"""
    print("🧪 Testing OptiCore initialization...")
    
    try:
        from opticore import get_opticore, initialize_opticore, shutdown_opticore
        
        # Get OptiCore instance
        opticore = get_opticore()
        print("✅ OptiCore instance created")
        
        # Initialize
        success = initialize_opticore(start_monitoring=False)
        assert success, "OptiCore initialization failed"
        print("✅ OptiCore initialized")
        
        # Get system status
        status = opticore.get_system_status()
        assert isinstance(status, dict), "System status should be a dict"
        print("✅ System status retrieved")
        
        # Shutdown
        shutdown_opticore()
        print("✅ OptiCore shutdown")
        
        return True
        
    except Exception as e:
        print(f"❌ OptiCore initialization test failed: {e}")
        return False

def test_memory_management():
    """Test memory management components"""
    print("🧪 Testing memory management...")
    
    try:
        from opticore import opticore_memory, opticore_pooling
        
        # Test memory allocator
        allocator = opticore_memory()
        assert allocator is not None, "Memory allocator should not be None"
        print("✅ Memory allocator accessible")
        
        # Test pooling engine
        pooling = opticore_pooling()
        assert pooling is not None, "Pooling engine should not be None"
        print("✅ Pooling engine accessible")
        
        return True
        
    except Exception as e:
        print(f"❌ Memory management test failed: {e}")
        return False

def test_precision_tuning():
    """Test precision tuning components"""
    print("🧪 Testing precision tuning...")
    
    try:
        from opticore import opticore_precision
        
        # Test precision tuner
        tuner = opticore_precision()
        assert tuner is not None, "Precision tuner should not be None"
        print("✅ Precision tuner accessible")
        
        # Test precision tuning
        gradients = [[0.1, 0.2, 0.3], [0.05, 0.15, 0.25]]
        precision = tuner.tune_precision(0.5, gradients)
        assert isinstance(precision, str), "Precision should be a string"
        print(f"✅ Precision tuning works: {precision}")
        
        return True
        
    except Exception as e:
        print(f"❌ Precision tuning test failed: {e}")
        return False

def test_telemetry():
    """Test telemetry components"""
    print("🧪 Testing telemetry...")
    
    try:
        from opticore import opticore_telemetry
        
        # Test telemetry layer
        telemetry = opticore_telemetry()
        assert telemetry is not None, "Telemetry layer should not be None"
        print("✅ Telemetry layer accessible")
        
        # Test metric recording
        telemetry.record_metric("test_metric", 42.5)
        value = telemetry.get_latest_metric("test_metric")
        assert value is not None, "Should retrieve recorded metric"
        print("✅ Metric recording works")
        
        return True
        
    except Exception as e:
        print(f"❌ Telemetry test failed: {e}")
        return False

def test_energy_management():
    """Test energy management components"""
    print("🧪 Testing energy management...")
    
    try:
        from opticore import opticore_energy
        
        # Test energy controller
        energy = opticore_energy()
        assert energy is not None, "Energy controller should not be None"
        print("✅ Energy controller accessible")
        
        # Test optimization step
        result = energy.optimize_step(batch_time=2.0, batch_size=32, current_power=100.0)
        assert isinstance(result, dict), "Optimization result should be a dict"
        print("✅ Energy optimization works")
        
        return True
        
    except Exception as e:
        print(f"❌ Energy management test failed: {e}")
        return False

def test_diagnostics():
    """Test diagnostics components"""
    print("🧪 Testing diagnostics...")
    
    try:
        from opticore import opticore_diagnostics
        
        # Test diagnostics
        diagnostics = opticore_diagnostics()
        assert diagnostics is not None, "Diagnostics should not be None"
        print("✅ Diagnostics accessible")
        
        # Test logging
        diagnostics.log_message("INFO", "Test message", "test_component")
        logs = diagnostics.get_recent_logs(1)
        assert len(logs) > 0, "Should have logged messages"
        print("✅ Diagnostic logging works")
        
        return True
        
    except Exception as e:
        print(f"❌ Diagnostics test failed: {e}")
        return False

def test_compatibility_layer():
    """Test compatibility layer"""
    print("🧪 Testing compatibility layer...")
    
    try:
        from opticore.compatibility import get_compatibility_layer
        
        # Get compatibility layer
        compat = get_compatibility_layer()
        assert compat is not None, "Compatibility layer should not be None"
        print("✅ Compatibility layer accessible")
        
        # Test memory allocation
        buffer = compat.memory_allocate(1024)
        assert buffer is not None, "Should allocate memory"
        success = compat.memory_deallocate(buffer)
        assert success, "Should deallocate memory"
        print("✅ Compatibility memory management works")
        
        return True
        
    except Exception as e:
        print(f"❌ Compatibility layer test failed: {e}")
        return False

def run_all_tests():
    """Run all OptiCore tests"""
    print("🚀 Running MAHIA OptiCore Test Suite")
    print("=" * 50)
    
    tests = [
        test_opticore_initialization,
        test_memory_management,
        test_precision_tuning,
        test_telemetry,
        test_energy_management,
        test_diagnostics,
        test_compatibility_layer
    ]
    
    passed = 0
    failed = 0
    
    start_time = time.time()
    
    for test in tests:
        try:
            if test():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"💥 Test {test.__name__} crashed: {e}")
            failed += 1
        print()
    
    end_time = time.time()
    
    print("=" * 50)
    print(f"📊 Test Results: {passed} passed, {failed} failed")
    print(f"⏱️  Total time: {end_time - start_time:.2f} seconds")
    
    if failed == 0:
        print("🎉 All tests passed! OptiCore is ready for use.")
        return True
    else:
        print("❌ Some tests failed. Please check the implementation.")
        return False

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)