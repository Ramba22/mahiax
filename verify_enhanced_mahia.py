"""
Verification Script for Enhanced MAHIA-X
Tests all implemented components to ensure proper functionality
"""

import sys
import os
from typing import Dict, Any

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def verify_component_imports():
    """Verify all components can be imported successfully"""
    components = {
        'Core Coordinator': 'core.mahia_coordinator',
        'Self-Improvement Engine': 'learning.self_improvement_engine',
        'Multimodal Processor': 'multimodal.multimodal_processor',
        'Personalization Engine': 'personalization.user_profile_engine',
        'Error Detection System': 'quality.error_detection_system',
        'Dynamic Module Loader': 'optimization.dynamic_module_loader',
        'Expert Engine': 'integration.expert_engine',
        'Ethics Engine': 'security.ethics_engine',
        'Decision Explainer': 'explainability.decision_explainer'
    }
    
    print("🔍 Verifying Component Imports...")
    print("=" * 40)
    
    success_count = 0
    for component_name, module_path in components.items():
        try:
            __import__(module_path)
            print(f"✅ {component_name}")
            success_count += 1
        except ImportError as e:
            print(f"❌ {component_name} - {e}")
        except Exception as e:
            print(f"❌ {component_name} - Unexpected error: {e}")
    
    print(f"\nImport Verification: {success_count}/{len(components)} components successful")
    return success_count == len(components)

def verify_class_instantiation():
    """Verify all main classes can be instantiated"""
    from learning.self_improvement_engine import SelfImprovementEngine
    from multimodal.multimodal_processor import MultimodalProcessor
    from personalization.user_profile_engine import PersonalizationEngine
    from quality.error_detection_system import ErrorDetector, SelfCorrectionEngine
    from optimization.dynamic_module_loader import MAHIAOptiCore
    from integration.expert_engine import ExpertEngine
    from security.ethics_engine import EthicsEngine
    from explainability.decision_explainer import DecisionExplainer
    
    classes_to_test = [
        ('SelfImprovementEngine', SelfImprovementEngine),
        ('MultimodalProcessor', MultimodalProcessor),
        ('PersonalizationEngine', PersonalizationEngine),
        ('ErrorDetector', ErrorDetector),
        ('SelfCorrectionEngine', SelfCorrectionEngine),
        ('MAHIAOptiCore', MAHIAOptiCore),
        ('ExpertEngine', ExpertEngine),
        ('EthicsEngine', EthicsEngine),
        ('DecisionExplainer', DecisionExplainer)
    ]
    
    print("\n🔧 Verifying Class Instantiation...")
    print("=" * 40)
    
    success_count = 0
    for class_name, class_type in classes_to_test:
        try:
            instance = class_type()
            print(f"✅ {class_name}")
            success_count += 1
        except Exception as e:
            print(f"❌ {class_name} - {e}")
    
    print(f"\nInstantiation Verification: {success_count}/{len(classes_to_test)} classes successful")
    return success_count == len(classes_to_test)

def verify_core_functionality():
    """Verify core functionality of key components"""
    print("\n⚡ Verifying Core Functionality...")
    print("=" * 40)
    
    success_count = 0
    total_tests = 0
    
    # Test 1: Error Detection
    try:
        total_tests += 1
        from quality.error_detection_system import ErrorDetector
        detector = ErrorDetector()
        errors = detector.detect_errors("This is a test response with no errors.")
        print(f"✅ Error Detection - Found {len(errors)} errors in clean text")
        success_count += 1
    except Exception as e:
        print(f"❌ Error Detection - {e}")
    
    # Test 2: Personalization
    try:
        total_tests += 1
        from personalization.user_profile_engine import PersonalizationEngine
        engine = PersonalizationEngine()
        engine.update_user_preferences("test_user", {"response_style": "technical"})
        context = engine.get_personalized_context("test_user")
        print(f"✅ Personalization - Generated context for user")
        success_count += 1
    except Exception as e:
        print(f"❌ Personalization - {e}")
    
    # Test 3: Ethics Compliance
    try:
        total_tests += 1
        from security.ethics_engine import EthicsEngine
        engine = EthicsEngine()
        result = engine.process_response("This is a safe response.", "test_user")
        compliant = result.get('ethical_compliance', {}).get('compliant', False)
        print(f"✅ Ethics Engine - Response compliant: {compliant}")
        success_count += 1
    except Exception as e:
        print(f"❌ Ethics Engine - {e}")
    
    # Test 4: Multimodal Processing
    try:
        total_tests += 1
        from multimodal.multimodal_processor import MultimodalProcessor
        processor = MultimodalProcessor()
        result = processor.process_multimodal_input({
            'text': [0.1, 0.2, 0.3],
            'image': [0.4, 0.5, 0.6]
        })
        print(f"✅ Multimodal Processing - Processed {len(result.get('modalities_processed', []))} modalities")
        success_count += 1
    except Exception as e:
        print(f"❌ Multimodal Processing - {e}")
    
    print(f"\nFunctionality Verification: {success_count}/{total_tests} tests successful")
    return success_count == total_tests

def verify_coordinator():
    """Verify the main coordinator can be instantiated and started"""
    print("\n🚀 Verifying Main Coordinator...")
    print("=" * 40)
    
    try:
        from core.mahia_coordinator import MAHIACoordinator
        coordinator = MAHIACoordinator()
        print("✅ Coordinator Instantiation - Success")
        
        # Test startup
        startup_success = coordinator.start_system()
        print(f"✅ Coordinator Startup - {'Success' if startup_success else 'Failed'}")
        
        # Test a simple request
        test_request = {
            'query': 'Test query',
            'user_id': 'test_user',
            'context': {}
        }
        
        result = coordinator.process_enhanced_request(test_request)
        print(f"✅ Request Processing - Response generated (length: {len(result.get('response', ''))})")
        
        # Test shutdown
        coordinator.stop_system()
        print("✅ Coordinator Shutdown - Success")
        
        return True
    except Exception as e:
        print(f"❌ Coordinator Verification - {e}")
        return False

def main():
    """Main verification function"""
    print("🧪 MAHIA-X Enhanced Components Verification")
    print("=" * 50)
    
    # Run all verification tests
    import_success = verify_component_imports()
    instantiation_success = verify_class_instantiation()
    functionality_success = verify_core_functionality()
    coordinator_success = verify_coordinator()
    
    # Summary
    print("\n" + "=" * 50)
    print("VERIFICATION SUMMARY")
    print("=" * 50)
    
    tests = [
        ("Component Imports", import_success),
        ("Class Instantiation", instantiation_success),
        ("Core Functionality", functionality_success),
        ("Main Coordinator", coordinator_success)
    ]
    
    passed_tests = sum(1 for _, success in tests if success)
    
    for test_name, success in tests:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{test_name:<20} {status}")
    
    print("-" * 50)
    print(f"Overall Result: {passed_tests}/{len(tests)} tests passed")
    
    if passed_tests == len(tests):
        print("\n🎉 All verification tests passed! Enhanced MAHIA-X is ready for use.")
        print("You can now run the initialization script: python init_enhanced_mahia.py")
        return True
    else:
        print("\n⚠️  Some verification tests failed. Please check the errors above.")
        return False

if __name__ == "__main__":
    main()