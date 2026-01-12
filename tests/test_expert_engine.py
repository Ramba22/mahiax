"""
Test suite for MAHIA Expert Engine components
"""

import sys
import os
import time

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

def test_expert_registry():
    """Test Expert Registry functionality"""
    print("🧪 Testing Expert Registry...")
    
    try:
        from expert_registry import get_expert_registry
        
        # Get registry instance
        registry = get_expert_registry()
        print("✅ ExpertRegistry instance created")
        
        # Register an expert
        expert_id = registry.register_expert(
            capabilities=["nlp", "translation"],
            embedding_signature=[0.1, 0.2, 0.3, 0.4],
            device="cuda:0",
            memory_footprint_mb=1024.0
        )
        print(f"✅ Expert registered: {expert_id}")
        
        # Get expert
        expert = registry.get_expert(expert_id)
        assert expert is not None, "Expert should not be None"
        assert expert.expert_id == expert_id, "Expert ID should match"
        print("✅ Expert retrieval works")
        
        # Update expert metadata
        success = registry.update_expert_metadata(expert_id, health_status="degraded")
        assert success, "Metadata update should succeed"
        print("✅ Expert metadata update works")
        
        # Get active experts
        active_experts = registry.get_active_experts()
        assert len(active_experts) > 0, "Should have active experts"
        print("✅ Active experts retrieval works")
        
        return True
        
    except Exception as e:
        print(f"❌ Expert Registry test failed: {e}")
        return False

def test_contextual_router():
    """Test Contextual Router functionality"""
    print("🧪 Testing Contextual Router...")
    
    try:
        from contextual_router import get_contextual_router
        from expert_registry import get_expert_registry
        
        # Register some experts first
        registry = get_expert_registry()
        expert1_id = registry.register_expert(
            capabilities=["nlp", "translation"],
            embedding_signature=[0.1, 0.2, 0.3, 0.4],
            device="cuda:0",
            memory_footprint_mb=1024.0
        )
        
        expert2_id = registry.register_expert(
            capabilities=["vision", "classification"],
            embedding_signature=[0.4, 0.3, 0.2, 0.1],
            device="cuda:1",
            memory_footprint_mb=2048.0
        )
        
        # Get router instance
        router = get_contextual_router()
        print("✅ ContextualRouter instance created")
        
        # Test routing
        test_inputs = [0.15, 0.25, 0.35, 0.45]
        result = router.route(test_inputs, k=2)
        assert result is not None, "Routing result should not be None"
        print("✅ Basic routing works")
        
        # Test different modes
        from contextual_router import RoutingMode
        result_topk = router.route(test_inputs, k=1, mode=RoutingMode.TOP_K)
        result_reflective = router.route(test_inputs, k=1, mode=RoutingMode.REFLECTIVE)
        assert result_topk.expert_ids, "Top-K routing should return experts"
        assert result_reflective.expert_ids, "Reflective routing should return experts"
        print("✅ Multiple routing modes work")
        
        return True
        
    except Exception as e:
        print(f"❌ Contextual Router test failed: {e}")
        return False

def test_diversity_controller():
    """Test Diversity Controller functionality"""
    print("🧪 Testing Diversity Controller...")
    
    try:
        from diversity_controller import get_diversity_controller
        
        # Get controller instance
        controller = get_diversity_controller()
        print("✅ DiversityController instance created")
        
        # Test diversity loss computation
        outputs = [
            [0.1, 0.2, 0.3, 0.4],
            [0.2, 0.3, 0.4, 0.5],
            [0.8, 0.7, 0.6, 0.5]
        ]
        
        loss = controller.compute_diversity_loss(outputs)
        assert isinstance(loss, float), "Diversity loss should be a float"
        print(f"✅ Diversity loss computation works: {loss:.3f}")
        
        # Test metrics calculation
        metrics = controller.calculate_diversity_metrics()
        assert metrics is not None, "Metrics should not be None"
        print("✅ Diversity metrics calculation works")
        
        return True
        
    except Exception as e:
        print(f"❌ Diversity Controller test failed: {e}")
        return False

def test_utilization_balancer():
    """Test Utilization Balancer functionality"""
    print("🧪 Testing Utilization Balancer...")
    
    try:
        from utilization_balancer import get_utilization_balancer
        from expert_registry import get_expert_registry
        
        # Register an expert
        registry = get_expert_registry()
        expert_id = registry.register_expert(
            capabilities=["nlp"],
            embedding_signature=[0.1, 0.2, 0.3, 0.4],
            device="cuda:0",
            memory_footprint_mb=1024.0
        )
        
        # Get balancer instance
        balancer = get_utilization_balancer()
        print("✅ UtilizationBalancer instance created")
        
        # Track usage
        balancer.track_usage(expert_id, response_time=0.1, success=True)
        balancer.track_usage(expert_id, response_time=0.2, success=False)
        print("✅ Usage tracking works")
        
        # Get usage stats
        stats = balancer.get_usage_stats(expert_id)
        assert stats is not None, "Usage stats should not be None"
        assert stats.total_requests == 2, "Should have 2 requests tracked"
        print("✅ Usage stats retrieval works")
        
        # Check balancing actions
        actions = balancer.check_balancing_actions()
        assert isinstance(actions, list), "Actions should be a list"
        print("✅ Balancing actions check works")
        
        return True
        
    except Exception as e:
        print(f"❌ Utilization Balancer test failed: {e}")
        return False

def test_evolution_module():
    """Test Evolution Module functionality"""
    print("🧪 Testing Evolution Module...")
    
    try:
        from evolution_module import get_evolution_module
        from expert_registry import get_expert_registry
        
        # Register an expert
        registry = get_expert_registry()
        expert_id = registry.register_expert(
            capabilities=["nlp"],
            embedding_signature=[0.1, 0.2, 0.3, 0.4],
            device="cuda:0",
            memory_footprint_mb=1024.0
        )
        
        # Get evolution module instance
        evolution = get_evolution_module()
        print("✅ EvolutionModule instance created")
        
        # Check split criteria
        gradient_vars = [0.6, 0.7, 0.8]
        error_clusters = [[0.1, 0.2], [0.8, 0.9]]
        
        should_split = evolution.check_split_criteria(expert_id, gradient_vars, error_clusters)
        assert isinstance(should_split, bool), "Split criteria check should return boolean"
        print("✅ Split criteria check works")
        
        # Simulate evolution
        from evolution_module import EvolutionType
        simulation_data = {
            "gradient_variances": gradient_vars,
            "error_clusters": error_clusters
        }
        
        result = evolution.simulate_evolution(expert_id, EvolutionType.SPLIT, simulation_data)
        assert result is not None, "Simulation result should not be None"
        print("✅ Evolution simulation works")
        
        return True
        
    except Exception as e:
        print(f"❌ Evolution Module test failed: {e}")
        return False

def test_distillation_interface():
    """Test Distillation Interface functionality"""
    print("🧪 Testing Distillation Interface...")
    
    try:
        from distillation_interface import get_distillation_interface
        from expert_registry import get_expert_registry
        
        # Register experts
        registry = get_expert_registry()
        general_expert_id = registry.register_expert(
            capabilities=["general"],
            embedding_signature=[0.1, 0.1, 0.1, 0.1],
            device="cuda:0",
            memory_footprint_mb=2048.0,
            version="1.0.0"
        )
        
        target_expert_id = registry.register_expert(
            capabilities=["nlp"],
            embedding_signature=[0.2, 0.3, 0.4, 0.5],
            device="cuda:1",
            memory_footprint_mb=1024.0,
            version="1.0.0"
        )
        
        # Get distillation interface instance
        distillation = get_distillation_interface()
        print("✅ DistillationInterface instance created")
        
        # Test distillation
        from distillation_interface import DistillationConfig, DistillationDirection
        config = DistillationConfig(
            direction=DistillationDirection.GENERAL_TO_EXPERT,
            layers_to_distill=[0, 1, 2]
        )
        
        result = distillation.distill_to_expert(general_expert_id, target_expert_id, config)
        assert result is not None, "Distillation result should not be None"
        assert hasattr(result, 'success'), "Result should have success attribute"
        print("✅ Knowledge distillation works")
        
        # Test checkpointing
        checkpoint_id = distillation.create_checkpoint(target_expert_id, {"test": "data"})
        assert checkpoint_id.startswith("ckpt_"), "Checkpoint ID should start with 'ckpt_'"
        print("✅ Checkpoint creation works")
        
        return True
        
    except Exception as e:
        print(f"❌ Distillation Interface test failed: {e}")
        return False

def test_lifecycle_manager():
    """Test Lifecycle Manager functionality"""
    print("🧪 Testing Lifecycle Manager...")
    
    try:
        from lifecycle_manager import get_lifecycle_manager
        from expert_registry import get_expert_registry
        
        # Register an expert
        registry = get_expert_registry()
        expert_id = registry.register_expert(
            capabilities=["nlp"],
            embedding_signature=[0.1, 0.2, 0.3, 0.4],
            device="cuda:0",
            memory_footprint_mb=1024.0
        )
        
        # Get lifecycle manager instance
        lifecycle = get_lifecycle_manager()
        print("✅ LifecycleManager instance created")
        
        # Test state transition
        from lifecycle_manager import LifecycleState
        success = lifecycle.transition_state(expert_id, LifecycleState.PAUSED, "test")
        assert success, "State transition should succeed"
        print("✅ State transition works")
        
        # Get lifecycle report
        report = lifecycle.get_lifecycle_report()
        assert report is not None, "Report should not be None"
        print("✅ Lifecycle report generation works")
        
        return True
        
    except Exception as e:
        print(f"❌ Lifecycle Manager test failed: {e}")
        return False

def run_all_tests():
    """Run all Expert Engine tests"""
    print("🚀 Running MAHIA Expert Engine Test Suite")
    print("=" * 50)
    
    tests = [
        test_expert_registry,
        test_contextual_router,
        test_diversity_controller,
        test_utilization_balancer,
        test_evolution_module,
        test_distillation_interface,
        test_lifecycle_manager
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
        print("🎉 All tests passed! Expert Engine is ready for use.")
        return True
    else:
        print("❌ Some tests failed. Please check the implementation.")
        return False

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)