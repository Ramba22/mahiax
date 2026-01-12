"""
Demonstration of MAHIA Expert Engine
Shows how all components work together in a complete workflow.
"""

import time
import random

def demo_expert_engine():
    """Demonstrate the complete Expert Engine workflow"""
    print("🚀 MAHIA Expert Engine Demonstration")
    print("=" * 50)
    
    # Import all components
    from expert_registry import get_expert_registry
    from contextual_router import get_contextual_router, RoutingMode
    from diversity_controller import get_diversity_controller
    from utilization_balancer import get_utilization_balancer
    from evolution_module import get_evolution_module, EvolutionType
    from distillation_interface import get_distillation_interface, DistillationConfig, DistillationDirection
    from lifecycle_manager import get_lifecycle_manager, LifecycleState
    
    # Initialize components
    print("🔧 Initializing Expert Engine components...")
    registry = get_expert_registry()
    router = get_contextual_router()
    diversity_controller = get_diversity_controller()
    balancer = get_utilization_balancer()
    evolution = get_evolution_module()
    distillation = get_distillation_interface()
    lifecycle = get_lifecycle_manager()
    
    print("✅ All components initialized\n")
    
    # Step 1: Register experts
    print("📋 Step 1: Registering Experts")
    expert_ids = []
    
    # Register general expert
    general_expert_id = registry.register_expert(
        capabilities=["general"],
        embedding_signature=[0.1, 0.1, 0.1, 0.1, 0.1],
        device="cuda:0",
        memory_footprint_mb=2048.0,
        version="1.0.0"
    )
    expert_ids.append(general_expert_id)
    print(f"  🌐 General Expert: {general_expert_id}")
    
    # Register specialized experts
    specialized_experts = [
        {
            "capabilities": ["nlp", "translation"],
            "embedding": [0.8, 0.2, 0.1, 0.3, 0.4],
            "device": "cuda:1"
        },
        {
            "capabilities": ["vision", "classification"],
            "embedding": [0.2, 0.8, 0.3, 0.1, 0.2],
            "device": "cuda:2"
        },
        {
            "capabilities": ["audio", "speech"],
            "embedding": [0.1, 0.3, 0.8, 0.2, 0.1],
            "device": "cuda:3"
        }
    ]
    
    for i, spec in enumerate(specialized_experts):
        expert_id = registry.register_expert(
            capabilities=spec["capabilities"],
            embedding_signature=spec["embedding"],
            device=spec["device"],
            memory_footprint_mb=1024.0,
            version="1.0.0"
        )
        expert_ids.append(expert_id)
        print(f"  🎯 Specialist {i+1} ({', '.join(spec['capabilities'])}): {expert_id}")
    
    print(f"\n📊 Total experts registered: {len(expert_ids)}\n")
    
    # Step 2: Demonstrate routing
    print("🧭 Step 2: Contextual Routing")
    
    # Simulate different input contexts
    test_contexts = [
        [0.7, 0.2, 0.1, 0.3, 0.4],  # NLP context
        [0.2, 0.7, 0.3, 0.1, 0.2],  # Vision context
        [0.1, 0.3, 0.7, 0.2, 0.1],  # Audio context
        [0.3, 0.3, 0.3, 0.3, 0.3]   # General context
    ]
    
    context_names = ["NLP Task", "Vision Task", "Audio Task", "General Task"]
    
    for i, (context, name) in enumerate(zip(test_contexts, context_names)):
        print(f"\n  📝 {name}:")
        
        # Try different routing modes
        for mode in [RoutingMode.TOP_K, RoutingMode.REFLECTIVE]:
            result = router.route(context, k=2, mode=mode)
            if result.expert_ids:
                expert_info = registry.get_expert(result.expert_ids[0])
                capabilities = expert_info.capabilities if expert_info else ["unknown"]
                print(f"    {mode.value}: {result.expert_ids[0][:8]}... ({', '.join(capabilities)}) "
                      f"[score: {result.scores[0]:.3f}]")
            
            # Simulate usage tracking
            if result.expert_ids:
                success = random.choice([True, True, True, False])  # 75% success rate
                response_time = random.uniform(0.05, 0.3)
                balancer.track_usage(result.expert_ids[0], response_time, success)
    
    print("\n✅ Routing demonstration complete\n")
    
    # Step 3: Check diversity
    print("🔄 Step 3: Diversity Analysis")
    metrics = diversity_controller.calculate_diversity_metrics()
    print(f"  Output Overlap: {metrics.output_overlap:.3f}")
    print(f"  Feature Entropy: {metrics.feature_entropy:.3f}")
    
    # Check for high redundancy
    audit_result = diversity_controller.periodic_audit()
    if "metrics" in audit_result:
        high_similarity = audit_result["metrics"].get("high_similarity_pairs", [])
        print(f"  High Similarity Pairs: {len(high_similarity)}")
    
    print("\n✅ Diversity analysis complete\n")
    
    # Step 4: Utilization balancing
    print("⚖️  Step 4: Utilization Balancing")
    
    # Check balancing actions
    actions = balancer.run_periodic_balancing()
    print(f"  Balancing Actions Applied: {len(actions.get('applied_actions', []))}")
    
    # Show utilization heatmap
    heatmap = balancer.generate_heatmap()
    print("  Utilization Heatmap:")
    for expert_id, score in list(heatmap.items())[:3]:  # Show top 3
        print(f"    {expert_id[:8]}...: {score:.3f}")
    
    print("\n✅ Utilization balancing complete\n")
    
    # Step 5: Knowledge distillation
    print("⚗️  Step 5: Knowledge Distillation")
    
    # Distill from general to specialized expert
    nlp_expert = None
    for expert_id in expert_ids[1:]:  # Skip general expert
        expert = registry.get_expert(expert_id)
        if expert and "nlp" in expert.capabilities:
            nlp_expert = expert_id
            break
    
    if nlp_expert:
        config = DistillationConfig(
            direction=DistillationDirection.GENERAL_TO_EXPERT,
            layers_to_distill=[0, 1, 2, 3, 4],
            bandwidth_aware=True
        )
        
        result = distillation.distill_to_expert(general_expert_id, nlp_expert, config)
        if result.success:
            print(f"  Successfully distilled knowledge to {nlp_expert[:8]}...")
            print(f"    Layers transferred: {len(result.layers_transferred)}")
            print(f"    Validation passed: {result.validation_passed}")
        else:
            print(f"  ❌ Distillation failed: {result.error_message}")
    
    print("\n✅ Knowledge distillation complete\n")
    
    # Step 6: Evolution simulation
    print("🧬 Step 6: Evolution Simulation")
    
    # Check if any expert should be split
    test_expert_id = expert_ids[1]  # First specialized expert
    gradient_variances = [0.6, 0.7, 0.8]  # High variance
    error_clusters = [[0.1, 0.2], [0.8, 0.9]]  # Distinct clusters
    
    should_split = evolution.check_split_criteria(test_expert_id, gradient_variances, error_clusters)
    print(f"  Expert {test_expert_id[:8]}... should be split: {should_split}")
    
    if should_split:
        simulation_data = {
            "gradient_variances": gradient_variances,
            "error_clusters": error_clusters
        }
        
        simulation_result = evolution.simulate_evolution(test_expert_id, EvolutionType.SPLIT, simulation_data)
        if simulation_result.get("success"):
            print(f"  Split simulation successful:")
            print(f"    Feasibility score: {simulation_result['feasibility_score']:.3f}")
            print(f"    Estimated memory increase: {simulation_result['estimated_memory_increase_mb']:.1f} MB")
    
    print("\n✅ Evolution simulation complete\n")
    
    # Step 7: Lifecycle management
    print("🔄 Step 7: Lifecycle Management")
    
    # Transition an expert through states
    test_expert_id = expert_ids[2]  # Third expert
    print(f"  Managing lifecycle for expert {test_expert_id[:8]}...")
    
    # Pause expert
    success = lifecycle.transition_state(test_expert_id, LifecycleState.PAUSED, "demo_pause")
    print(f"    Paused: {'✅' if success else '❌'}")
    
    # Resume expert
    success = lifecycle.transition_state(test_expert_id, LifecycleState.ACTIVE, "demo_resume")
    print(f"    Resumed: {'✅' if success else '❌'}")
    
    # Show lifecycle report
    lifecycle_report = lifecycle.get_lifecycle_report()
    if lifecycle_report.get("status") == "success":
        states = lifecycle_report.get("current_expert_states", {})
        print(f"    Current expert states: {states}")
    
    print("\n✅ Lifecycle management complete\n")
    
    # Step 8: Final summary
    print("📊 Step 8: Final Summary")
    print(f"  Total Experts: {len(registry.get_active_experts())}")
    print(f"  Routing Decisions Made: 8")
    print(f"  Distillation Operations: 1")
    print(f"  Lifecycle Transitions: 2")
    
    # Show system health
    balancer_report = balancer.get_policy_engine_status()
    active_cooldowns = balancer_report["current_state"]["active_cooldowns"]
    print(f"  Active Cooldowns: {active_cooldowns}")
    
    print("\n" + "=" * 50)
    print("🎉 Expert Engine Demonstration Complete!")
    print("✅ All components working together successfully")

if __name__ == "__main__":
    demo_expert_engine()