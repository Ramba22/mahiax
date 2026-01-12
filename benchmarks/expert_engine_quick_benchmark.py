"""
Quick benchmark for MAHIA Expert Engine
Tests key performance metrics without verbose output.
"""

import sys
import os
import time
import random
import psutil

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

def benchmark_expert_registry():
    """Benchmark Expert Registry performance"""
    print("📊 Benchmarking Expert Registry...")
    
    from expert_registry import get_expert_registry
    registry = get_expert_registry()
    
    # Clear existing experts
    registry.experts.clear()
    
    # Measure registration performance
    start_time = time.time()
    expert_ids = []
    
    for i in range(100):
        expert_id = registry.register_expert(
            capabilities=["benchmark", f"capability_{i}"],
            embedding_signature=[random.random() for _ in range(10)],
            device=f"cuda:{i % 4}",
            memory_footprint_mb=100.0 + (i % 100)
        )
        expert_ids.append(expert_id)
    
    registration_time = time.time() - start_time
    
    # Measure retrieval performance
    start_time = time.time()
    for expert_id in expert_ids[:50]:  # Test first 50
        expert = registry.get_expert(expert_id)
        assert expert is not None
    
    retrieval_time = time.time() - start_time
    
    # Measure active experts retrieval
    start_time = time.time()
    active_experts = registry.get_active_experts()
    active_retrieval_time = time.time() - start_time
    
    result = {
        "registration_time_ms": registration_time * 1000,
        "retrieval_time_ms": retrieval_time * 1000,
        "active_retrieval_time_ms": active_retrieval_time * 1000,
        "experts_registered": len(expert_ids),
        "experts_retrieved": len(active_experts)
    }
    
    print(f"  Registration: {result['registration_time_ms']:.2f}ms for {len(expert_ids)} experts")
    print(f"  Retrieval: {result['retrieval_time_ms']:.2f}ms for 50 experts")
    print(f"  Active retrieval: {result['active_retrieval_time_ms']:.2f}ms")
    
    return result

def benchmark_contextual_router():
    """Benchmark Contextual Router performance"""
    print("🧭 Benchmarking Contextual Router...")
    
    from contextual_router import get_contextual_router, RoutingMode
    from expert_registry import get_expert_registry
    
    registry = get_expert_registry()
    router = get_contextual_router()
    
    # Register test experts
    expert_ids = []
    for i in range(50):
        expert_id = registry.register_expert(
            capabilities=["benchmark", f"domain_{i % 5}"],
            embedding_signature=[random.random() for _ in range(20)],
            device=f"cuda:{i % 4}",
            memory_footprint_mb=200.0
        )
        expert_ids.append(expert_id)
    
    # Generate test contexts
    test_contexts = [[random.random() for _ in range(20)] for _ in range(100)]
    
    # Benchmark different routing modes
    routing_results = {}
    
    for mode_name, mode in [("TOP_K", RoutingMode.TOP_K), 
                           ("REFLECTIVE", RoutingMode.REFLECTIVE)]:
        start_time = time.time()
        
        for context in test_contexts:
            result = router.route(context, k=3, mode=mode)
            assert result is not None
        
        total_time = time.time() - start_time
        avg_time_ms = (total_time / len(test_contexts)) * 1000
        
        routing_results[mode_name] = {
            "total_time_ms": total_time * 1000,
            "avg_time_ms": avg_time_ms,
            "routes_per_second": len(test_contexts) / total_time
        }
        
        print(f"  {mode_name}: {avg_time_ms:.3f}ms per route ({len(test_contexts) / total_time:.0f} routes/sec)")
    
    return routing_results

def benchmark_memory_usage():
    """Benchmark memory usage"""
    print("💾 Benchmarking Memory Usage...")
    
    # Get process memory info
    process = psutil.Process()
    memory_info = process.memory_info()
    
    result = {
        "rss_mb": memory_info.rss / (1024 * 1024),
        "vms_mb": memory_info.vms / (1024 * 1024),
        "percent": process.memory_percent()
    }
    
    print(f"  RSS: {result['rss_mb']:.1f} MB")
    print(f"  VMS: {result['vms_mb']:.1f} MB")
    print(f"  Percentage: {result['percent']:.1f}%")
    
    return result

def run_quick_benchmark():
    """Run quick benchmarks and collect results"""
    print("🚀 Running Quick MAHIA Expert Engine Benchmark")
    print("=" * 50)
    
    start_time = time.time()
    
    # Run individual benchmarks
    registry_results = benchmark_expert_registry()
    router_results = benchmark_contextual_router()
    memory_results = benchmark_memory_usage()
    
    total_time = time.time() - start_time
    
    # Compile final results
    results = {
        "timestamp": time.time(),
        "total_benchmark_time_seconds": total_time,
        "expert_registry": registry_results,
        "contextual_router": router_results,
        "memory_usage": memory_results
    }
    
    return results

def generate_report(results):
    """Generate a formatted benchmark report"""
    if not results:
        return "No benchmark results available. Run benchmarks first."
    
    report = []
    report.append("📈 MAHIA Expert Engine Quick Benchmark Report")
    report.append("=" * 50)
    report.append(f"Total Benchmark Time: {results['total_benchmark_time_seconds']:.2f} seconds")
    report.append("")
    
    # Expert Registry Results
    registry = results['expert_registry']
    report.append("📚 Expert Registry Performance:")
    report.append(f"  Registration: {registry['registration_time_ms']:.2f}ms for {registry['experts_registered']} experts")
    report.append(f"  Retrieval: {registry['retrieval_time_ms']:.2f}ms for 50 experts")
    report.append(f"  Active Experts Retrieval: {registry['active_retrieval_time_ms']:.2f}ms")
    report.append("")
    
    # Contextual Router Results
    router = results['contextual_router']
    report.append("🧭 Contextual Router Performance:")
    for mode, stats in router.items():
        report.append(f"  {mode}: {stats['avg_time_ms']:.3f}ms/route ({stats['routes_per_second']:.0f} routes/sec)")
    report.append("")
    
    # Memory Usage Results
    memory = results['memory_usage']
    report.append("💾 Memory Usage:")
    report.append(f"  RSS: {memory['rss_mb']:.1f} MB")
    report.append(f"  VMS: {memory['vms_mb']:.1f} MB")
    report.append(f"  Percentage: {memory['percent']:.1f}%")
    report.append("")
    
    # Summary Metrics
    report.append("🏆 Key Performance Indicators:")
    
    # Calculate overall performance metrics
    total_routes = sum(stats['routes_per_second'] for stats in router.values())
    avg_route_time = sum(stats['avg_time_ms'] for stats in router.values()) / len(router)
    
    report.append(f"  Overall Routing Throughput: {total_routes:.0f} routes/sec")
    report.append(f"  Average Route Time: {avg_route_time:.3f}ms")
    report.append(f"  Registry Throughput: {registry['experts_registered'] / (registry['registration_time_ms'] / 1000):.0f} experts/sec")
    report.append(f"  Memory Efficiency: {memory['rss_mb']:.1f} MB for {registry['experts_registered']} experts")
    
    return "\n".join(report)

def main():
    """Main benchmark function"""
    # Run quick benchmark
    results = run_quick_benchmark()
    
    # Generate and print report
    report = generate_report(results)
    print(report)
    
    # Save results to file
    timestamp = int(time.time())
    filename = f"expert_engine_quick_benchmark_{timestamp}.txt"
    
    with open(filename, "w", encoding="utf-8") as f:
        f.write(report)
    
    print(f"\n💾 Benchmark report saved to: {filename}")
    
    # Print key metrics
    print("\n📋 Key Metrics Summary:")
    registry = results['expert_registry']
    router = results['contextual_router']
    memory = results['memory_usage']
    
    print(f"  Experts Registered: {registry['experts_registered']}")
    print(f"  Registry Performance: {registry['experts_registered'] / (registry['registration_time_ms'] / 1000):.0f} experts/sec")
    print(f"  Routing Performance: {router['TOP_K']['routes_per_second']:.0f} routes/sec (TOP_K)")
    print(f"  Memory Usage: {memory['rss_mb']:.1f} MB")
    print(f"  Memory per Expert: {memory['rss_mb'] / registry['experts_registered'] * 1000:.1f} KB/expert")

if __name__ == "__main__":
    main()