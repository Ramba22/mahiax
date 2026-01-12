"""
Comprehensive benchmark for MAHIA Expert Engine
Tests performance, scalability, and resource usage.
"""

import sys
import os
import time
import random
import psutil
import threading
from typing import List, Dict, Any

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

class ExpertEngineBenchmark:
    """Benchmark suite for MAHIA Expert Engine"""
    
    def __init__(self):
        self.results = {}
        
    def benchmark_expert_registry(self) -> Dict[str, Any]:
        """Benchmark Expert Registry performance"""
        print("📊 Benchmarking Expert Registry...")
        
        from expert_registry import get_expert_registry
        registry = get_expert_registry()
        
        # Measure registration performance
        start_time = time.time()
        expert_ids = []
        
        for i in range(1000):
            expert_id = registry.register_expert(
                capabilities=["benchmark", f"capability_{i}"],
                embedding_signature=[random.random() for _ in range(10)],
                device=f"cuda:{i % 4}",
                memory_footprint_mb=100.0 + (i % 1000)
            )
            expert_ids.append(expert_id)
        
        registration_time = time.time() - start_time
        
        # Measure retrieval performance
        start_time = time.time()
        for expert_id in expert_ids[:100]:  # Test first 100
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
        print(f"  Retrieval: {result['retrieval_time_ms']:.2f}ms for 100 experts")
        print(f"  Active retrieval: {result['active_retrieval_time_ms']:.2f}ms")
        
        return result
    
    def benchmark_contextual_router(self) -> Dict[str, Any]:
        """Benchmark Contextual Router performance"""
        print("🧭 Benchmarking Contextual Router...")
        
        from contextual_router import get_contextual_router, RoutingMode
        from expert_registry import get_expert_registry
        
        registry = get_expert_registry()
        router = get_contextual_router()
        
        # Register test experts
        expert_ids = []
        for i in range(100):
            expert_id = registry.register_expert(
                capabilities=["benchmark", f"domain_{i % 10}"],
                embedding_signature=[random.random() for _ in range(20)],
                device=f"cuda:{i % 4}",
                memory_footprint_mb=200.0
            )
            expert_ids.append(expert_id)
        
        # Generate test contexts
        test_contexts = [[random.random() for _ in range(20)] for _ in range(1000)]
        
        # Benchmark different routing modes
        routing_results = {}
        
        for mode_name, mode in [("TOP_K", RoutingMode.TOP_K), 
                               ("REFLECTIVE", RoutingMode.REFLECTIVE),
                               ("ADAPTIVE", RoutingMode.ADAPTIVE)]:
            start_time = time.time()
            
            for context in test_contexts:
                result = router.route(context, k=5, mode=mode)
                assert result is not None
            
            total_time = time.time() - start_time
            avg_time_ms = (total_time / len(test_contexts)) * 1000
            
            routing_results[mode_name] = {
                "total_time_ms": total_time * 1000,
                "avg_time_ms": avg_time_ms,
                "routes_per_second": len(test_contexts) / total_time
            }
            
            print(f"  {mode_name}: {avg_time_ms:.3f}ms per route ({len(test_contexts) / total_time:.0f} routes/sec)")
        
        # Benchmark ensemble routing (separate as it's more expensive)
        start_time = time.time()
        for context in test_contexts[:100]:  # Fewer ensemble routes due to cost
            result = router.route(context, k=5, mode=RoutingMode.ENSEMBLE)
            assert result is not None
        
        ensemble_time = time.time() - start_time
        ensemble_avg_ms = (ensemble_time / 100) * 1000 if 100 > 0 else 0
        
        routing_results["ENSEMBLE"] = {
            "total_time_ms": ensemble_time * 1000,
            "avg_time_ms": ensemble_avg_ms,
            "routes_per_second": 100 / ensemble_time if ensemble_time > 0 else 0
        }
        
        print(f"  ENSEMBLE: {ensemble_avg_ms:.3f}ms per route ({100 / ensemble_time:.0f} routes/sec)")
        
        return routing_results
    
    def benchmark_diversity_controller(self) -> Dict[str, Any]:
        """Benchmark Diversity Controller performance"""
        print("🔄 Benchmarking Diversity Controller...")
        
        from diversity_controller import get_diversity_controller
        from expert_registry import get_expert_registry
        
        registry = get_expert_registry()
        controller = get_diversity_controller()
        
        # Register diverse experts
        for i in range(200):
            registry.register_expert(
                capabilities=[f"capability_{j}" for j in range(i % 10)],
                embedding_signature=[random.random() for _ in range(15)],
                device=f"cuda:{i % 4}",
                memory_footprint_mb=150.0
            )
        
        # Benchmark diversity metrics calculation
        start_time = time.time()
        for _ in range(100):
            metrics = controller.calculate_diversity_metrics()
            assert metrics is not None
        
        calculation_time = time.time() - start_time
        avg_calculation_time_ms = (calculation_time / 100) * 1000
        
        # Benchmark diversity loss computation
        start_time = time.time()
        test_outputs = [[random.random() for _ in range(10)] for _ in range(50)]
        
        for _ in range(1000):
            loss = controller.compute_diversity_loss(test_outputs)
            assert isinstance(loss, float)
        
        loss_time = time.time() - start_time
        avg_loss_time_ms = (loss_time / 1000) * 1000
        
        result = {
            "metrics_calculation": {
                "total_time_ms": calculation_time * 1000,
                "avg_time_ms": avg_calculation_time_ms,
                "calculations_per_second": 100 / calculation_time
            },
            "diversity_loss": {
                "total_time_ms": loss_time * 1000,
                "avg_time_ms": avg_loss_time_ms,
                "calculations_per_second": 1000 / loss_time
            }
        }
        
        print(f"  Metrics calculation: {avg_calculation_time_ms:.3f}ms per calculation")
        print(f"  Diversity loss: {avg_loss_time_ms:.3f}ms per calculation")
        
        return result
    
    def benchmark_utilization_balancer(self) -> Dict[str, Any]:
        """Benchmark Utilization Balancer performance"""
        print("⚖️  Benchmarking Utilization Balancer...")
        
        from utilization_balancer import get_utilization_balancer
        from expert_registry import get_expert_registry
        
        registry = get_expert_registry()
        balancer = get_utilization_balancer()
        
        # Register experts
        expert_ids = []
        for i in range(150):
            expert_id = registry.register_expert(
                capabilities=["benchmark"],
                embedding_signature=[random.random() for _ in range(10)],
                device=f"cuda:{i % 4}",
                memory_footprint_mb=125.0
            )
            expert_ids.append(expert_id)
        
        # Simulate usage tracking
        start_time = time.time()
        for _ in range(5000):
            expert_id = random.choice(expert_ids)
            response_time = random.uniform(0.01, 0.5)
            success = random.choice([True, True, False])  # 66% success rate
            balancer.track_usage(expert_id, response_time, success)
        
        tracking_time = time.time() - start_time
        
        # Benchmark usage stats retrieval
        start_time = time.time()
        for expert_id in expert_ids[:50]:
            stats = balancer.get_usage_stats(expert_id)
            assert stats is not None
        
        stats_time = time.time() - start_time
        
        # Benchmark heatmap generation
        start_time = time.time()
        for _ in range(100):
            heatmap = balancer.generate_heatmap()
            assert isinstance(heatmap, dict)
        
        heatmap_time = time.time() - start_time
        avg_heatmap_time_ms = (heatmap_time / 100) * 1000
        
        # Benchmark balancing actions
        start_time = time.time()
        for _ in range(100):
            actions = balancer.check_balancing_actions()
            assert isinstance(actions, list)
        
        actions_time = time.time() - start_time
        avg_actions_time_ms = (actions_time / 100) * 1000
        
        result = {
            "usage_tracking": {
                "total_time_ms": tracking_time * 1000,
                "operations_per_second": 5000 / tracking_time
            },
            "stats_retrieval": {
                "total_time_ms": stats_time * 1000,
                "avg_time_ms": (stats_time / 50) * 1000,
                "retrievals_per_second": 50 / stats_time
            },
            "heatmap_generation": {
                "total_time_ms": heatmap_time * 1000,
                "avg_time_ms": avg_heatmap_time_ms,
                "generations_per_second": 100 / heatmap_time
            },
            "balancing_actions": {
                "total_time_ms": actions_time * 1000,
                "avg_time_ms": avg_actions_time_ms,
                "checks_per_second": 100 / actions_time
            }
        }
        
        print(f"  Usage tracking: {5000 / tracking_time:.0f} operations/sec")
        print(f"  Stats retrieval: {(stats_time / 50) * 1000:.3f}ms per retrieval")
        print(f"  Heatmap generation: {avg_heatmap_time_ms:.3f}ms per generation")
        print(f"  Balancing actions: {avg_actions_time_ms:.3f}ms per check")
        
        return result
    
    def benchmark_memory_usage(self) -> Dict[str, Any]:
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
    
    def run_comprehensive_benchmark(self) -> Dict[str, Any]:
        """Run all benchmarks and collect results"""
        print("🚀 Running Comprehensive MAHIA Expert Engine Benchmark")
        print("=" * 60)
        
        start_time = time.time()
        
        # Run individual benchmarks
        registry_results = self.benchmark_expert_registry()
        router_results = self.benchmark_contextual_router()
        diversity_results = self.benchmark_diversity_controller()
        balancer_results = self.benchmark_utilization_balancer()
        memory_results = self.benchmark_memory_usage()
        
        total_time = time.time() - start_time
        
        # Compile final results
        self.results = {
            "timestamp": time.time(),
            "total_benchmark_time_seconds": total_time,
            "expert_registry": registry_results,
            "contextual_router": router_results,
            "diversity_controller": diversity_results,
            "utilization_balancer": balancer_results,
            "memory_usage": memory_results
        }
        
        return self.results
    
    def generate_report(self) -> str:
        """Generate a formatted benchmark report"""
        if not self.results:
            return "No benchmark results available. Run benchmarks first."
        
        report = []
        report.append("📈 MAHIA Expert Engine Benchmark Report")
        report.append("=" * 50)
        report.append(f"Total Benchmark Time: {self.results['total_benchmark_time_seconds']:.2f} seconds")
        report.append("")
        
        # Expert Registry Results
        registry = self.results['expert_registry']
        report.append("📚 Expert Registry Performance:")
        report.append(f"  Registration: {registry['registration_time_ms']:.2f}ms for {registry['experts_registered']} experts")
        report.append(f"  Retrieval: {registry['retrieval_time_ms']:.2f}ms for 100 experts")
        report.append(f"  Active Experts Retrieval: {registry['active_retrieval_time_ms']:.2f}ms")
        report.append("")
        
        # Contextual Router Results
        router = self.results['contextual_router']
        report.append("🧭 Contextual Router Performance:")
        for mode, stats in router.items():
            report.append(f"  {mode}: {stats['avg_time_ms']:.3f}ms/route ({stats['routes_per_second']:.0f} routes/sec)")
        report.append("")
        
        # Diversity Controller Results
        diversity = self.results['diversity_controller']
        report.append("🔄 Diversity Controller Performance:")
        report.append(f"  Metrics Calculation: {diversity['metrics_calculation']['avg_time_ms']:.3f}ms/calculation")
        report.append(f"  Diversity Loss: {diversity['diversity_loss']['avg_time_ms']:.3f}ms/calculation")
        report.append("")
        
        # Utilization Balancer Results
        balancer = self.results['utilization_balancer']
        report.append("⚖️  Utilization Balancer Performance:")
        report.append(f"  Usage Tracking: {balancer['usage_tracking']['operations_per_second']:.0f} ops/sec")
        report.append(f"  Stats Retrieval: {balancer['stats_retrieval']['avg_time_ms']:.3f}ms/retrieval")
        report.append(f"  Heatmap Generation: {balancer['heatmap_generation']['avg_time_ms']:.3f}ms/generation")
        report.append("")
        
        # Memory Usage Results
        memory = self.results['memory_usage']
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
    benchmark = ExpertEngineBenchmark()
    
    # Run comprehensive benchmark
    results = benchmark.run_comprehensive_benchmark()
    
    # Generate and print report
    report = benchmark.generate_report()
    print(report)
    
    # Save results to file
    timestamp = int(time.time())
    filename = f"expert_engine_benchmark_{timestamp}.txt"
    
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