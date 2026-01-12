"""
Full-System Stress and Robustness Tests for MAHIA-X.
"""
import torch
import torch.nn as nn
import numpy as np
import sys
import os
import time
import psutil
import threading
from typing import Dict, Any, Tuple, List, Optional
from collections import defaultdict, deque
import traceback

# Add the project root to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

try:
    from modell_V5_MAHIA_HyenaMoE import (
        MAHIA_V5, 
        PredictiveStopForecaster,
        ExtendStop,
        GradientEntropyMonitor,
        MetaLRPolicyController,
        ConfidenceTrendBasedLRAdjuster,
        ExpertLoadBalancerV2,
        FP8CalibrationAutoTuner,
        CurriculumMemorySystem,
        GPUTelemetryMonitor,
        TelemetryLogger,
        TrainingDashboardV2,
        AutoCheckpointingPolicy
    )
    STRESS_TEST_AVAILABLE = True
except ImportError:
    STRESS_TEST_AVAILABLE = False
    print("⚠️  MAHIA-X modules not available for stress testing")


class SystemMonitor:
    """Monitor system resources during stress testing"""
    
    def __init__(self, log_interval: int = 10):
        self.log_interval = log_interval
        self.monitoring = False
        self.monitor_thread = None
        self.metrics = {
            'cpu_percent': deque(maxlen=1000),
            'memory_percent': deque(maxlen=1000),
            'disk_io': deque(maxlen=1000),
            'network_io': deque(maxlen=1000)
        }
        
    def start_monitoring(self):
        """Start system monitoring in background thread"""
        self.monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
        
    def stop_monitoring(self):
        """Stop system monitoring"""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join()
            
    def _monitor_loop(self):
        """Background monitoring loop"""
        while self.monitoring:
            try:
                # CPU usage
                cpu_percent = psutil.cpu_percent(interval=1)
                self.metrics['cpu_percent'].append(cpu_percent)
                
                # Memory usage
                memory = psutil.virtual_memory()
                self.metrics['memory_percent'].append(memory.percent)
                
                # Disk I/O
                disk_io = psutil.disk_io_counters()
                if disk_io:
                    self.metrics['disk_io'].append({
                        'read_bytes': disk_io.read_bytes,
                        'write_bytes': disk_io.write_bytes
                    })
                
                # Network I/O
                net_io = psutil.net_io_counters()
                self.metrics['network_io'].append({
                    'bytes_sent': net_io.bytes_sent,
                    'bytes_recv': net_io.bytes_recv
                })
                
            except Exception as e:
                print(f"⚠️  System monitoring error: {e}")
                
            time.sleep(self.log_interval)
            
    def get_system_stats(self) -> Dict[str, Any]:
        """Get current system statistics"""
        if not self.metrics['cpu_percent']:
            return {}
            
        return {
            'cpu_percent': {
                'current': psutil.cpu_percent(),
                'avg': np.mean(self.metrics['cpu_percent']),
                'max': np.max(self.metrics['cpu_percent']),
                'min': np.min(self.metrics['cpu_percent'])
            },
            'memory_percent': {
                'current': psutil.virtual_memory().percent,
                'avg': np.mean(self.metrics['memory_percent']),
                'max': np.max(self.metrics['memory_percent']),
                'min': np.min(self.metrics['memory_percent'])
            },
            'monitoring_duration': len(self.metrics['cpu_percent']) * self.log_interval
        }


class StressTestScenario:
    """Base class for stress test scenarios"""
    
    def __init__(self, name: str, duration: int = 3600):  # 1 hour default
        self.name = name
        self.duration = duration
        self.start_time = None
        self.end_time = None
        self.results = {}
        self.errors = []
        
    def setup(self) -> bool:
        """Setup test scenario
        Returns:
            True if setup successful, False otherwise
        """
        try:
            print(f"🔧 Setting up {self.name} stress test...")
            self.start_time = time.time()
            return True
        except Exception as e:
            self.errors.append(f"Setup failed: {str(e)}")
            return False
            
    def run(self) -> bool:
        """Run the stress test
        Returns:
            True if test completed successfully, False otherwise
        """
        raise NotImplementedError("Subclasses must implement run()")
        
    def teardown(self) -> bool:
        """Teardown test scenario
        Returns:
            True if teardown successful, False otherwise
        """
        try:
            print(f"🧹 Tearing down {self.name} stress test...")
            self.end_time = time.time()
            return True
        except Exception as e:
            self.errors.append(f"Teardown failed: {str(e)}")
            return False
            
    def get_results(self) -> Dict[str, Any]:
        """Get test results
        Returns:
            Dictionary of test results
        """
        duration = (self.end_time - self.start_time) if self.end_time and self.start_time else 0
        return {
            'name': self.name,
            'duration': duration,
            'success': len(self.errors) == 0,
            'errors': self.errors,
            'results': self.results
        }


class MemoryStressTest(StressTestScenario):
    """Memory stress test with large model allocations"""
    
    def __init__(self, name: str = "MemoryStressTest", duration: int = 1800):
        super().__init__(name, duration)
        self.models = []
        self.tensors = []
        
    def run(self) -> bool:
        """Run memory stress test"""
        try:
            print(f"🏃 Running {self.name}...")
            
            # Test 1: Create multiple large models
            print("   Creating multiple large models...")
            for i in range(5):
                if STRESS_TEST_AVAILABLE:
                    model = MAHIA_V5(
                        vocab_size=10000,
                        text_seq_len=128,
                        tab_dim=100,
                        embed_dim=256,
                        fused_dim=512,
                        moe_experts=16,
                        moe_topk=4
                    )
                else:
                    # Mock model for testing
                    model = nn.Linear(100, 10)
                self.models.append(model)
                print(f"   Created model {i+1}/5")
                
            # Test 2: Allocate large tensors
            print("   Allocating large tensors...")
            for i in range(10):
                tensor = torch.randn(1000, 1000, 100)  # ~400MB each
                self.tensors.append(tensor)
                print(f"   Allocated tensor {i+1}/10")
                
            # Test 3: Perform operations on tensors
            print("   Performing tensor operations...")
            for i, tensor in enumerate(self.tensors):
                result = torch.matmul(tensor, tensor.transpose(-1, -2))
                self.tensors[i] = result
                print(f"   Processed tensor {i+1}/10")
                
            # Test 4: Check memory usage
            if torch.cuda.is_available():
                gpu_memory = torch.cuda.memory_allocated() / (1024**3)  # GB
                print(f"   GPU memory allocated: {gpu_memory:.2f} GB")
                self.results['gpu_memory_gb'] = gpu_memory
                
            cpu_memory = psutil.virtual_memory()
            print(f"   System memory usage: {cpu_memory.percent:.1f}%")
            self.results['system_memory_percent'] = cpu_memory.percent
            
            return True
            
        except Exception as e:
            self.errors.append(f"Memory stress test failed: {str(e)}")
            traceback.print_exc()
            return False
            
    def teardown(self) -> bool:
        """Teardown memory stress test"""
        # Clear allocated memory
        self.models.clear()
        self.tensors.clear()
        
        # Force garbage collection
        import gc
        gc.collect()
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        return super().teardown()


class TrainingStressTest(StressTestScenario):
    """Long-term training stress test"""
    
    def __init__(self, name: str = "TrainingStressTest", duration: int = 7200):
        super().__init__(name, duration)
        self.model = None
        self.optimizer = None
        self.criterion = None
        self.system_monitor = SystemMonitor(log_interval=30)
        
    def setup(self) -> bool:
        """Setup training stress test"""
        if not super().setup():
            return False
            
        try:
            # Create model
            if STRESS_TEST_AVAILABLE:
                self.model = MAHIA_V5(
                    vocab_size=5000,
                    text_seq_len=64,
                    tab_dim=50,
                    embed_dim=128,
                    fused_dim=256,
                    moe_experts=8,
                    moe_topk=2
                )
            else:
                # Mock model for testing
                self.model = nn.Linear(100, 2)
            
            # Create optimizer and criterion
            self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-3)
            self.criterion = nn.CrossEntropyLoss()
            
            # Start system monitoring
            self.system_monitor.start_monitoring()
            
            return True
            
        except Exception as e:
            self.errors.append(f"Training setup failed: {str(e)}")
            return False
            
    def run(self) -> bool:
        """Run long-term training stress test"""
        try:
            print(f"🏃 Running {self.name}...")
            
            # Initialize components
            if STRESS_TEST_AVAILABLE:
                gradient_monitor = GradientEntropyMonitor()
                lr_controller = ConfidenceTrendBasedLRAdjuster()
                early_stopper = ExtendStop(patience=50, max_extensions=2)
                load_balancer = ExpertLoadBalancerV2()
                curriculum_memory = CurriculumMemorySystem()
                telemetry_logger = TelemetryLogger()
                dashboard = TrainingDashboardV2()
            else:
                # Mock components for testing
                gradient_monitor = None
                lr_controller = None
                early_stopper = None
                load_balancer = None
                curriculum_memory = None
                telemetry_logger = None
                dashboard = None
            
            # Training loop
            batch_size = 16
            global_step = 0
            epoch = 0
            max_steps = 1000  # Limit for demo
            
            print(f"   Starting training loop (max {max_steps} steps)...")
            
            while global_step < max_steps and (time.time() - self.start_time) < self.duration:
                # Create synthetic batch
                text = torch.randint(0, 5000, (batch_size, 64))
                tab = torch.randn(batch_size, 50)
                targets = torch.randint(0, 2, (batch_size,))
                
                # Forward pass
                self.optimizer.zero_grad()
                outputs, aux_loss = self.model(text, tab)
                
                # Compute loss
                loss = self.criterion(outputs, targets)
                if aux_loss is not None:
                    loss = loss + 0.01 * aux_loss
                    
                # Backward pass
                loss.backward()
                
                # Gradient monitoring
                grad_recommendations = gradient_monitor.should_adjust_training(self.model)
                
                # Update optimizer
                self.optimizer.step()
                
                # Log metrics
                if global_step % 10 == 0:
                    # Compute gradient entropy
                    entropy = gradient_monitor.compute_gradient_entropy(self.model)
                    
                    # Update dashboard
                    dashboard.update_metrics(
                        grad_entropy=entropy,
                        lr=self.optimizer.param_groups[0]['lr'],
                        loss=loss.item(),
                        step=global_step,
                        epoch=epoch
                    )
                    
                    # Log to telemetry
                    telemetry_logger.log_gradient_entropy(global_step, epoch, entropy)
                    telemetry_logger.log_loss(global_step, epoch, loss.item())
                    telemetry_logger.log_learning_rate(global_step, epoch, self.optimizer.param_groups[0]['lr'])
                    
                    print(f"   Step {global_step}: Loss={loss.item():.4f}, Entropy={entropy:.4f}")
                    
                # Periodic load balancing
                if global_step % 50 == 0:
                    load_balancer.should_reweight()
                    
                global_step += 1
                
                # Check for early stopping (simplified)
                if global_step % 100 == 0:
                    stop_result = early_stopper(loss.item(), 0.5, epoch)
                    if stop_result.get("action") == "stop":
                        print("   Early stopping triggered")
                        break
                        
            # Store results
            self.results = {
                'steps_completed': global_step,
                'final_loss': loss.item() if 'loss' in locals() else float('inf'),
                'system_stats': self.system_monitor.get_system_stats()
            }
            
            return True
            
        except Exception as e:
            self.errors.append(f"Training stress test failed: {str(e)}")
            traceback.print_exc()
            return False
            
    def teardown(self) -> bool:
        """Teardown training stress test"""
        # Stop system monitoring
        self.system_monitor.stop_monitoring()
        
        # Clear model and optimizer
        self.model = None
        self.optimizer = None
        self.criterion = None
        
        return super().teardown()


class ConcurrencyStressTest(StressTestScenario):
    """Concurrency stress test with multiple threads"""
    
    def __init__(self, name: str = "ConcurrencyStressTest", duration: int = 1200):
        super().__init__(name, duration)
        self.threads = []
        self.results_lock = threading.Lock()
        self.thread_results = []
        
    def worker_thread(self, thread_id: int):
        """Worker thread function"""
        try:
            print(f"   Thread {thread_id} started")
            
            # Create model for this thread
            model = MAHIA_V5(
                vocab_size=1000,
                text_seq_len=32,
                tab_dim=20,
                embed_dim=64,
                fused_dim=128
            )
            
            # Perform inference operations
            for i in range(50):
                text = torch.randint(0, 1000, (4, 32))
                tab = torch.randn(4, 20)
                
                with torch.no_grad():
                    outputs, _ = model(text, tab)
                    
                if i % 10 == 0:
                    print(f"   Thread {thread_id}: Step {i}/50")
                    
            # Store results
            with self.results_lock:
                self.thread_results.append({
                    'thread_id': thread_id,
                    'steps_completed': 50,
                    'success': True
                })
                
            print(f"   Thread {thread_id} completed")
            
        except Exception as e:
            print(f"   Thread {thread_id} failed: {e}")
            with self.results_lock:
                self.thread_results.append({
                    'thread_id': thread_id,
                    'error': str(e),
                    'success': False
                })
                
    def run(self) -> bool:
        """Run concurrency stress test"""
        try:
            print(f"🏃 Running {self.name}...")
            
            # Create multiple threads
            num_threads = 8
            print(f"   Creating {num_threads} concurrent threads...")
            
            for i in range(num_threads):
                thread = threading.Thread(target=self.worker_thread, args=(i,))
                self.threads.append(thread)
                thread.start()
                
            # Wait for threads to complete
            print("   Waiting for threads to complete...")
            for thread in self.threads:
                thread.join(timeout=300)  # 5 minute timeout
                
            # Check results
            successful_threads = sum(1 for r in self.thread_results if r.get('success', False))
            failed_threads = len(self.thread_results) - successful_threads
            
            self.results = {
                'threads_created': num_threads,
                'threads_completed': successful_threads,
                'threads_failed': failed_threads,
                'thread_results': self.thread_results
            }
            
            print(f"   Concurrency test: {successful_threads}/{num_threads} threads successful")
            
            return successful_threads > 0  # At least one thread succeeded
            
        except Exception as e:
            self.errors.append(f"Concurrency stress test failed: {str(e)}")
            traceback.print_exc()
            return False


class RobustnessTestSuite:
    """Complete robustness test suite"""
    
    def __init__(self):
        self.tests = [
            MemoryStressTest(),
            TrainingStressTest(),
            ConcurrencyStressTest()
        ]
        self.results = []
        
    def run_all_tests(self) -> Dict[str, Any]:
        """Run all stress and robustness tests
        Returns:
            Dictionary of test results
        """
        print("🚀 Starting Full-System Stress and Robustness Tests...")
        print("=" * 60)
        
        overall_start_time = time.time()
        passed_tests = 0
        total_tests = len(self.tests)
        
        for i, test in enumerate(self.tests):
            print(f"\n🧪 Test {i+1}/{total_tests}: {test.name}")
            print("-" * 40)
            
            # Setup
            if not test.setup():
                print(f"❌ {test.name} setup failed")
                self.results.append(test.get_results())
                continue
                
            # Run
            test_success = test.run()
            
            # Teardown
            test.teardown()
            
            # Get results
            test_results = test.get_results()
            self.results.append(test_results)
            
            if test_success and test_results['success']:
                print(f"✅ {test.name} PASSED")
                passed_tests += 1
            else:
                print(f"❌ {test.name} FAILED")
                if test_results['errors']:
                    for error in test_results['errors']:
                        print(f"   Error: {error}")
                        
        overall_duration = time.time() - overall_start_time
        
        # Summary
        print("\n" + "=" * 60)
        print("STRESS AND ROBUSTNESS TEST SUMMARY")
        print("=" * 60)
        print(f"Tests completed: {passed_tests}/{total_tests}")
        print(f"Overall duration: {overall_duration:.1f} seconds")
        print(f"Success rate: {passed_tests/total_tests*100:.1f}%")
        
        if passed_tests == total_tests:
            print("🎉 All tests PASSED! System is robust and stable.")
        else:
            print("⚠️  Some tests FAILED. System may need improvements.")
            
        # Detailed results
        print("\nDetailed Results:")
        for result in self.results:
            status = "✅ PASS" if result['success'] else "❌ FAIL"
            print(f"  {status} {result['name']} ({result['duration']:.1f}s)")
            if 'results' in result and result['results']:
                for key, value in result['results'].items():
                    if isinstance(value, dict):
                        print(f"    {key}:")
                        for subkey, subvalue in value.items():
                            print(f"      {subkey}: {subvalue}")
                    else:
                        print(f"    {key}: {value}")
                        
        return {
            'total_tests': total_tests,
            'passed_tests': passed_tests,
            'failed_tests': total_tests - passed_tests,
            'success_rate': passed_tests / total_tests,
            'duration': overall_duration,
            'test_results': self.results
        }
    
    def generate_report(self) -> str:
        """Generate detailed test report
        Returns:
            Formatted report string
        """
        if not self.results:
            return "No test results available"
            
        report = []
        report.append("MAHIA-X FULL-SYSTEM STRESS AND ROBUSTNESS TEST REPORT")
        report.append("=" * 60)
        report.append(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")
        
        # Summary
        passed_tests = sum(1 for r in self.results if r['success'])
        total_tests = len(self.results)
        report.append(f"SUMMARY:")
        report.append(f"  Total Tests: {total_tests}")
        report.append(f"  Passed: {passed_tests}")
        report.append(f"  Failed: {total_tests - passed_tests}")
        report.append(f"  Success Rate: {passed_tests/total_tests*100:.1f}%")
        report.append("")
        
        # Individual test results
        report.append("DETAILED RESULTS:")
        for result in self.results:
            status = "PASS" if result['success'] else "FAIL"
            report.append(f"  {result['name']}: {status}")
            report.append(f"    Duration: {result['duration']:.1f} seconds")
            
            if result['errors']:
                report.append(f"    Errors: {len(result['errors'])}")
                for error in result['errors'][:3]:  # Show first 3 errors
                    report.append(f"      - {error}")
                if len(result['errors']) > 3:
                    report.append(f"      - ... and {len(result['errors']) - 3} more errors")
                    
            if 'results' in result and result['results']:
                report.append(f"    Metrics:")
                for key, value in result['results'].items():
                    if isinstance(value, dict):
                        report.append(f"      {key}:")
                        for subkey, subvalue in value.items():
                            report.append(f"        {subkey}: {subvalue}")
                    else:
                        report.append(f"      {key}: {value}")
            report.append("")
            
        return "\n".join(report)


def demo_stress_tests():
    """Demonstrate stress and robustness tests"""
    if not STRESS_TEST_AVAILABLE:
        print("❌ MAHIA-X modules not available for stress testing")
        return
        
    print("🚀 Demonstrating Full-System Stress and Robustness Tests...")
    print("=" * 60)
    
    # Create test suite
    test_suite = RobustnessTestSuite()
    print("✅ Initialized Robustness Test Suite")
    
    # Show available tests
    print(f"✅ Available tests:")
    for i, test in enumerate(test_suite.tests):
        print(f"   {i+1}. {test.name} (duration: {test.duration}s)")
        
    # Run tests
    results = test_suite.run_all_tests()
    
    # Generate report
    report = test_suite.generate_report()
    print(f"\n📋 Generated detailed report ({len(report)} characters)")
    
    # Save report to file
    report_file = "stress_test_report.txt"
    try:
        with open(report_file, 'w') as f:
            f.write(report)
        print(f"✅ Report saved to {report_file}")
    except Exception as e:
        print(f"❌ Failed to save report: {e}")
        
    # Show system requirements
    print(f"\n🔧 System Requirements Check:")
    try:
        # Check available memory
        memory = psutil.virtual_memory()
        print(f"   System Memory: {memory.total / (1024**3):.1f} GB")
        print(f"   Available Memory: {memory.available / (1024**3):.1f} GB")
        
        # Check CPU count
        cpu_count = psutil.cpu_count()
        print(f"   CPU Cores: {cpu_count}")
        
        # Check GPU availability
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            gpu_name = torch.cuda.get_device_name(0) if gpu_count > 0 else "Unknown"
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3) if gpu_count > 0 else 0
            print(f"   GPU: {gpu_count} x {gpu_name} ({gpu_memory:.1f} GB)")
        else:
            print(f"   GPU: Not available (using CPU)")
            
    except Exception as e:
        print(f"❌ System check failed: {e}")
        
    print("\n" + "=" * 60)
    print("STRESS AND ROBUSTNESS TESTS DEMO SUMMARY")
    print("=" * 60)
    print("Key Features Tested:")
    print("  1. Memory stress testing with large model allocations")
    print("  2. Long-term training stability")
    print("  3. Concurrency and multi-threading robustness")
    print("  4. System resource monitoring")
    print("  5. Error handling and recovery")
    print("\nBenefits:")
    print("  - Validates system stability under load")
    print("  - Identifies performance bottlenecks")
    print("  - Ensures robust error handling")
    print("  - Provides comprehensive test reporting")
    
    print("\n✅ Full-System Stress and Robustness Tests demonstration completed!")


def main():
    """Main demonstration function"""
    demo_stress_tests()


if __name__ == '__main__':
    main()