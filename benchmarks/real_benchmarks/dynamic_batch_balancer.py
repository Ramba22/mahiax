"""
Dynamic Batch Balancer for MAHIA
Adjusts batch size dynamically based on GPU utilization to minimize idle time
"""

import torch
import time
from typing import Optional, Dict, Any
import psutil
import threading

class GPUMonitor:
    """Monitor GPU utilization and memory usage"""
    
    def __init__(self, device_id: int = 0):
        self.device_id = device_id
        self.utilization_history = []
        self.memory_history = []
        
    def get_gpu_utilization(self) -> Optional[float]:
        """Get current GPU utilization percentage"""
        try:
            if torch.cuda.is_available():
                # Get current GPU utilization
                util = torch.cuda.utilization(self.device_id)
                self.utilization_history.append(util)
                # Keep only last 100 measurements
                if len(self.utilization_history) > 100:
                    self.utilization_history.pop(0)
                return float(util)
            else:
                # Fallback to CPU utilization
                cpu_util = psutil.cpu_percent()
                self.utilization_history.append(cpu_util)
                if len(self.utilization_history) > 100:
                    self.utilization_history.pop(0)
                return cpu_util
        except Exception as e:
            print(f"⚠️  Failed to get GPU utilization: {e}")
            return None
    
    def get_memory_usage(self) -> Optional[Dict[str, float]]:
        """Get current GPU memory usage"""
        try:
            if torch.cuda.is_available():
                allocated = torch.cuda.memory_allocated(self.device_id) / 1024 / 1024  # MB
                reserved = torch.cuda.memory_reserved(self.device_id) / 1024 / 1024    # MB
                max_allocated = torch.cuda.max_memory_allocated(self.device_id) / 1024 / 1024  # MB
                
                memory_info = {
                    "allocated_mb": allocated,
                    "reserved_mb": reserved,
                    "max_allocated_mb": max_allocated,
                    "utilization_percent": (allocated / reserved * 100) if reserved > 0 else 0
                }
                
                self.memory_history.append(memory_info)
                if len(self.memory_history) > 100:
                    self.memory_history.pop(0)
                    
                return memory_info
            else:
                # Fallback to system memory
                memory = psutil.virtual_memory()
                memory_info = {
                    "allocated_mb": memory.used / 1024 / 1024,
                    "reserved_mb": memory.total / 1024 / 1024,
                    "utilization_percent": memory.percent
                }
                self.memory_history.append(memory_info)
                if len(self.memory_history) > 100:
                    self.memory_history.pop(0)
                return memory_info
        except Exception as e:
            print(f"⚠️  Failed to get memory usage: {e}")
            return None

class DynamicBatchBalancer:
    """Dynamic batch size balancer to optimize GPU utilization"""
    
    def __init__(self, 
                 initial_batch_size: int = 32,
                 min_batch_size: int = 1,
                 max_batch_size: int = 256,
                 target_utilization: float = 80.0,
                 adaptation_rate: float = 0.1,
                 smoothing_factor: float = 0.3):
        """
        Initialize dynamic batch balancer
        
        Args:
            initial_batch_size: Starting batch size
            min_batch_size: Minimum allowed batch size
            max_batch_size: Maximum allowed batch size
            target_utilization: Target GPU utilization percentage
            adaptation_rate: How quickly to adapt batch size
            smoothing_factor: Smoothing factor for utilization measurements
        """
        self.current_batch_size = initial_batch_size
        self.min_batch_size = min_batch_size
        self.max_batch_size = max_batch_size
        self.target_utilization = target_utilization
        self.adaptation_rate = adaptation_rate
        self.smoothing_factor = smoothing_factor
        
        # GPU monitoring
        self.gpu_monitor = GPUMonitor()
        
        # Performance tracking
        self.batch_times = []
        self.throughput_history = []
        self.utilization_history = []
        
        # Adaptive parameters
        self.last_utilization = 0.0
        self.last_batch_time = 0.0
        self.stable_count = 0
        
    def get_current_utilization(self) -> float:
        """Get smoothed current GPU utilization"""
        raw_util = self.gpu_monitor.get_gpu_utilization()
        if raw_util is None:
            return self.last_utilization
            
        # Apply exponential smoothing
        if self.last_utilization == 0:
            smoothed = raw_util
        else:
            smoothed = (self.smoothing_factor * raw_util + 
                       (1 - self.smoothing_factor) * self.last_utilization)
        
        self.last_utilization = smoothed
        self.utilization_history.append(smoothed)
        
        # Keep only last 50 measurements
        if len(self.utilization_history) > 50:
            self.utilization_history.pop(0)
            
        return smoothed
    
    def adjust_batch_size(self) -> int:
        """Adjust batch size based on current GPU utilization"""
        current_util = self.get_current_utilization()
        
        # Calculate utilization error
        error = self.target_utilization - current_util
        
        # Adjust batch size based on error
        if abs(error) > 5.0:  # Only adjust if error is significant
            # Calculate adjustment amount
            adjustment = int(self.current_batch_size * self.adaptation_rate * (error / 100.0))
            
            # Apply adjustment with bounds checking
            new_batch_size = self.current_batch_size + adjustment
            new_batch_size = max(self.min_batch_size, min(self.max_batch_size, new_batch_size))
            
            # Only update if there's a meaningful change
            if abs(new_batch_size - self.current_batch_size) >= 1:
                old_size = self.current_batch_size
                self.current_batch_size = new_batch_size
                print(f"🔄 Batch size adjusted: {old_size} → {self.current_batch_size} "
                      f"(util: {current_util:.1f}%, target: {self.target_utilization}%)")
        
        return self.current_batch_size
    
    def record_batch_time(self, batch_time: float):
        """Record batch processing time for throughput analysis"""
        self.batch_times.append(batch_time)
        self.last_batch_time = batch_time
        
        # Calculate throughput (samples per second)
        if batch_time > 0:
            throughput = self.current_batch_size / batch_time
            self.throughput_history.append(throughput)
            
            # Keep only last 50 measurements
            if len(self.throughput_history) > 50:
                self.throughput_history.pop(0)
        
        # Keep only last 50 batch times
        if len(self.batch_times) > 50:
            self.batch_times.pop(0)
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get current performance metrics"""
        avg_utilization = (sum(self.utilization_history) / len(self.utilization_history) 
                          if self.utilization_history else 0)
        
        avg_batch_time = (sum(self.batch_times) / len(self.batch_times) 
                         if self.batch_times else 0)
        
        avg_throughput = (sum(self.throughput_history) / len(self.throughput_history) 
                         if self.throughput_history else 0)
        
        memory_info = self.gpu_monitor.get_memory_usage()
        
        return {
            "current_batch_size": self.current_batch_size,
            "current_utilization": self.last_utilization,
            "avg_utilization": avg_utilization,
            "avg_batch_time": avg_batch_time,
            "avg_throughput": avg_throughput,
            "memory_info": memory_info,
            "target_utilization": self.target_utilization
        }
    
    def is_stable(self, threshold: float = 2.0) -> bool:
        """Check if the system is stable (utilization near target)"""
        if not self.utilization_history:
            return False
            
        recent_util = self.utilization_history[-10:] if len(self.utilization_history) >= 10 else self.utilization_history
        if not recent_util:
            return False
            
        avg_recent = sum(recent_util) / len(recent_util)
        deviation = abs(avg_recent - self.target_utilization)
        
        return deviation < threshold

class BatchBalancedBenchmarkRunner:
    """Benchmark runner with dynamic batch balancing"""
    
    def __init__(self, model, device: Optional[str] = None):
        self.model = model
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        if hasattr(self.model, "to"):
            self.model.to(self.device)
        
        # Initialize batch balancer
        self.batch_balancer = DynamicBatchBalancer(
            initial_batch_size=32,
            min_batch_size=1,
            max_batch_size=128,
            target_utilization=75.0,
            adaptation_rate=0.15
        )
        
        # Benchmark settings
        self.warmup_batches = 5
        self.measurement_batches = 20
    
    def create_mock_data(self, batch_size: int, seq_length: int = 64):
        """Create mock data for benchmarking"""
        return {
            "input_ids": torch.randint(0, 1000, (batch_size, seq_length)),
            "attention_mask": torch.ones(batch_size, seq_length),
            "labels": torch.randint(0, 2, (batch_size,))
        }
    
    def run_balanced_benchmark(self, 
                              task_type: str = "glue",
                              max_batches: int = 50,
                              seq_length: int = 64) -> Dict[str, Any]:
        """
        Run benchmark with dynamic batch balancing
        
        Args:
            task_type: Type of task to benchmark
            max_batches: Maximum number of batches to process
            seq_length: Sequence length for input data
        """
        print(f"🚀 Running {task_type.upper()} benchmark with dynamic batch balancing...")
        
        # Warmup phase
        print("🔥 Warming up...")
        for i in range(self.warmup_batches):
            batch_size = self.batch_balancer.current_batch_size
            mock_data = self.create_mock_data(batch_size, seq_length)
            
            # Move to device
            for key in mock_data:
                if torch.is_tensor(mock_data[key]):
                    mock_data[key] = mock_data[key].to(self.device)
            
            # Forward pass
            start_time = time.time()
            try:
                if hasattr(self.model, "forward"):
                    outputs = self.model(
                        input_ids=mock_data["input_ids"],
                        attention_mask=mock_data["attention_mask"]
                    )
                else:
                    outputs = self.model(
                        mock_data["input_ids"],
                        mock_data["attention_mask"]
                    )
            except Exception:
                outputs = self.model(mock_data["input_ids"])
            
            batch_time = time.time() - start_time
            self.batch_balancer.record_batch_time(batch_time)
            
            if (i + 1) % 2 == 0:
                print(f"   Warmup batch {i+1}/{self.warmup_batches}")
        
        # Measurement phase
        print("📊 Measuring performance with dynamic batch balancing...")
        total_time = 0
        total_samples = 0
        batch_count = 0
        
        # Evaluation mode
        if hasattr(self.model, "eval"):
            self.model.eval()
        
        with torch.no_grad():
            for i in range(max_batches):
                # Adjust batch size based on GPU utilization
                current_batch_size = self.batch_balancer.adjust_batch_size()
                
                # Create mock data with current batch size
                mock_data = self.create_mock_data(current_batch_size, seq_length)
                
                # Move to device
                for key in mock_data:
                    if torch.is_tensor(mock_data[key]):
                        mock_data[key] = mock_data[key].to(self.device)
                
                # Forward pass
                start_time = time.time()
                try:
                    if hasattr(self.model, "forward"):
                        outputs = self.model(
                            input_ids=mock_data["input_ids"],
                            attention_mask=mock_data["attention_mask"]
                        )
                    else:
                        outputs = self.model(
                            mock_data["input_ids"],
                            mock_data["attention_mask"]
                        )
                except Exception:
                    outputs = self.model(mock_data["input_ids"])
                
                batch_time = time.time() - start_time
                total_time += batch_time
                total_samples += current_batch_size
                batch_count += 1
                
                # Record batch time for adaptation
                self.batch_balancer.record_batch_time(batch_time)
                
                # Print progress
                if (i + 1) % 5 == 0:
                    metrics = self.batch_balancer.get_performance_metrics()
                    throughput = current_batch_size / batch_time if batch_time > 0 else 0
                    print(f"   Batch {i+1}/{max_batches} - "
                          f"Size: {current_batch_size}, "
                          f"Time: {batch_time:.4f}s, "
                          f"Throughput: {throughput:.1f} samples/s, "
                          f"GPU: {metrics['current_utilization']:.1f}%")
                
                # Check for stability
                if self.batch_balancer.is_stable() and i > 10:
                    print("✅ System stable, continuing with optimal batch size...")
        
        # Calculate final metrics
        avg_batch_time = total_time / batch_count if batch_count > 0 else 0
        overall_throughput = total_samples / total_time if total_time > 0 else 0
        
        final_metrics = self.batch_balancer.get_performance_metrics()
        
        results = {
            "task_type": task_type,
            "total_batches": batch_count,
            "total_samples": total_samples,
            "total_time": total_time,
            "avg_batch_time": avg_batch_time,
            "overall_throughput": overall_throughput,
            "final_batch_size": self.batch_balancer.current_batch_size,
            "final_utilization": final_metrics["current_utilization"],
            "avg_utilization": final_metrics["avg_utilization"],
            "memory_info": final_metrics["memory_info"]
        }
        
        print(f"✅ Benchmark Complete:")
        print(f"   Final Batch Size: {results['final_batch_size']}")
        print(f"   Overall Throughput: {results['overall_throughput']:.1f} samples/s")
        print(f"   Average GPU Utilization: {results['avg_utilization']:.1f}%")
        print(f"   Total Time: {results['total_time']:.2f}s")
        
        return results

# Example usage
def example_dynamic_batch_balancing():
    """Example of dynamic batch balancing"""
    print("🔧 Setting up dynamic batch balancing example...")
    
    # Simple model for demonstration
    import torch.nn as nn
    
    class SimpleModel(nn.Module):
        def __init__(self, vocab_size=1000, hidden_size=128, num_classes=2):
            super().__init__()
            self.embedding = nn.Embedding(vocab_size, hidden_size)
            self.transformer_layer = nn.TransformerEncoderLayer(
                d_model=hidden_size, 
                nhead=8,
                dim_feedforward=hidden_size * 4,
                batch_first=True
            )
            self.classifier = nn.Linear(hidden_size, num_classes)
            
        def forward(self, input_ids, attention_mask=None):
            x = self.embedding(input_ids)
            x = self.transformer_layer(x)
            x = x.mean(dim=1)  # Global average pooling
            logits = self.classifier(x)
            return logits
    
    # Create model
    model = SimpleModel()
    print(f"✅ Created model with {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M parameters")
    
    # Create benchmark runner with dynamic batch balancing
    benchmark = BatchBalancedBenchmarkRunner(model)
    
    # Run benchmark
    print("\n" + "="*60)
    results = benchmark.run_balanced_benchmark(
        task_type="glue",
        max_batches=15,
        seq_length=64
    )
    
    # Print final metrics
    print("\n" + "="*60)
    print("📈 Final Performance Metrics:")
    metrics = benchmark.batch_balancer.get_performance_metrics()
    print(f"   Current Batch Size: {metrics['current_batch_size']}")
    print(f"   Current GPU Utilization: {metrics['current_utilization']:.1f}%")
    print(f"   Average Throughput: {metrics['avg_throughput']:.1f} samples/s")
    
    if metrics["memory_info"]:
        mem_info = metrics["memory_info"]
        print(f"   Memory Allocated: {mem_info['allocated_mb']:.1f}MB")
        print(f"   Memory Reserved: {mem_info['reserved_mb']:.1f}MB")
    
    return results

if __name__ == "__main__":
    example_dynamic_batch_balancing()