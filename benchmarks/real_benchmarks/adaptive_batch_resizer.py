"""
Adaptive Batch Resizer for MAHIA
Dynamically adjusts batch sizes based on available GPU memory to prevent OOM errors
"""

import torch
import time
from typing import Dict, Optional, Any, List
import threading
from collections import deque

class GPUMemoryMonitor:
    """Monitor GPU memory usage and available free memory"""
    
    def __init__(self, device_id: int = 0, safety_margin: float = 0.1):
        """
        Initialize GPU memory monitor
        
        Args:
            device_id: GPU device ID to monitor
            safety_margin: Safety margin to keep free (0.1 = 10%)
        """
        self.device_id = device_id
        self.safety_margin = safety_margin
        self.gpu_available = torch.cuda.is_available()
        
        # Memory history tracking
        self.memory_history = deque(maxlen=100)
        self.lock = threading.Lock()
        
        print(f"✅ GPUMemoryMonitor initialized for device {device_id}")
        if self.gpu_available:
            props = torch.cuda.get_device_properties(self.device_id)
            print(f"   GPU: {props.name}")
            print(f"   Total Memory: {props.total_memory / (1024**3):.2f} GB")
        
    def get_memory_info(self) -> Optional[Dict[str, float]]:
        """
        Get comprehensive GPU memory information
        
        Returns:
            Dictionary with memory metrics or None if GPU not available
        """
        if not self.gpu_available:
            return None
            
        try:
            with self.lock:
                allocated = torch.cuda.memory_allocated(self.device_id)
                reserved = torch.cuda.memory_reserved(self.device_id)
                max_allocated = torch.cuda.max_memory_allocated(self.device_id)
                
                total_memory = torch.cuda.get_device_properties(self.device_id).total_memory
                
                # Calculate free memory
                free_memory = total_memory - allocated
                
                memory_info = {
                    "allocated_bytes": float(allocated),
                    "reserved_bytes": float(reserved),
                    "max_allocated_bytes": float(max_allocated),
                    "total_bytes": float(total_memory),
                    "free_bytes": float(free_memory),
                    "utilization_percent": (allocated / total_memory) * 100 if total_memory > 0 else 0,
                    "free_percent": (free_memory / total_memory) * 100 if total_memory > 0 else 0,
                    "allocated_gb": allocated / (1024**3),
                    "reserved_gb": reserved / (1024**3),
                    "total_gb": total_memory / (1024**3),
                    "free_gb": free_memory / (1024**3)
                }
                
                # Store in history
                self.memory_history.append((time.time(), memory_info))
                
                return memory_info
                
        except Exception as e:
            print(f"⚠️  Failed to get GPU memory info: {e}")
            return None
            
    def get_free_memory_gb(self) -> float:
        """
        Get available free GPU memory in GB
        
        Returns:
            Free memory in GB
        """
        memory_info = self.get_memory_info()
        if memory_info:
            return memory_info["free_gb"]
        return 0.0
        
    def get_safe_batch_size(self, 
                          current_batch_size: int, 
                          memory_per_sample_gb: float,
                          target_free_memory_gb: Optional[float] = None) -> int:
        """
        Calculate safe batch size based on available memory
        
        Args:
            current_batch_size: Current batch size
            memory_per_sample_gb: Memory usage per sample in GB
            target_free_memory_gb: Target free memory to maintain (None for auto)
            
        Returns:
            Safe batch size
        """
        if not self.gpu_available:
            return current_batch_size
            
        memory_info = self.get_memory_info()
        if not memory_info:
            return current_batch_size
            
        free_memory_gb = memory_info["free_gb"]
        total_memory_gb = memory_info["total_gb"]
        
        # Calculate target free memory
        if target_free_memory_gb is None:
            target_free_memory_gb = total_memory_gb * self.safety_margin
            
        # Calculate maximum safe batch size
        available_memory_gb = max(0, free_memory_gb - target_free_memory_gb)
        max_safe_batch_size = int(available_memory_gb / memory_per_sample_gb) if memory_per_sample_gb > 0 else current_batch_size
        
        # Ensure we don't go below minimum batch size
        min_batch_size = 1
        max_safe_batch_size = max(min_batch_size, max_safe_batch_size)
        
        # Adjust batch size gradually to prevent drastic changes
        if max_safe_batch_size < current_batch_size:
            # Reduce batch size more conservatively
            new_batch_size = max(min_batch_size, int(current_batch_size * 0.8))
            new_batch_size = min(new_batch_size, max_safe_batch_size)
        elif max_safe_batch_size > current_batch_size:
            # Increase batch size gradually
            new_batch_size = min(int(current_batch_size * 1.2), max_safe_batch_size)
        else:
            new_batch_size = current_batch_size
            
        return new_batch_size
        
    def get_memory_pressure_score(self) -> float:
        """
        Get memory pressure score (0-1, where 1 is high pressure)
        
        Returns:
            Memory pressure score
        """
        memory_info = self.get_memory_info()
        if memory_info:
            return memory_info["utilization_percent"] / 100.0
        return 0.0
        
    def is_memory_critical(self, threshold: float = 0.9) -> bool:
        """
        Check if memory usage is critical
        
        Args:
            threshold: Utilization threshold (0.9 = 90%)
            
        Returns:
            True if memory is critical
        """
        memory_info = self.get_memory_info()
        if memory_info:
            return memory_info["utilization_percent"] / 100.0 > threshold
        return False

class AdaptiveBatchResizer:
    """Adaptive batch resizer that adjusts batch sizes based on GPU memory"""
    
    def __init__(self,
                 initial_batch_size: int = 32,
                 min_batch_size: int = 1,
                 max_batch_size: int = 256,
                 device_id: int = 0,
                 safety_margin: float = 0.15,
                 memory_per_sample_estimate_gb: float = 0.01,
                 adaptation_rate: float = 0.1):
        """
        Initialize adaptive batch resizer
        
        Args:
            initial_batch_size: Starting batch size
            min_batch_size: Minimum allowed batch size
            max_batch_size: Maximum allowed batch size
            device_id: GPU device ID to monitor
            safety_margin: Safety margin for free memory (0.15 = 15%)
            memory_per_sample_estimate_gb: Initial estimate of memory per sample
            adaptation_rate: How quickly to adapt to memory conditions
        """
        self.initial_batch_size = initial_batch_size
        self.min_batch_size = min_batch_size
        self.max_batch_size = max_batch_size
        self.safety_margin = safety_margin
        self.memory_per_sample_estimate_gb = memory_per_sample_estimate_gb
        self.adaptation_rate = adaptation_rate
        
        # GPU monitoring
        self.memory_monitor = GPUMemoryMonitor(device_id, safety_margin)
        
        # Current state
        self.current_batch_size = initial_batch_size
        self.last_adjustment_time = time.time()
        self.adjustment_cooldown = 2.0  # Seconds between adjustments
        
        # Performance tracking
        self.batch_time_history = deque(maxlen=50)
        self.memory_usage_history = deque(maxlen=50)
        self.resize_history = deque(maxlen=100)
        
        # Statistics
        self.resizes = 0
        self.oom_avoided = 0
        self.batch_timeouts = 0
        
        print(f"🚀 AdaptiveBatchResizer initialized")
        print(f"   Initial batch size: {initial_batch_size}")
        print(f"   Memory safety margin: {safety_margin*100:.1f}%")
        print(f"   Estimated memory per sample: {memory_per_sample_estimate_gb*1000:.2f} MB")
        
    def estimate_memory_per_sample(self, batch_size: int, batch_time: float) -> float:
        """
        Estimate memory usage per sample based on batch processing
        
        Args:
            batch_size: Size of processed batch
            batch_time: Time taken to process batch
            
        Returns:
            Estimated memory per sample in GB
        """
        if batch_size <= 0:
            return self.memory_per_sample_estimate_gb
            
        memory_info = self.memory_monitor.get_memory_info()
        if not memory_info:
            return self.memory_per_sample_estimate_gb
            
        # Estimate memory per sample based on allocated memory change
        allocated_gb = memory_info["allocated_gb"]
        
        # Simple estimation - assume memory scales linearly with batch size
        # This is a rough estimate and would be more accurate with actual profiling
        estimated_memory_per_sample = allocated_gb / batch_size if batch_size > 0 else 0
        
        # Apply exponential smoothing to the estimate
        if self.memory_per_sample_estimate_gb > 0:
            self.memory_per_sample_estimate_gb = (
                self.adaptation_rate * estimated_memory_per_sample +
                (1 - self.adaptation_rate) * self.memory_per_sample_estimate_gb
            )
        else:
            self.memory_per_sample_estimate_gb = estimated_memory_per_sample
            
        return self.memory_per_sample_estimate_gb
        
    def adjust_batch_size(self, force: bool = False) -> int:
        """
        Adjust batch size based on current memory conditions
        
        Args:
            force: Force adjustment regardless of cooldown
            
        Returns:
            New batch size
        """
        current_time = time.time()
        
        # Check cooldown
        if not force and (current_time - self.last_adjustment_time) < self.adjustment_cooldown:
            return self.current_batch_size
            
        self.last_adjustment_time = current_time
        
        # Get current memory state
        memory_info = self.memory_monitor.get_memory_info()
        if not memory_info:
            return self.current_batch_size
            
        # Check for critical memory conditions
        if self.memory_monitor.is_memory_critical(0.95):
            # Emergency reduction
            new_batch_size = max(self.min_batch_size, int(self.current_batch_size * 0.5))
            if new_batch_size != self.current_batch_size:
                self._record_resize(self.current_batch_size, new_batch_size, "emergency_reduction")
                self.current_batch_size = new_batch_size
                self.oom_avoided += 1
                print(f"🚨 Emergency batch size reduction: {self.current_batch_size} → {new_batch_size}")
            return self.current_batch_size
            
        # Calculate safe batch size
        safe_batch_size = self.memory_monitor.get_safe_batch_size(
            self.current_batch_size,
            self.memory_per_sample_estimate_gb
        )
        
        # Apply bounds
        safe_batch_size = max(self.min_batch_size, min(self.max_batch_size, safe_batch_size))
        
        # Gradual adjustment
        if safe_batch_size != self.current_batch_size:
            old_size = self.current_batch_size
            self.current_batch_size = safe_batch_size
            self._record_resize(old_size, self.current_batch_size, "memory_based")
            self.resizes += 1
            print(f"🔄 Batch size adjusted: {old_size} → {self.current_batch_size} "
                  f"(free memory: {memory_info['free_gb']:.2f} GB)")
                  
        return self.current_batch_size
        
    def _record_resize(self, old_size: int, new_size: int, reason: str):
        """Record batch size resize event"""
        self.resize_history.append({
            "timestamp": time.time(),
            "old_size": old_size,
            "new_size": new_size,
            "reason": reason,
            "memory_pressure": self.memory_monitor.get_memory_pressure_score()
        })
        
    def record_batch_completion(self, batch_size: int, batch_time: float, success: bool = True):
        """
        Record completion of a batch for performance tracking
        
        Args:
            batch_size: Size of completed batch
            batch_time: Time taken to process batch
            success: Whether batch processing was successful
        """
        self.batch_time_history.append((time.time(), batch_time, batch_size, success))
        
        # Update memory estimate
        if success and batch_size > 0:
            self.estimate_memory_per_sample(batch_size, batch_time)
            
        # Record memory usage
        memory_info = self.memory_monitor.get_memory_info()
        if memory_info:
            self.memory_usage_history.append((time.time(), memory_info))
            
        # Check if we should adjust batch size
        if len(self.batch_time_history) % 5 == 0:  # Adjust every 5 batches
            self.adjust_batch_size()
            
    def get_batch_size(self) -> int:
        """
        Get current batch size
        
        Returns:
            Current batch size
        """
        return self.current_batch_size
        
    def get_performance_summary(self) -> Dict[str, Any]:
        """
        Get performance summary
        
        Returns:
            Dictionary with performance metrics
        """
        memory_info = self.memory_monitor.get_memory_info()
        
        # Calculate average batch time
        avg_batch_time = 0.0
        if self.batch_time_history:
            successful_times = [bt for _, bt, _, success in self.batch_time_history if success]
            if successful_times:
                avg_batch_time = sum(successful_times) / len(successful_times)
                
        # Calculate throughput
        throughput = 0.0
        if avg_batch_time > 0:
            throughput = self.current_batch_size / avg_batch_time
            
        return {
            "current_batch_size": self.current_batch_size,
            "min_batch_size": self.min_batch_size,
            "max_batch_size": self.max_batch_size,
            "memory_per_sample_estimate_gb": self.memory_per_sample_estimate_gb,
            "memory_per_sample_estimate_mb": self.memory_per_sample_estimate_gb * 1024,
            "avg_batch_time": avg_batch_time,
            "throughput_samples_per_second": throughput,
            "resizes": self.resizes,
            "oom_avoided": self.oom_avoided,
            "batch_timeouts": self.batch_timeouts,
            "memory_info": memory_info,
            "memory_pressure_score": self.memory_monitor.get_memory_pressure_score()
        }
        
    def reset(self):
        """Reset to initial state"""
        self.current_batch_size = self.initial_batch_size
        self.last_adjustment_time = time.time()
        self.batch_time_history.clear()
        self.memory_usage_history.clear()
        self.resize_history.clear()
        self.resizes = 0
        self.oom_avoided = 0
        self.batch_timeouts = 0
        self.memory_per_sample_estimate_gb = 0.01  # Reset to default estimate
        print("🔄 AdaptiveBatchResizer reset to initial state")

# Example usage and testing
class DummyModel(torch.nn.Module):
    """Dummy model for testing batch resizer"""
    
    def __init__(self, hidden_size: int = 512):
        super().__init__()
        self.embedding = torch.nn.Embedding(10000, hidden_size)
        self.transformer_layers = torch.nn.ModuleList([
            torch.nn.TransformerEncoderLayer(
                d_model=hidden_size,
                nhead=8,
                dim_feedforward=hidden_size * 4,
                batch_first=True
            ) for _ in range(6)
        ])
        self.classifier = torch.nn.Linear(hidden_size, 1000)
        
    def forward(self, input_ids, attention_mask=None):
        x = self.embedding(input_ids)
        for layer in self.transformer_layers:
            x = layer(x)
        x = x.mean(dim=1)  # Global average pooling
        logits = self.classifier(x)
        return logits

def example_adaptive_batch_resizing():
    """Example of adaptive batch resizing usage"""
    print("🔧 Setting up adaptive batch resizer example...")
    
    # Check if CUDA is available
    if not torch.cuda.is_available():
        print("⚠️  CUDA not available, using CPU (limited memory monitoring)")
        
    # Create adaptive batch resizer
    batch_resizer = AdaptiveBatchResizer(
        initial_batch_size=32,
        min_batch_size=1,
        max_batch_size=128,
        device_id=0,
        safety_margin=0.2,  # 20% safety margin
        memory_per_sample_estimate_gb=0.005,  # 5MB per sample estimate
        adaptation_rate=0.1
    )
    
    # Simulate training loop
    print("\n🔄 Simulating training with adaptive batch resizing...")
    for step in range(30):
        # Get current batch size
        batch_size = batch_resizer.get_batch_size()
        
        # Simulate batch processing
        # In a real scenario, this would be actual model training
        processing_time = 0.05 + (batch_size / 200.0) + (torch.rand(1).item() * 0.02)
        
        # Simulate memory usage increasing with batch size
        memory_pressure = min(1.0, batch_size / 128.0)
        
        # Simulate occasional high memory usage
        if step % 10 == 7:
            memory_pressure = min(1.0, memory_pressure + 0.3)
            
        # Simulate processing
        time.sleep(processing_time)
        
        # Record completion (success or failure based on memory pressure)
        success = memory_pressure < 0.95  # Fail if memory pressure too high
        if not success:
            print(f"⚠️  Simulated OOM at step {step} with batch size {batch_size}")
            
        batch_resizer.record_batch_completion(batch_size, processing_time, success)
        
        # Print progress periodically
        if step % 5 == 0:
            summary = batch_resizer.get_performance_summary()
            memory_info = summary["memory_info"]
            if memory_info:
                print(f"Step {step}: Batch size {batch_size}, "
                      f"Free memory {memory_info['free_gb']:.2f} GB, "
                      f"Utilization {memory_info['utilization_percent']:.1f}%")
            else:
                print(f"Step {step}: Batch size {batch_size}")
    
    # Print final summary
    print("\n" + "="*60)
    print("📈 Adaptive Batch Resizer Final Summary:")
    summary = batch_resizer.get_performance_summary()
    print(f"   Current Batch Size: {summary['current_batch_size']}")
    print(f"   Memory per Sample: {summary['memory_per_sample_estimate_mb']:.2f} MB")
    print(f"   Average Batch Time: {summary['avg_batch_time']:.3f}s")
    print(f"   Throughput: {summary['throughput_samples_per_second']:.1f} samples/s")
    print(f"   Batch Resizes: {summary['resizes']}")
    print(f"   OOM Avoided: {summary['oom_avoided']}")
    
    if summary["memory_info"]:
        mem_info = summary["memory_info"]
        print(f"   Current Free Memory: {mem_info['free_gb']:.2f} GB")
        print(f"   Memory Utilization: {mem_info['utilization_percent']:.1f}%")
        
    print("\n✅ Adaptive batch resizing example completed!")

if __name__ == "__main__":
    example_adaptive_batch_resizing()