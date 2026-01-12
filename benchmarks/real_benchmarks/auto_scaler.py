"""
Auto-Scaler for MAHIA
Dynamic batch scaling across multiple GPUs with load balancing and performance optimization
"""

import torch
import time
from typing import Optional, Dict, Any, List, Tuple
import threading
from collections import deque
import math

class MultiGPUMonitor:
    """Monitor multiple GPUs for utilization, memory usage, and performance metrics"""
    
    def __init__(self, device_ids: Optional[List[int]] = None):
        """
        Initialize multi-GPU monitor
        
        Args:
            device_ids: List of GPU device IDs to monitor (None for all available)
        """
        if device_ids is None:
            if torch.cuda.is_available():
                device_ids = list(range(torch.cuda.device_count()))
            else:
                device_ids = [0]  # CPU fallback
                
        self.device_ids = device_ids
        self.device_monitors = {}
        
        # Initialize monitor for each device
        for device_id in self.device_ids:
            self.device_monitors[device_id] = {
                "utilization_history": deque(maxlen=100),
                "memory_history": deque(maxlen=100),
                "batch_time_history": deque(maxlen=100),
                "throughput_history": deque(maxlen=100)
            }
            
        print(f"✅ MultiGPUMonitor initialized for devices: {self.device_ids}")
        
    def get_gpu_utilization(self, device_id: int) -> Optional[float]:
        """
        Get current GPU utilization for a specific device
        
        Args:
            device_id: GPU device ID
            
        Returns:
            GPU utilization percentage or None if not available
        """
        try:
            if torch.cuda.is_available() and device_id < torch.cuda.device_count():
                # Get current GPU utilization
                util = torch.cuda.utilization(device_id)
                self.device_monitors[device_id]["utilization_history"].append((time.time(), util))
                return float(util)
            else:
                # Fallback to 0 for CPU or unavailable devices
                return 0.0
        except Exception as e:
            print(f"⚠️  Failed to get GPU {device_id} utilization: {e}")
            return None
            
    def get_memory_usage(self, device_id: int) -> Optional[Dict[str, float]]:
        """
        Get current memory usage for a specific device
        
        Args:
            device_id: GPU device ID
            
        Returns:
            Dictionary with memory usage metrics or None if not available
        """
        try:
            if torch.cuda.is_available() and device_id < torch.cuda.device_count():
                allocated = torch.cuda.memory_allocated(device_id) / (1024 * 1024)  # MB
                reserved = torch.cuda.memory_reserved(device_id) / (1024 * 1024)    # MB
                max_allocated = torch.cuda.max_memory_allocated(device_id) / (1024 * 1024)  # MB
                
                memory_info = {
                    "allocated_mb": allocated,
                    "reserved_mb": reserved,
                    "max_allocated_mb": max_allocated,
                    "utilization_percent": (allocated / reserved * 100) if reserved > 0 else 0
                }
                
                self.device_monitors[device_id]["memory_history"].append((time.time(), memory_info))
                return memory_info
            else:
                # Fallback for CPU or unavailable devices
                return {
                    "allocated_mb": 0.0,
                    "reserved_mb": 0.0,
                    "max_allocated_mb": 0.0,
                    "utilization_percent": 0.0
                }
        except Exception as e:
            print(f"⚠️  Failed to get GPU {device_id} memory usage: {e}")
            return None
            
    def record_batch_time(self, device_id: int, batch_time: float):
        """
        Record batch processing time for a specific device
        
        Args:
            device_id: GPU device ID
            batch_time: Time taken to process batch (seconds)
        """
        if device_id in self.device_monitors:
            self.device_monitors[device_id]["batch_time_history"].append((time.time(), batch_time))
            
            # Calculate throughput (samples per second)
            # We need the batch size to calculate this, so we'll store batch_time for now
            # and let the caller calculate throughput if needed
            
    def get_device_metrics(self, device_id: int) -> Dict[str, Any]:
        """
        Get comprehensive metrics for a specific device
        
        Args:
            device_id: GPU device ID
            
        Returns:
            Dictionary with device metrics
        """
        if device_id not in self.device_monitors:
            return {}
            
        monitor = self.device_monitors[device_id]
        
        # Calculate averages
        avg_utilization = 0.0
        if monitor["utilization_history"]:
            avg_utilization = sum(util for _, util in monitor["utilization_history"]) / len(monitor["utilization_history"])
            
        avg_batch_time = 0.0
        if monitor["batch_time_history"]:
            avg_batch_time = sum(bt for _, bt in monitor["batch_time_history"]) / len(monitor["batch_time_history"])
            
        memory_info = self.get_memory_usage(device_id) or {}
        
        return {
            "device_id": device_id,
            "current_utilization": self.get_gpu_utilization(device_id),
            "avg_utilization": avg_utilization,
            "current_batch_time": batch_time if (batch_time := self._get_latest_batch_time(device_id)) else 0.0,
            "avg_batch_time": avg_batch_time,
            "memory_info": memory_info
        }
        
    def _get_latest_batch_time(self, device_id: int) -> Optional[float]:
        """Get the most recent batch time for a device"""
        if device_id in self.device_monitors and self.device_monitors[device_id]["batch_time_history"]:
            return self.device_monitors[device_id]["batch_time_history"][-1][1]
        return None
        
    def get_all_device_metrics(self) -> Dict[int, Dict[str, Any]]:
        """
        Get metrics for all monitored devices
        
        Returns:
            Dictionary mapping device IDs to their metrics
        """
        return {device_id: self.get_device_metrics(device_id) for device_id in self.device_ids}

class AutoScaler:
    """Auto-scaler for dynamic batch scaling across multiple GPUs"""
    
    def __init__(self,
                 device_ids: Optional[List[int]] = None,
                 initial_batch_size: int = 32,
                 min_batch_size: int = 1,
                 max_batch_size: int = 256,
                 target_utilization: float = 80.0,
                 adaptation_rate: float = 0.1,
                 smoothing_factor: float = 0.3,
                 load_balancing_strategy: str = "round_robin",
                 scaling_strategy: str = "proportional"):
        """
        Initialize auto-scaler
        
        Args:
            device_ids: List of GPU device IDs to manage (None for all available)
            initial_batch_size: Starting batch size per device
            min_batch_size: Minimum batch size per device
            max_batch_size: Maximum batch size per device
            target_utilization: Target GPU utilization percentage
            adaptation_rate: How quickly to adapt batch sizes
            smoothing_factor: Smoothing factor for utilization measurements
            load_balancing_strategy: How to distribute batches ("round_robin", "utilization_based", "performance_based")
            scaling_strategy: How to scale batches ("proportional", "adaptive", "conservative")
        """
        self.device_ids = device_ids or (list(range(torch.cuda.device_count())) if torch.cuda.is_available() else [0])
        self.initial_batch_size = initial_batch_size
        self.min_batch_size = min_batch_size
        self.max_batch_size = max_batch_size
        self.target_utilization = target_utilization
        self.adaptation_rate = adaptation_rate
        self.smoothing_factor = smoothing_factor
        self.load_balancing_strategy = load_balancing_strategy
        self.scaling_strategy = scaling_strategy
        
        # GPU monitoring
        self.gpu_monitor = MultiGPUMonitor(self.device_ids)
        
        # Batch size tracking per device
        self.device_batch_sizes = {device_id: initial_batch_size for device_id in self.device_ids}
        
        # Performance tracking
        self.global_throughput_history = deque(maxlen=100)
        self.scaling_history = deque(maxlen=1000)
        
        # Load balancing state
        self.current_device_index = 0
        
        # Adaptive parameters
        self.last_global_utilization = 0.0
        self.stable_count = 0
        self.last_scaling_time = time.time()
        
        print(f"🚀 AutoScaler initialized for {len(self.device_ids)} devices")
        print(f"   Initial batch size: {initial_batch_size}")
        print(f"   Target utilization: {target_utilization}%")
        print(f"   Load balancing: {load_balancing_strategy}")
        print(f"   Scaling strategy: {scaling_strategy}")
        
    def get_next_device(self) -> int:
        """
        Get the next device for batch assignment based on load balancing strategy
        
        Returns:
            Device ID for next batch
        """
        if self.load_balancing_strategy == "round_robin":
            device_id = self.device_ids[self.current_device_index]
            self.current_device_index = (self.current_device_index + 1) % len(self.device_ids)
            return device_id
            
        elif self.load_balancing_strategy == "utilization_based":
            # Assign to device with lowest current utilization
            metrics = self.gpu_monitor.get_all_device_metrics()
            if metrics:
                min_util_device = min(metrics.keys(), key=lambda d: metrics[d].get("current_utilization", 0))
                return min_util_device
            else:
                # Fallback to round-robin
                device_id = self.device_ids[self.current_device_index]
                self.current_device_index = (self.current_device_index + 1) % len(self.device_ids)
                return device_id
                
        elif self.load_balancing_strategy == "performance_based":
            # Assign to device with best recent performance (lowest batch time)
            metrics = self.gpu_monitor.get_all_device_metrics()
            if metrics:
                min_time_device = min(
                    metrics.keys(), 
                    key=lambda d: metrics[d].get("current_batch_time", float('inf'))
                )
                return min_time_device
            else:
                # Fallback to round-robin
                device_id = self.device_ids[self.current_device_index]
                self.current_device_index = (self.current_device_index + 1) % len(self.device_ids)
                return device_id
                
        else:
            # Default to round-robin
            device_id = self.device_ids[self.current_device_index]
            self.current_device_index = (self.current_device_index + 1) % len(self.device_ids)
            return device_id
            
    def get_global_utilization(self) -> float:
        """
        Get average utilization across all devices
        
        Returns:
            Average GPU utilization percentage
        """
        metrics = self.gpu_monitor.get_all_device_metrics()
        if not metrics:
            return self.last_global_utilization
            
        utilizations = [
            metrics[device_id].get("current_utilization", 0) 
            for device_id in metrics.keys()
        ]
        
        if utilizations:
            avg_util = sum(utilizations) / len(utilizations)
            # Apply exponential smoothing
            if self.last_global_utilization == 0:
                smoothed = avg_util
            else:
                smoothed = (self.smoothing_factor * avg_util + 
                           (1 - self.smoothing_factor) * self.last_global_utilization)
            self.last_global_utilization = smoothed
            return smoothed
        else:
            return self.last_global_utilization
            
    def adjust_batch_sizes(self):
        """
        Adjust batch sizes for all devices based on current utilization
        """
        current_time = time.time()
        
        # Only adjust every 5 seconds to prevent excessive scaling
        if current_time - self.last_scaling_time < 5.0:
            return
            
        self.last_scaling_time = current_time
        global_util = self.get_global_utilization()
        
        # Calculate utilization error
        error = self.target_utilization - global_util
        
        # Only adjust if error is significant
        if abs(error) > 5.0:
            # Calculate adjustment amount based on scaling strategy
            if self.scaling_strategy == "proportional":
                adjustment_factor = self.adaptation_rate * (error / 100.0)
            elif self.scaling_strategy == "adaptive":
                # More aggressive scaling when error is large
                adjustment_factor = self.adaptation_rate * (error / 50.0)
            else:  # conservative
                adjustment_factor = self.adaptation_rate * (error / 200.0)
                
            # Apply adjustment to all devices
            for device_id in self.device_ids:
                current_size = self.device_batch_sizes[device_id]
                adjustment = int(current_size * adjustment_factor)
                new_size = current_size + adjustment
                new_size = max(self.min_batch_size, min(self.max_batch_size, new_size))
                
                # Only update if there's a meaningful change
                if abs(new_size - current_size) >= 1:
                    old_size = current_size
                    self.device_batch_sizes[device_id] = new_size
                    
                    # Record scaling event
                    self.scaling_history.append({
                        "timestamp": current_time,
                        "device_id": device_id,
                        "old_size": old_size,
                        "new_size": new_size,
                        "global_utilization": global_util,
                        "error": error
                    })
                    
                    print(f"🔄 Device {device_id} batch size: {old_size} → {new_size} "
                          f"(global util: {global_util:.1f}%, target: {self.target_utilization}%)")
                          
    def get_batch_size_for_device(self, device_id: int) -> int:
        """
        Get current batch size for a specific device
        
        Args:
            device_id: GPU device ID
            
        Returns:
            Current batch size for the device
        """
        return self.device_batch_sizes.get(device_id, self.initial_batch_size)
        
    def record_batch_completion(self, device_id: int, batch_time: float, batch_size: int):
        """
        Record completion of a batch for performance tracking
        
        Args:
            device_id: GPU device ID
            batch_time: Time taken to process batch (seconds)
            batch_size: Size of the batch
        """
        # Record in GPU monitor
        self.gpu_monitor.record_batch_time(device_id, batch_time)
        
        # Calculate throughput (samples per second)
        if batch_time > 0:
            throughput = batch_size / batch_time
            self.global_throughput_history.append((time.time(), throughput))
            
            # Print performance info periodically
            if len(self.global_throughput_history) % 10 == 0:
                avg_throughput = sum(t for _, t in self.global_throughput_history) / len(self.global_throughput_history)
                print(f"📊 Global throughput: {avg_throughput:.1f} samples/s")
                
    def get_scaling_summary(self) -> Dict[str, Any]:
        """
        Get a summary of the current scaling state
        
        Returns:
            Dictionary with scaling summary information
        """
        metrics = self.gpu_monitor.get_all_device_metrics()
        avg_throughput = 0.0
        if self.global_throughput_history:
            avg_throughput = sum(t for _, t in self.global_throughput_history) / len(self.global_throughput_history)
            
        return {
            "device_count": len(self.device_ids),
            "device_batch_sizes": dict(self.device_batch_sizes),
            "global_utilization": self.get_global_utilization(),
            "target_utilization": self.target_utilization,
            "avg_throughput": avg_throughput,
            "device_metrics": metrics,
            "scaling_events": len(self.scaling_history),
            "load_balancing_strategy": self.load_balancing_strategy,
            "scaling_strategy": self.scaling_strategy
        }
        
    def reset_scaling(self):
        """Reset scaling to initial state"""
        for device_id in self.device_ids:
            self.device_batch_sizes[device_id] = self.initial_batch_size
        self.last_global_utilization = 0.0
        self.stable_count = 0
        self.global_throughput_history.clear()
        self.scaling_history.clear()
        print("🔄 AutoScaler reset to initial state")

# Example usage and testing
class DummyModel(torch.nn.Module):
    """Dummy model for testing auto-scaler"""
    
    def __init__(self, hidden_size: int = 256):
        super().__init__()
        self.embedding = torch.nn.Embedding(1000, hidden_size)
        self.transformer_layer = torch.nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=8,
            dim_feedforward=hidden_size * 4,
            batch_first=True
        )
        self.classifier = torch.nn.Linear(hidden_size, 10)
        
    def forward(self, input_ids, attention_mask=None):
        x = self.embedding(input_ids)
        x = self.transformer_layer(x)
        x = x.mean(dim=1)  # Global average pooling
        logits = self.classifier(x)
        return logits

def example_auto_scaling():
    """Example of auto-scaler usage"""
    print("🔧 Setting up auto-scaler example...")
    
    # Check if CUDA is available
    if not torch.cuda.is_available():
        print("⚠️  CUDA not available, using CPU")
        device_ids = [0]
    else:
        device_count = torch.cuda.device_count()
        device_ids = list(range(min(2, device_count)))  # Use up to 2 GPUs
        print(f"✅ CUDA available with {device_count} GPU(s)")
        
    # Create auto-scaler
    auto_scaler = AutoScaler(
        device_ids=device_ids,
        initial_batch_size=16,
        min_batch_size=1,
        max_batch_size=64,
        target_utilization=75.0,
        adaptation_rate=0.15,
        load_balancing_strategy="utilization_based",
        scaling_strategy="adaptive"
    )
    
    # Simulate training loop
    print("\n🔄 Simulating training with auto-scaling...")
    for step in range(20):
        # Get next device
        device_id = auto_scaler.get_next_device()
        
        # Get batch size for this device
        batch_size = auto_scaler.get_batch_size_for_device(device_id)
        
        # Simulate batch processing
        processing_time = 0.1 + (batch_size / 100.0) + (torch.rand(1).item() * 0.05)
        time.sleep(processing_time)  # Simulate processing
        
        # Record completion
        auto_scaler.record_batch_completion(device_id, processing_time, batch_size)
        
        # Adjust batch sizes periodically
        if step % 5 == 4:
            auto_scaler.adjust_batch_sizes()
            
        # Print progress
        if step % 5 == 0:
            summary = auto_scaler.get_scaling_summary()
            print(f"Step {step}: Device {device_id}, Batch size {batch_size}, "
                  f"Utilization {summary['global_utilization']:.1f}%")
    
    # Print final summary
    print("\n" + "="*60)
    print("📈 Auto-Scaler Final Summary:")
    summary = auto_scaler.get_scaling_summary()
    print(f"   Devices: {summary['device_count']}")
    print(f"   Global Utilization: {summary['global_utilization']:.1f}%")
    print(f"   Target Utilization: {summary['target_utilization']:.1f}%")
    print(f"   Average Throughput: {summary['avg_throughput']:.1f} samples/s")
    print(f"   Scaling Events: {summary['scaling_events']}")
    
    for device_id, batch_size in summary['device_batch_sizes'].items():
        device_metrics = summary['device_metrics'].get(device_id, {})
        util = device_metrics.get('current_utilization', 0)
        print(f"   Device {device_id}: Batch size {batch_size}, Utilization {util:.1f}%")
        
    print("\n✅ Auto-scaling example completed!")

if __name__ == "__main__":
    example_auto_scaling()