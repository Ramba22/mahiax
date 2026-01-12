"""
Telemetry Layer for MAHIA OptiCore
Integration with NVML, Torch CUDA Stats and API-compatible metrics.
"""

import time
import threading
from typing import Dict, Any, Optional, List, Union, Tuple
from collections import defaultdict, deque

# Conditional imports with fallbacks
NVML_AVAILABLE = False
pynvml = None

TORCH_AVAILABLE = False
torch = None

try:
    import pynvml
    NVML_AVAILABLE = True
except ImportError:
    pass

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    pass

class TelemetryLayer:
    """Telemetry layer with standardized metrics"""
    
    def __init__(self, sampling_interval: float = 1.0):
        self.sampling_interval = sampling_interval
        self.is_monitoring = False
        self.monitoring_thread = None
        self.metrics = defaultdict(deque)
        self.metric_limits = defaultdict(lambda: 1000)  # Default limit per metric
        self.lock = threading.Lock()
        
        # Try to initialize NVML
        self.nvml_initialized = False
        if NVML_AVAILABLE and pynvml is not None:
            try:
                pynvml.nvmlInit()
                self.nvml_initialized = True
                self.device_count = pynvml.nvmlDeviceGetCount()
                print(f"📊 TelemetryLayer initialized with NVML support for {self.device_count} GPU(s)")
            except Exception as e:
                print(f"⚠️  NVML initialization failed: {e}")
        else:
            print("📊 TelemetryLayer initialized without NVML support")
            
    def start_monitoring(self):
        """Start real-time monitoring"""
        if self.is_monitoring:
            print("⚠️  Telemetry monitoring is already running")
            return
            
        self.is_monitoring = True
        self.monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitoring_thread.start()
        print("📈 Started telemetry monitoring")
        
    def stop_monitoring(self):
        """Stop real-time monitoring"""
        self.is_monitoring = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=2.0)
        print("⏹️  Stopped telemetry monitoring")
        
    def _monitoring_loop(self):
        """Background monitoring loop"""
        while self.is_monitoring:
            self._collect_metrics()
            time.sleep(self.sampling_interval)
            
    def _collect_metrics(self):
        """Collect system metrics"""
        timestamp = time.time()
        
        # Collect GPU metrics if NVML is available
        if self.nvml_initialized:
            self._collect_gpu_metrics(timestamp)
            
        # Collect PyTorch CUDA metrics if available
        if TORCH_AVAILABLE and torch is not None and torch.cuda.is_available():
            self._collect_torch_metrics(timestamp)
            
    def _collect_gpu_metrics(self, timestamp: float):
        """Collect GPU metrics using NVML"""
        if not NVML_AVAILABLE or pynvml is None:
            return
            
        try:
            for i in range(self.device_count):
                handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                
                # Temperature
                try:
                    temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
                    self.record_metric(f"gpu_{i}_temperature", temp, timestamp)
                except:
                    pass
                    
                # Power usage
                try:
                    power = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0  # Convert to watts
                    self.record_metric(f"gpu_{i}_power", power, timestamp)
                except:
                    pass
                    
                # Memory usage
                try:
                    mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                    self.record_metric(f"gpu_{i}_memory_used", mem_info.used / (1024**3), timestamp)  # GB
                    self.record_metric(f"gpu_{i}_memory_total", mem_info.total / (1024**3), timestamp)  # GB
                except:
                    pass
                    
                # Utilization
                try:
                    util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                    self.record_metric(f"gpu_{i}_utilization", util.gpu, timestamp)
                except:
                    pass
                    
        except Exception as e:
            print(f"❌ Error collecting GPU metrics: {e}")
            
    def _collect_torch_metrics(self, timestamp: float):
        """Collect PyTorch CUDA metrics"""
        if not TORCH_AVAILABLE or torch is None:
            return
            
        try:
            if torch.cuda.is_available():
                device = torch.cuda.current_device()
                
                # Memory allocation
                allocated = torch.cuda.memory_allocated(device)
                reserved = torch.cuda.memory_reserved(device)
                
                self.record_metric("torch_memory_allocated", allocated / (1024**3), timestamp)  # GB
                self.record_metric("torch_memory_reserved", reserved / (1024**3), timestamp)  # GB
                
                # Cache information
                try:
                    cache_info = torch.cuda.memory_stats(device)
                    self.record_metric("torch_cache_allocated", 
                                     cache_info.get("allocated_bytes.all.current", 0) / (1024**3), timestamp)
                except:
                    pass
                
        except Exception as e:
            print(f"❌ Error collecting PyTorch metrics: {e}")
    
    def record_metric(self, metric_name: str, value: Union[float, int], 
                     timestamp: Optional[float] = None):
        """
        Record a metric value.
        
        Args:
            metric_name: Name of the metric
            value: Metric value
            timestamp: Optional timestamp (defaults to current time)
        """
        if timestamp is None:
            timestamp = time.time()
            
        with self.lock:
            # Add to metrics deque
            self.metrics[metric_name].append((timestamp, value))
            
            # Trim if exceeding limit
            limit = self.metric_limits[metric_name]
            if len(self.metrics[metric_name]) > limit:
                # Remove oldest entries
                while len(self.metrics[metric_name]) > limit:
                    self.metrics[metric_name].popleft()
    
    def get_metric(self, metric_name: str, 
                   since_timestamp: Optional[float] = None) -> List[Tuple[float, Union[float, int]]]:
        """
        Get metric values.
        
        Args:
            metric_name: Name of the metric
            since_timestamp: Optional timestamp to filter results
            
        Returns:
            List of (timestamp, value) tuples
        """
        with self.lock:
            if metric_name not in self.metrics:
                return []
                
            values = list(self.metrics[metric_name])
            
            if since_timestamp is not None:
                values = [(t, v) for t, v in values if t >= since_timestamp]
                
            return values
    
    def get_latest_metric(self, metric_name: str) -> Optional[Tuple[float, Union[float, int]]]:
        """
        Get the latest value of a metric.
        
        Args:
            metric_name: Name of the metric
            
        Returns:
            Latest (timestamp, value) tuple or None if not found
        """
        with self.lock:
            if metric_name not in self.metrics or not self.metrics[metric_name]:
                return None
            return self.metrics[metric_name][-1]
    
    def get_metric_names(self) -> List[str]:
        """Get all available metric names"""
        with self.lock:
            return list(self.metrics.keys())
    
    def set_metric_limit(self, metric_name: str, limit: int):
        """
        Set the limit for a metric (number of stored values).
        
        Args:
            metric_name: Name of the metric
            limit: Maximum number of values to store
        """
        with self.lock:
            self.metric_limits[metric_name] = limit
    
    def get_stats(self) -> Dict[str, Any]:
        """Get telemetry statistics"""
        with self.lock:
            stats = {}
            for metric_name, values in self.metrics.items():
                stats[metric_name] = {
                    "count": len(values),
                    "latest": values[-1] if values else None
                }
            return stats
    
    def clear_metrics(self):
        """Clear all metrics"""
        with self.lock:
            self.metrics.clear()
        print("🗑️  All telemetry metrics cleared")
    
    def record_event(self, event_type: str, data: Optional[Dict[str, Any]] = None):
        """
        Record a telemetry event.
        
        Args:
            event_type: Type of event
            data: Event data
        """
        event_data = {
            "type": event_type,
            "timestamp": time.time(),
            "data": data or {}
        }
        self.record_metric(f"event_{event_type}", 1, event_data["timestamp"])
        print(f"🔔 Telemetry event: {event_type}")

# Global instance
_telemetry_layer = None

def get_telemetry_layer() -> TelemetryLayer:
    """Get the global telemetry layer instance"""
    global _telemetry_layer
    if _telemetry_layer is None:
        _telemetry_layer = TelemetryLayer()
    return _telemetry_layer

if __name__ == "__main__":
    # Example usage
    telemetry = get_telemetry_layer()
    
    # Record some metrics
    telemetry.record_metric("test_metric", 42.5)
    telemetry.record_metric("another_metric", 100)
    
    # Get metrics
    test_values = telemetry.get_metric("test_metric")
    print(f"Test metric values: {test_values}")
    
    # Get latest metric
    latest = telemetry.get_latest_metric("test_metric")
    print(f"Latest test metric: {latest}")
    
    # Record event
    telemetry.record_event("training_step", {"batch": 10, "loss": 0.5})