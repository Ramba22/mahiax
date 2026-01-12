"""
Energy Profiler for MAHIA Dashboard V3
Integrates with NVML to track energy consumption, cost, and efficiency metrics
"""

import time
import threading
from typing import Dict, Any, Optional, List, Tuple
from collections import defaultdict, deque
import json

# Conditional imports with fallbacks
NVML_AVAILABLE = False
pynvml = None

try:
    import pynvml
    NVML_AVAILABLE = True
except ImportError:
    pass

class EnergyProfiler:
    """Energy profiler with NVML integration for kWh tracking and cost analysis"""
    
    def __init__(self, 
                 sampling_interval: float = 1.0,
                 cost_per_kwh: float = 0.12,  # $0.12 per kWh as default
                 currency: str = "USD"):
        """
        Initialize the energy profiler
        
        Args:
            sampling_interval: Interval between energy measurements (seconds)
            cost_per_kwh: Cost per kilowatt-hour in specified currency
            currency: Currency code for cost calculations
        """
        self.sampling_interval = sampling_interval
        self.cost_per_kwh = cost_per_kwh
        self.currency = currency
        
        # Energy tracking
        self.is_monitoring = False
        self.monitoring_thread = None
        self.energy_data = defaultdict(deque)
        self.energy_limit = 10000  # Store up to 10,000 data points
        
        # Training session tracking
        self.session_start_time = None
        self.session_energy_joules = 0.0
        self.session_samples_processed = 0
        self.session_epochs_completed = 0
        
        # Lock for thread safety
        self.lock = threading.Lock()
        
        # Try to initialize NVML
        self.nvml_initialized = False
        self.device_count = 0
        self._initialize_nvml()
        
        print(f"🔋 EnergyProfiler initialized")
        print(f"   NVML Available: {self.nvml_initialized}")
        print(f"   Cost: {self.cost_per_kwh} {self.currency}/kWh")
        print(f"   Sampling Interval: {self.sampling_interval}s")
        
    def _initialize_nvml(self):
        """Initialize NVML for GPU energy monitoring"""
        if NVML_AVAILABLE and pynvml is not None:
            try:
                pynvml.nvmlInit()
                self.nvml_initialized = True
                self.device_count = pynvml.nvmlDeviceGetCount()
                print(f"✅ NVML initialized for {self.device_count} GPU(s)")
            except Exception as e:
                print(f"⚠️  NVML initialization failed: {e}")
        else:
            print("⚠️  NVML not available - using estimated power consumption")
            
    def start_monitoring(self, session_name: str = "training_session"):
        """
        Start energy monitoring
        
        Args:
            session_name: Name for this monitoring session
        """
        if self.is_monitoring:
            print("⚠️  Energy monitoring is already running")
            return
            
        self.session_start_time = time.time()
        self.session_energy_joules = 0.0
        self.session_samples_processed = 0
        self.session_epochs_completed = 0
        
        self.is_monitoring = True
        self.monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitoring_thread.start()
        print(f"📈 Started energy monitoring session: {session_name}")
        
    def stop_monitoring(self):
        """Stop energy monitoring"""
        self.is_monitoring = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=2.0)
        print("⏹️  Stopped energy monitoring")
        
    def _monitoring_loop(self):
        """Background monitoring loop"""
        last_timestamp = time.time()
        
        while self.is_monitoring:
            current_timestamp = time.time()
            
            # Collect energy metrics
            energy_metrics = self._collect_energy_metrics(current_timestamp)
            
            # Calculate energy consumption for this interval
            interval_seconds = current_timestamp - last_timestamp
            if interval_seconds > 0:
                interval_energy_joules = self._calculate_interval_energy(energy_metrics, interval_seconds)
                self.session_energy_joules += interval_energy_joules
                
                # Record metrics
                self._record_energy_metrics(energy_metrics, interval_energy_joules, current_timestamp)
                
            last_timestamp = current_timestamp
            time.sleep(self.sampling_interval)
            
    def _collect_energy_metrics(self, timestamp: float) -> Dict[str, Any]:
        """Collect energy metrics from all available sources"""
        metrics = {
            "timestamp": timestamp,
            "gpu_metrics": {},
            "estimated_power_watts": 0.0
        }
        
        # Collect GPU metrics if NVML is available
        if self.nvml_initialized:
            gpu_metrics = self._collect_gpu_energy_metrics()
            metrics["gpu_metrics"] = gpu_metrics
            
            # Sum power from all GPUs
            total_power = sum(gpu.get("power_watts", 0) for gpu in gpu_metrics.values())
            metrics["estimated_power_watts"] = total_power
        else:
            # Estimate power consumption based on typical values
            # Assuming 250W per GPU as a rough estimate
            estimated_power = self.device_count * 250.0
            metrics["estimated_power_watts"] = estimated_power
            
        return metrics
        
    def _collect_gpu_energy_metrics(self) -> Dict[str, Dict[str, float]]:
        """Collect detailed GPU energy metrics using NVML"""
        if not NVML_AVAILABLE or pynvml is None:
            return {}
            
        gpu_metrics = {}
        
        try:
            for i in range(self.device_count):
                handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                gpu_name = f"gpu_{i}"
                
                gpu_data = {}
                
                # Power usage (in watts)
                try:
                    power_mw = pynvml.nvmlDeviceGetPowerUsage(handle)
                    gpu_data["power_watts"] = power_mw / 1000.0
                except:
                    gpu_data["power_watts"] = 0.0
                    
                # Temperature
                try:
                    temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
                    gpu_data["temperature_c"] = temp
                except:
                    gpu_data["temperature_c"] = 0.0
                    
                # Memory usage
                try:
                    mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                    gpu_data["memory_used_gb"] = mem_info.used / (1024**3)
                    gpu_data["memory_total_gb"] = mem_info.total / (1024**3)
                except:
                    gpu_data["memory_used_gb"] = 0.0
                    gpu_data["memory_total_gb"] = 0.0
                    
                # Utilization
                try:
                    util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                    gpu_data["gpu_utilization"] = util.gpu
                    gpu_data["memory_utilization"] = util.memory
                except:
                    gpu_data["gpu_utilization"] = 0.0
                    gpu_data["memory_utilization"] = 0.0
                    
                gpu_metrics[gpu_name] = gpu_data
                
        except Exception as e:
            print(f"❌ Error collecting GPU energy metrics: {e}")
            
        return gpu_metrics
        
    def _calculate_interval_energy(self, metrics: Dict[str, Any], interval_seconds: float) -> float:
        """
        Calculate energy consumption for the interval
        
        Args:
            metrics: Energy metrics collected
            interval_seconds: Duration of the interval in seconds
            
        Returns:
            Energy consumption in joules
        """
        power_watts = metrics["estimated_power_watts"]
        energy_joules = power_watts * interval_seconds
        return energy_joules
        
    def _record_energy_metrics(self, metrics: Dict[str, Any], interval_energy_joules: float, timestamp: float):
        """
        Record energy metrics
        
        Args:
            metrics: Energy metrics collected
            interval_energy_joules: Energy consumed in this interval
            timestamp: Timestamp of the measurement
        """
        with self.lock:
            # Add to energy data
            energy_record = {
                "timestamp": timestamp,
                "interval_energy_joules": interval_energy_joules,
                "cumulative_energy_joules": self.session_energy_joules,
                "power_watts": metrics["estimated_power_watts"],
                "gpu_metrics": metrics["gpu_metrics"]
            }
            
            self.energy_data["energy_profile"].append(energy_record)
            
            # Trim if exceeding limit
            if len(self.energy_data["energy_profile"]) > self.energy_limit:
                while len(self.energy_data["energy_profile"]) > self.energy_limit:
                    self.energy_data["energy_profile"].popleft()
                    
    def record_training_progress(self, samples_processed: int = 0, epochs_completed: int = 0):
        """
        Record training progress for efficiency calculations
        
        Args:
            samples_processed: Number of samples processed in this update
            epochs_completed: Number of epochs completed in this update
        """
        with self.lock:
            self.session_samples_processed += samples_processed
            self.session_epochs_completed += epochs_completed
            
    def get_energy_stats(self) -> Dict[str, Any]:
        """
        Get current energy statistics
        
        Returns:
            Dictionary with energy statistics
        """
        with self.lock:
            # Calculate time elapsed
            elapsed_seconds = 0.0
            if self.session_start_time:
                elapsed_seconds = time.time() - self.session_start_time
                
            # Convert to kWh
            energy_kwh = self.session_energy_joules / (3600 * 1000)  # Joules to kWh
            
            # Calculate cost
            cost = energy_kwh * self.cost_per_kwh
            
            # Calculate efficiency metrics
            samples_per_kwh = 0.0
            epochs_per_kwh = 0.0
            if energy_kwh > 0:
                samples_per_kwh = self.session_samples_processed / energy_kwh if self.session_samples_processed > 0 else 0.0
                epochs_per_kwh = self.session_epochs_completed / energy_kwh if self.session_epochs_completed > 0 else 0.0
                
            # Calculate power metrics
            average_power_watts = 0.0
            if elapsed_seconds > 0:
                average_power_watts = self.session_energy_joules / elapsed_seconds
                
            return {
                "session_elapsed_seconds": elapsed_seconds,
                "session_energy_joules": self.session_energy_joules,
                "session_energy_kwh": energy_kwh,
                "session_cost": cost,
                "currency": self.currency,
                "cost_per_kwh": self.cost_per_kwh,
                "samples_processed": self.session_samples_processed,
                "epochs_completed": self.session_epochs_completed,
                "samples_per_kwh": samples_per_kwh,
                "epochs_per_kwh": epochs_per_kwh,
                "average_power_watts": average_power_watts,
                "device_count": self.device_count,
                "nvml_available": self.nvml_initialized
            }
            
    def get_efficiency_score(self) -> float:
        """
        Calculate an efficiency score based on energy consumption and training progress
        
        Returns:
            Efficiency score (higher is better)
        """
        stats = self.get_energy_stats()
        
        # Simple efficiency score: samples per kWh normalized to 0-100 range
        # Assuming 100,000 samples per kWh is excellent efficiency
        base_efficiency = stats["samples_per_kwh"] / 100000.0 if stats["samples_per_kwh"] > 0 else 0.0
        efficiency_score = min(100.0, base_efficiency * 100.0)
        
        return efficiency_score
        
    def get_energy_profile(self) -> List[Dict[str, Any]]:
        """
        Get detailed energy profile data
        
        Returns:
            List of energy measurements
        """
        with self.lock:
            return list(self.energy_data["energy_profile"])
            
    def generate_report(self) -> str:
        """
        Generate a comprehensive energy report
        
        Returns:
            Formatted report string
        """
        stats = self.get_energy_stats()
        
        report = f"""
🔋 MAHIA Energy Profiler Report
============================

⏱️  Session Statistics:
   Elapsed Time: {stats['session_elapsed_seconds']:.2f} seconds
   Samples Processed: {stats['samples_processed']:,}
   Epochs Completed: {stats['epochs_completed']}

⚡ Energy Consumption:
   Total Energy: {stats['session_energy_joules']:.2f} Joules
   Energy (kWh): {stats['session_energy_kwh']:.6f} kWh
   Average Power: {stats['average_power_watts']:.2f} Watts

💰 Cost Analysis:
   Cost per kWh: {stats['cost_per_kwh']} {stats['currency']}
   Total Cost: {stats['session_cost']:.4f} {stats['currency']}

📈 Efficiency Metrics:
   Samples per kWh: {stats['samples_per_kwh']:,.2f}
   Epochs per kWh: {stats['epochs_per_kwh']:.6f}
   Efficiency Score: {self.get_efficiency_score():.2f}/100

🖥️  System Information:
   GPUs Detected: {stats['device_count']}
   NVML Available: {stats['nvml_available']}
"""
        
        return report
        
    def export_to_json(self, filepath: str):
        """
        Export energy data to JSON file
        
        Args:
            filepath: Path to save the JSON file
        """
        data = {
            "energy_stats": self.get_energy_stats(),
            "energy_profile": self.get_energy_profile(),
            "export_timestamp": time.time()
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
            
        print(f"✅ Energy data exported to {filepath}")
        
    def clear_session(self):
        """Clear current session data"""
        with self.lock:
            self.session_energy_joules = 0.0
            self.session_samples_processed = 0
            self.session_epochs_completed = 0
            self.session_start_time = None
            self.energy_data.clear()
        print("🗑️  Energy session data cleared")

# Global instance
_energy_profiler = None

def get_energy_profiler() -> EnergyProfiler:
    """Get the global energy profiler instance"""
    global _energy_profiler
    if _energy_profiler is None:
        _energy_profiler = EnergyProfiler()
    return _energy_profiler

# Example usage
def example_energy_profiling():
    """Example of energy profiling usage"""
    print("🔧 Setting up energy profiling example...")
    
    # Create energy profiler
    profiler = get_energy_profiler()
    
    # Start monitoring
    profiler.start_monitoring("demo_session")
    
    # Simulate training progress
    for epoch in range(5):
        # Simulate processing samples
        for batch in range(10):
            # Record progress
            profiler.record_training_progress(samples_processed=32)
            
            # Simulate work
            time.sleep(0.1)
            
        # Record epoch completion
        profiler.record_training_progress(epochs_completed=1)
        print(f"Epoch {epoch + 1} completed")
        
    # Stop monitoring
    profiler.stop_monitoring()
    
    # Print report
    print(profiler.generate_report())
    
    # Export to JSON
    profiler.export_to_json("energy_report.json")
    
    print("✅ Energy profiling example completed!")

if __name__ == "__main__":
    example_energy_profiling()