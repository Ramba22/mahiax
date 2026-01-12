"""
Energy Per Sample Metric for MAHIA-X
This module integrates energy-per-sample metrics into telemetry for performance monitoring.
"""

import time
from typing import Dict, Any, Optional, List
from collections import defaultdict

# Conditional imports with fallbacks
TORCH_AVAILABLE = False
torch = None

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    pass

class EnergyPerSampleMetric:
    """Energy Per Sample Metric tracker for performance monitoring"""
    
    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self.energy_samples = []
        self.processing_times = []
        self.sample_counts = []
        self.lock = threading.Lock() if 'threading' in globals() else None
        
        print(f"💡 EnergyPerSampleMetric initialized with window size: {window_size}")
        
    def record_energy_sample(self, energy_joules: float, processing_time_seconds: float, 
                            sample_count: int = 1) -> Dict[str, Any]:
        """
        Record energy consumption for a batch of samples.
        
        Args:
            energy_joules: Energy consumed in joules
            processing_time_seconds: Time taken to process the samples
            sample_count: Number of samples processed
            
        Returns:
            Dict containing energy metrics
        """
        timestamp = time.time()
        
        # Calculate metrics
        energy_per_sample = energy_joules / sample_count if sample_count > 0 else 0.0
        samples_per_second = sample_count / processing_time_seconds if processing_time_seconds > 0 else 0.0
        energy_per_second = energy_joules / processing_time_seconds if processing_time_seconds > 0 else 0.0
        
        energy_record = {
            "timestamp": timestamp,
            "energy_joules": energy_joules,
            "processing_time_seconds": processing_time_seconds,
            "sample_count": sample_count,
            "energy_per_sample_joules": energy_per_sample,
            "samples_per_second": samples_per_second,
            "energy_per_second_watts": energy_per_second
        }
        
        # Store record
        if self.lock:
            with self.lock:
                self.energy_samples.append(energy_record)
                self.processing_times.append(processing_time_seconds)
                self.sample_counts.append(sample_count)
                
                # Keep only last window_size entries to prevent memory bloat
                if len(self.energy_samples) > self.window_size:
                    self.energy_samples = self.energy_samples[-self.window_size:]
                    self.processing_times = self.processing_times[-self.window_size:]
                    self.sample_counts = self.sample_counts[-self.window_size:]
        else:
            self.energy_samples.append(energy_record)
            self.processing_times.append(processing_time_seconds)
            self.sample_counts.append(sample_count)
            
            # Keep only last window_size entries to prevent memory bloat
            if len(self.energy_samples) > self.window_size:
                self.energy_samples = self.energy_samples[-self.window_size:]
                self.processing_times = self.processing_times[-self.window_size:]
                self.sample_counts = self.sample_counts[-self.window_size:]
        
        # Print summary
        print(f"💡 Energy Metrics - {sample_count} samples:")
        print(f"   Energy: {energy_joules:.4f}J ({energy_per_sample:.6f}J/sample)")
        print(f"   Throughput: {samples_per_second:.2f} samples/sec")
        print(f"   Power: {energy_per_second:.4f}W")
        
        return energy_record
    
    def get_energy_samples(self) -> List[Dict[str, Any]]:
        """Get all energy sample records"""
        if self.lock:
            with self.lock:
                return self.energy_samples.copy()
        else:
            return self.energy_samples.copy()
    
    def clear_energy_samples(self):
        """Clear all energy sample records"""
        if self.lock:
            with self.lock:
                self.energy_samples.clear()
                self.processing_times.clear()
                self.sample_counts.clear()
        else:
            self.energy_samples.clear()
            self.processing_times.clear()
            self.sample_counts.clear()
        print("🗑️  Energy sample records cleared")
    
    def get_average_metrics(self) -> Dict[str, float]:
        """Calculate average energy metrics over the recorded window"""
        samples = self.get_energy_samples()
        if not samples:
            return {
                "avg_energy_per_sample_joules": 0.0,
                "avg_samples_per_second": 0.0,
                "avg_energy_per_second_watts": 0.0,
                "total_samples": 0,
                "total_energy_joules": 0.0
            }
        
        total_samples = sum(record["sample_count"] for record in samples)
        total_energy = sum(record["energy_joules"] for record in samples)
        total_time = sum(record["processing_time_seconds"] for record in samples)
        
        avg_energy_per_sample = total_energy / total_samples if total_samples > 0 else 0.0
        avg_samples_per_second = total_samples / total_time if total_time > 0 else 0.0
        avg_energy_per_second = total_energy / total_time if total_time > 0 else 0.0
        
        return {
            "avg_energy_per_sample_joules": avg_energy_per_sample,
            "avg_samples_per_second": avg_samples_per_second,
            "avg_energy_per_second_watts": avg_energy_per_second,
            "total_samples": total_samples,
            "total_energy_joules": total_energy
        }
    
    def estimate_training_energy(self, total_samples: int, batch_size: int) -> Dict[str, float]:
        """
        Estimate energy consumption for training a given number of samples.
        
        Args:
            total_samples: Total number of samples to train on
            batch_size: Batch size used for training
            
        Returns:
            Dict containing energy estimates
        """
        avg_metrics = self.get_average_metrics()
        
        if avg_metrics["avg_energy_per_sample_joules"] <= 0:
            return {
                "estimated_energy_joules": 0.0,
                "estimated_time_seconds": 0.0,
                "estimated_power_watts": 0.0,
                "batches": 0
            }
        
        batches = total_samples // batch_size
        estimated_energy = avg_metrics["avg_energy_per_sample_joules"] * total_samples
        estimated_time = total_samples / avg_metrics["avg_samples_per_second"] if avg_metrics["avg_samples_per_second"] > 0 else 0.0
        estimated_power = avg_metrics["avg_energy_per_second_watts"]
        
        return {
            "estimated_energy_joules": estimated_energy,
            "estimated_time_seconds": estimated_time,
            "estimated_power_watts": estimated_power,
            "batches": batches
        }
    
    def generate_report(self) -> str:
        """Generate a summary report of energy metrics"""
        samples = self.get_energy_samples()
        if not samples:
            return "No energy samples recorded"
        
        avg_metrics = self.get_average_metrics()
        
        report = f"""
💡 Energy Per Sample Metrics Report
==============================
Total Samples Recorded: {len(samples)}
Total Processed Samples: {avg_metrics['total_samples']}
Total Energy Consumed: {avg_metrics['total_energy_joules']:.4f}J

Average Metrics:
  Energy per Sample: {avg_metrics['avg_energy_per_sample_joules']:.6f}J
  Throughput: {avg_metrics['avg_samples_per_second']:.2f} samples/sec
  Power Consumption: {avg_metrics['avg_energy_per_second_watts']:.4f}W

Recent Samples:
"""
        
        # Show last 5 entries
        recent_samples = samples[-5:] if len(samples) >= 5 else samples
        for i, sample in enumerate(recent_samples):
            report += f"  {i+1}. {time.ctime(sample['timestamp'])}\n"
            report += f"     Samples: {sample['sample_count']} | "
            report += f"Energy: {sample['energy_joules']:.4f}J | "
            report += f"Time: {sample['processing_time_seconds']:.3f}s\n"
            report += f"     Per Sample: {sample['energy_per_sample_joules']:.6f}J | "
            report += f"Throughput: {sample['samples_per_second']:.2f}/s | "
            report += f"Power: {sample['energy_per_second_watts']:.4f}W\n"
        
        return report

# Global instance
_energy_metric = None

def get_energy_metric() -> EnergyPerSampleMetric:
    """Get the global energy per sample metric instance"""
    global _energy_metric
    if _energy_metric is None:
        _energy_metric = EnergyPerSampleMetric()
    return _energy_metric

# Import threading at the end to avoid circular import issues
import threading

if __name__ == "__main__":
    # Example usage
    energy_tracker = get_energy_metric()
    
    # Simulate some energy recordings
    energy_tracker.record_energy_sample(energy_joules=15.2, processing_time_seconds=2.5, sample_count=32)
    energy_tracker.record_energy_sample(energy_joules=14.8, processing_time_seconds=2.4, sample_count=32)
    energy_tracker.record_energy_sample(energy_joules=15.5, processing_time_seconds=2.6, sample_count=32)
    
    # Print report
    print(energy_tracker.generate_report())
    
    # Estimate energy for larger training
    estimate = energy_tracker.estimate_training_energy(total_samples=10000, batch_size=32)
    print(f"\n🔮 Energy Estimate for 10,000 samples:")
    print(f"   Estimated Energy: {estimate['estimated_energy_joules']:.2f}J")
    print(f"   Estimated Time: {estimate['estimated_time_seconds']:.2f}s")
    print(f"   Estimated Power: {estimate['estimated_power_watts']:.2f}W")
    print(f"   Batches: {estimate['batches']}")