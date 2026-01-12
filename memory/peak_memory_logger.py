"""
Peak Memory Logger for MAHIA-X
This module provides peak memory tracking and auto-reporting when thresholds are exceeded.
"""

import time
import threading
from typing import Dict, Any, Optional, List, Callable
from collections import defaultdict

# Conditional imports with fallbacks
TORCH_AVAILABLE = False
torch = None

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    pass

class PeakMemoryLogger:
    """Peak Memory Logger with auto-reporting capabilities"""
    
    def __init__(self, vram_threshold_gb: float = 16.0, ram_threshold_gb: float = 32.0,
                 activation_threshold_gb: float = 8.0, alert_callback: Optional[Callable] = None):
        self.vram_threshold_gb = vram_threshold_gb
        self.ram_threshold_gb = ram_threshold_gb
        self.activation_threshold_gb = activation_threshold_gb
        self.alert_callback = alert_callback
        self.peak_logs = []
        self.is_monitoring = False
        self.monitoring_thread = None
        self.lock = threading.Lock()
        
        print(f"📉 PeakMemoryLogger initialized with thresholds:")
        print(f"   VRAM: {vram_threshold_gb}GB")
        print(f"   RAM: {ram_threshold_gb}GB")
        print(f"   Activation: {activation_threshold_gb}GB")
        
    def check_and_log_peak_memory(self, vram_used: float = 0.0, ram_used: float = 0.0, 
                                 activation_memory: float = 0.0) -> Dict[str, Any]:
        """Check memory usage against thresholds and log peaks"""
        timestamp = time.time()
        peak_info = {
            "timestamp": timestamp,
            "vram_used_gb": vram_used,
            "ram_used_gb": ram_used,
            "activation_memory_gb": activation_memory,
            "alerts": []
        }
        
        # Check thresholds
        if vram_used >= self.vram_threshold_gb:
            alert = f"⚠️  VRAM threshold exceeded: {vram_used:.2f}GB >= {self.vram_threshold_gb}GB"
            peak_info["alerts"].append(alert)
            print(alert)
            
        if ram_used >= self.ram_threshold_gb:
            alert = f"⚠️  RAM threshold exceeded: {ram_used:.2f}GB >= {self.ram_threshold_gb}GB"
            peak_info["alerts"].append(alert)
            print(alert)
            
        if activation_memory >= self.activation_threshold_gb:
            alert = f"⚠️  Activation memory threshold exceeded: {activation_memory:.2f}GB >= {self.activation_threshold_gb}GB"
            peak_info["alerts"].append(alert)
            print(alert)
        
        # Store peak info
        with self.lock:
            self.peak_logs.append(peak_info)
            
            # Keep only last 1000 entries to prevent memory bloat
            if len(self.peak_logs) > 1000:
                self.peak_logs = self.peak_logs[-1000:]
        
        # Call alert callback if provided and there are alerts
        if self.alert_callback and peak_info["alerts"]:
            try:
                self.alert_callback(peak_info)
            except Exception as e:
                print(f"❌ Error in alert callback: {e}")
        
        return peak_info
    
    def get_peak_logs(self) -> List[Dict[str, Any]]:
        """Get all peak memory logs"""
        with self.lock:
            return self.peak_logs.copy()
    
    def clear_peak_logs(self):
        """Clear all peak memory logs"""
        with self.lock:
            self.peak_logs.clear()
        print("🗑️  Peak memory logs cleared")
    
    def get_torch_memory_info(self) -> tuple:
        """Get PyTorch memory information (VRAM used, total VRAM) in GB"""
        if not TORCH_AVAILABLE or torch is None or not torch.cuda.is_available():
            return 0.0, 0.0
            
        try:
            device = torch.cuda.current_device()
            vram_used = torch.cuda.memory_allocated(device) / (1024**3)  # Convert to GB
            vram_total = torch.cuda.get_device_properties(device).total_memory / (1024**3)
            return vram_used, vram_total
        except Exception:
            return 0.0, 0.0
    
    def auto_report_on_threshold_exceeded(self):
        """Auto-report when memory thresholds are exceeded"""
        vram_used, _ = self.get_torch_memory_info()
        
        # For system RAM, we would need psutil, but we'll skip that for now
        ram_used = 0.0
        activation_memory = 0.0  # This would be provided by the model
        
        return self.check_and_log_peak_memory(vram_used, ram_used, activation_memory)
    
    def start_auto_monitoring(self, interval_seconds: int = 5):
        """Start automatic monitoring in a background thread"""
        if self.is_monitoring:
            print("⚠️  Auto monitoring is already running")
            return
            
        self.is_monitoring = True
        self.monitoring_thread = threading.Thread(target=self._monitoring_loop, 
                                                 args=(interval_seconds,), daemon=True)
        self.monitoring_thread.start()
        print(f"📈 Started auto monitoring every {interval_seconds} seconds")
    
    def stop_auto_monitoring(self):
        """Stop automatic monitoring"""
        self.is_monitoring = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=2.0)
        print("⏹️  Stopped auto monitoring")
    
    def _monitoring_loop(self, interval_seconds: int):
        """Background monitoring loop"""
        while self.is_monitoring:
            self.auto_report_on_threshold_exceeded()
            time.sleep(interval_seconds)
    
    def generate_report(self) -> str:
        """Generate a summary report of peak memory usage"""
        logs = self.get_peak_logs()
        if not logs:
            return "No peak memory logs available"
        
        total_alerts = sum(len(log["alerts"]) for log in logs)
        vram_exceeded = sum(1 for log in logs if any("VRAM" in alert for alert in log["alerts"]))
        ram_exceeded = sum(1 for log in logs if any("RAM" in alert for alert in log["alerts"]))
        activation_exceeded = sum(1 for log in logs if any("Activation" in alert for alert in log["alerts"]))
        
        report = f"""
📉 Peak Memory Usage Report
========================
Total Logs: {len(logs)}
Total Alerts: {total_alerts}
VRAM Threshold Exceeded: {vram_exceeded} times
RAM Threshold Exceeded: {ram_exceeded} times
Activation Memory Threshold Exceeded: {activation_exceeded} times

Recent Peak Usage:
"""
        
        # Show last 5 entries
        recent_logs = logs[-5:] if len(logs) >= 5 else logs
        for i, log in enumerate(recent_logs):
            report += f"  {i+1}. {time.ctime(log['timestamp'])}\n"
            report += f"     VRAM: {log['vram_used_gb']:.2f}GB | "
            report += f"RAM: {log['ram_used_gb']:.2f}GB | "
            report += f"Activation: {log['activation_memory_gb']:.2f}GB\n"
            if log["alerts"]:
                report += f"     Alerts: {len(log['alerts'])}\n"
        
        return report

# Global instance
_peak_logger = None

def get_peak_logger() -> PeakMemoryLogger:
    """Get the global peak memory logger instance"""
    global _peak_logger
    if _peak_logger is None:
        _peak_logger = PeakMemoryLogger()
    return _peak_logger

if __name__ == "__main__":
    # Example usage
    def alert_handler(peak_info):
        print(f"🚨 ALERT: {peak_info['alerts']}")
    
    logger = get_peak_logger()
    logger.alert_callback = alert_handler
    
    # Simulate some memory usage
    logger.check_and_log_peak_memory(vram_used=18.5, ram_used=35.2, activation_memory=9.1)
    
    # Print report
    print(logger.generate_report())