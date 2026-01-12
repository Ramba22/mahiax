"""
Diagnostics for MAHIA OptiCore
Unified logging interface for metrics with export to JSON, CSV, Prometheus.
"""

import time
import json
import threading
from typing import Dict, Any, Optional, List, Union
from collections import defaultdict, deque
import csv

class Diagnostics:
    """Unified logging interface for OptiCore metrics and diagnostics"""
    
    def __init__(self, log_file: str = "opticore_diagnostics.log"):
        self.log_file = log_file
        self.log_buffer = deque(maxlen=1000)
        self.metrics = defaultdict(deque)
        self.metric_limits = defaultdict(lambda: 1000)
        self.lock = threading.Lock()
        
        print(f"📋 Diagnostics initialized with log file: {log_file}")
        
    def log_message(self, level: str, message: str, component: str = "unknown", 
                   data: Optional[Dict[str, Any]] = None):
        """
        Log a diagnostic message.
        
        Args:
            level: Log level (INFO, WARNING, ERROR, DEBUG)
            message: Log message
            component: Component name
            data: Additional data
        """
        timestamp = time.time()
        log_entry = {
            "timestamp": timestamp,
            "level": level,
            "component": component,
            "message": message,
            "data": data or {}
        }
        
        # Add to buffer
        with self.lock:
            self.log_buffer.append(log_entry)
            
        # Print to console
        time_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(timestamp))
        print(f"[{time_str}] {level:8} [{component:12}] {message}")
        
        # Write to file
        try:
            with open(self.log_file, "a", encoding="utf-8") as f:
                f.write(f"[{time_str}] {level:8} [{component:12}] {message}")
                if data:
                    f.write(f" {json.dumps(data)}")
                f.write("\n")
        except Exception as e:
            print(f"❌ Error writing to log file: {e}")
            
    def log_metric(self, metric_name: str, value: Union[float, int], 
                  component: str = "unknown", metadata: Optional[Dict[str, Any]] = None):
        """
        Log a metric value.
        
        Args:
            metric_name: Name of the metric
            value: Metric value
            component: Component name
            metadata: Additional metadata
        """
        timestamp = time.time()
        metric_entry = {
            "timestamp": timestamp,
            "metric_name": metric_name,
            "value": value,
            "component": component,
            "metadata": metadata or {}
        }
        
        # Store metric
        with self.lock:
            self.metrics[metric_name].append(metric_entry)
            
            # Trim if exceeding limit
            limit = self.metric_limits[metric_name]
            if len(self.metrics[metric_name]) > limit:
                # Remove oldest entries
                while len(self.metrics[metric_name]) > limit:
                    self.metrics[metric_name].popleft()
        
        # Also log as a message
        self.log_message("METRIC", f"{metric_name}: {value}", component, metadata)
        
    def export_to_json(self, filepath: str) -> bool:
        """
        Export diagnostics data to JSON file.
        
        Args:
            filepath: Path to export file
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            export_data = {
                "export_timestamp": time.time(),
                "logs": list(self.log_buffer),
                "metrics": {name: list(values) for name, values in self.metrics.items()}
            }
            
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(export_data, f, indent=2, default=str)
                
            print(f"💾 Diagnostics exported to JSON: {filepath}")
            return True
        except Exception as e:
            print(f"❌ Error exporting to JSON: {e}")
            return False
            
    def export_to_csv(self, filepath: str, metric_name: Optional[str] = None) -> bool:
        """
        Export metrics to CSV file.
        
        Args:
            filepath: Path to export file
            metric_name: Specific metric to export (None for all)
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            if metric_name:
                # Export specific metric
                if metric_name in self.metrics:
                    metrics_to_export = {metric_name: self.metrics[metric_name]}
                else:
                    print(f"⚠️  Metric not found: {metric_name}")
                    return False
            else:
                # Export all metrics
                metrics_to_export = self.metrics
                
            # Write CSV
            with open(filepath, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["timestamp", "metric_name", "value", "component"])
                
                for name, values in metrics_to_export.items():
                    for entry in values:
                        writer.writerow([
                            entry["timestamp"],
                            name,
                            entry["value"],
                            entry["component"]
                        ])
                        
            print(f"📊 Metrics exported to CSV: {filepath}")
            return True
        except Exception as e:
            print(f"❌ Error exporting to CSV: {e}")
            return False
            
    def export_to_prometheus(self, filepath: str) -> bool:
        """
        Export metrics in Prometheus format.
        
        Args:
            filepath: Path to export file
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            with open(filepath, "w", encoding="utf-8") as f:
                f.write("# HELP opticore_metrics OptiCore metrics\n")
                f.write("# TYPE opticore_metrics gauge\n")
                
                # Export latest values for each metric
                with self.lock:
                    for metric_name, values in self.metrics.items():
                        if values:
                            latest = values[-1]
                            # Format metric name for Prometheus (alphanumeric + underscore)
                            prom_metric_name = "".join(c if c.isalnum() or c == "_" else "_" 
                                                     for c in metric_name)
                            f.write(f"{prom_metric_name} {latest['value']} "
                                   f"{int(latest['timestamp'] * 1000)}\n")
                            
            print(f"📈 Metrics exported to Prometheus format: {filepath}")
            return True
        except Exception as e:
            print(f"❌ Error exporting to Prometheus format: {e}")
            return False
            
    def get_recent_logs(self, count: int = 50) -> List[Dict[str, Any]]:
        """
        Get recent log entries.
        
        Args:
            count: Number of recent entries to retrieve
            
        Returns:
            List of log entries
        """
        with self.lock:
            return list(self.log_buffer)[-count:]
            
    def get_metric_history(self, metric_name: str, 
                          since_timestamp: Optional[float] = None) -> List[Dict[str, Any]]:
        """
        Get history of a specific metric.
        
        Args:
            metric_name: Name of the metric
            since_timestamp: Optional timestamp to filter results
            
        Returns:
            List of metric entries
        """
        with self.lock:
            if metric_name not in self.metrics:
                return []
                
            values = list(self.metrics[metric_name])
            
            if since_timestamp is not None:
                values = [v for v in values if v["timestamp"] >= since_timestamp]
                
            return values
            
    def clear_logs(self):
        """Clear log buffer"""
        with self.lock:
            self.log_buffer.clear()
        print("🗑️  Log buffer cleared")
        
    def clear_metrics(self):
        """Clear all metrics"""
        with self.lock:
            self.metrics.clear()
        print("🧨 All metrics cleared")
        
    def set_metric_limit(self, metric_name: str, limit: int):
        """
        Set the limit for a metric (number of stored values).
        
        Args:
            metric_name: Name of the metric
            limit: Maximum number of values to store
        """
        with self.lock:
            self.metric_limits[metric_name] = limit

# Global instance
_diagnostics = None

def get_diagnostics() -> Diagnostics:
    """Get the global diagnostics instance"""
    global _diagnostics
    if _diagnostics is None:
        _diagnostics = Diagnostics()
    return _diagnostics

if __name__ == "__main__":
    # Example usage
    diagnostics = get_diagnostics()
    
    # Log some messages
    diagnostics.log_message("INFO", "System started", "core_manager")
    diagnostics.log_message("WARNING", "High memory usage detected", "memory_allocator", 
                          {"usage_percent": 85.5})
    diagnostics.log_message("ERROR", "Failed to allocate buffer", "pooling_engine", 
                          {"size": 1024*1024, "error": "OOM"})
    
    # Log some metrics
    diagnostics.log_metric("gpu_utilization", 75.5, "telemetry_layer", {"gpu_id": 0})
    diagnostics.log_metric("memory_allocated_gb", 2.5, "memory_allocator")
    diagnostics.log_metric("efficiency_score", 1250.0, "energy_controller")
    
    # Export to JSON
    diagnostics.export_to_json("diagnostics_export.json")
    
    # Export to CSV
    diagnostics.export_to_csv("metrics_export.csv")
    
    # Export to Prometheus
    diagnostics.export_to_prometheus("metrics_prometheus.txt")
    
    # Show recent logs
    recent_logs = diagnostics.get_recent_logs(5)
    print(f"\nRecent logs ({len(recent_logs)}):")
    for log in recent_logs:
        print(f"  {log['message']}")