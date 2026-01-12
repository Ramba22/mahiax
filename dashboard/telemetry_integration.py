"""
Telemetry Integration for MAHIA Dashboard V3
Bridges the telemetry system with the dashboard WebSocket server
"""

import json
import requests
import threading
import time
from typing import Dict, Any, Optional
from telemetry.standardized_telemetry import StandardizedTelemetryAPI

class DashboardTelemetryBridge:
    """Bridge between telemetry system and dashboard WebSocket server"""
    
    def __init__(self, 
                 telemetry_api: StandardizedTelemetryAPI,
                 dashboard_url: str = "http://localhost:8000/api/metrics"):
        """
        Initialize the telemetry bridge
        
        Args:
            telemetry_api: StandardizedTelemetryAPI instance
            dashboard_url: URL of the dashboard WebSocket server
        """
        self.telemetry_api = telemetry_api
        self.dashboard_url = dashboard_url
        self.running = False
        self.bridge_thread = None
        self.last_sent_metrics = {}
        
    def start(self):
        """Start the telemetry bridge"""
        if not self.running:
            self.running = True
            self.bridge_thread = threading.Thread(target=self._bridge_loop, daemon=True)
            self.bridge_thread.start()
            print("🌉 Telemetry bridge started")
            
    def stop(self):
        """Stop the telemetry bridge"""
        self.running = False
        if self.bridge_thread:
            self.bridge_thread.join()
        print("🛑 Telemetry bridge stopped")
        
    def _bridge_loop(self):
        """Main bridge loop - periodically send metrics to dashboard"""
        while self.running:
            try:
                # Send metrics to dashboard
                self._send_metrics_to_dashboard()
                time.sleep(1)  # Send every second
            except Exception as e:
                print(f"⚠️  Bridge error: {e}")
                time.sleep(5)  # Wait longer on error
                
    def _send_metrics_to_dashboard(self):
        """Send current metrics to dashboard"""
        # This is a simplified implementation
        # In a real implementation, you would extract metrics from the telemetry API
        # and send them to the dashboard
        
        # For now, we'll just send a heartbeat
        try:
            payload = {
                "type": "heartbeat",
                "timestamp": time.time(),
                "status": "active"
            }
            
            response = requests.post(self.dashboard_url, json=payload, timeout=5)
            if response.status_code == 200:
                print("✅ Heartbeat sent to dashboard")
            else:
                print(f"⚠️  Failed to send heartbeat: {response.status_code}")
        except Exception as e:
            print(f"⚠️  Failed to send heartbeat: {e}")

class TelemetryDashboardLogger:
    """Logger that sends telemetry data directly to dashboard"""
    
    def __init__(self, dashboard_url: str = "http://localhost:8000/api/metrics"):
        """
        Initialize the dashboard logger
        
        Args:
            dashboard_url: URL of the dashboard WebSocket server
        """
        self.dashboard_url = dashboard_url
        
    def log_metric(self, metric_type: str, data: Dict[str, Any]):
        """
        Log a metric to the dashboard
        
        Args:
            metric_type: Type of metric (e.g., "loss", "lr", "entropy")
            data: Metric data
        """
        try:
            payload = {
                "type": metric_type,
                "data": data,
                "timestamp": time.time()
            }
            
            response = requests.post(self.dashboard_url, json=payload, timeout=5)
            if response.status_code != 200:
                print(f"⚠️  Failed to log metric {metric_type}: {response.status_code}")
        except Exception as e:
            print(f"⚠️  Failed to log metric {metric_type}: {e}")
            
    def log_loss(self, step: int, train_loss: float, val_loss: Optional[float] = None):
        """Log loss metrics"""
        data = {
            "step": step,
            "train_loss": train_loss
        }
        if val_loss is not None:
            data["val_loss"] = val_loss
            
        self.log_metric("loss", data)
        
    def log_learning_rate(self, step: int, lr: float, scheduler: str = "unknown"):
        """Log learning rate"""
        data = {
            "step": step,
            "lr": lr,
            "scheduler": scheduler
        }
        self.log_metric("learning_rate", data)
        
    def log_entropy(self, step: int, entropy: float, gradient_norm: Optional[float] = None):
        """Log entropy metrics"""
        data = {
            "step": step,
            "entropy": entropy
        }
        if gradient_norm is not None:
            data["gradient_norm"] = gradient_norm
            
        self.log_metric("entropy", data)
        
    def log_gpu_utilization(self, step: int, gpu_util: float, memory_used: float, 
                          temperature: Optional[float] = None, power: Optional[float] = None):
        """Log GPU utilization metrics"""
        data = {
            "step": step,
            "gpu_util": gpu_util,
            "memory_used": memory_used
        }
        if temperature is not None:
            data["temperature"] = temperature
        if power is not None:
            data["power"] = power
            
        self.log_metric("gpu", data)
        
    def log_controller_action(self, step: int, controller: str, action: str, 
                            confidence: Optional[float] = None):
        """Log controller action"""
        data = {
            "step": step,
            "controller": controller,
            "action": action
        }
        if confidence is not None:
            data["confidence"] = confidence
            
        self.log_metric("controller", data)

# Example usage
def example_integration():
    """Example of how to integrate telemetry with dashboard"""
    print("🔧 Setting up telemetry-dashboard integration example...")
    
    # Create telemetry API
    telemetry_api = StandardizedTelemetryAPI(
        log_dir="./logs/demo",
        experiment_name="dashboard_demo"
    )
    
    # Create dashboard logger
    dashboard_logger = TelemetryDashboardLogger("http://localhost:8000/api/metrics")
    
    # Create bridge
    bridge = DashboardTelemetryBridge(telemetry_api, "http://localhost:8000/api/metrics")
    
    # Start bridge
    bridge.start()
    
    # Simulate some logging
    for step in range(10):
        # Log various metrics
        dashboard_logger.log_loss(step, 1.0 - step * 0.1, 1.1 - step * 0.08)
        dashboard_logger.log_learning_rate(step, 0.001 * (0.95 ** step))
        dashboard_logger.log_entropy(step, 1.0 - step * 0.1)
        dashboard_logger.log_gpu_utilization(step, 50 + step * 2, 1000 + step * 50)
        dashboard_logger.log_controller_action(step, "ExtendStop", "wait" if step % 3 != 2 else "extend")
        
        time.sleep(0.5)
        
    # Stop bridge
    bridge.stop()
    
    print("✅ Telemetry-dashboard integration example completed!")

if __name__ == "__main__":
    example_integration()