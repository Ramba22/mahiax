"""
MAHIA OptiCore Main Module
Central interface for all OptiCore functionality.
"""

import time
from typing import Dict, Any, Optional, List
from contextlib import contextmanager

# Import all OptiCore components
from .core_manager import get_core_manager
from .memory_allocator import get_memory_allocator
from .pooling_engine import get_pooling_engine
from .activation_checkpoint import get_activation_checkpoint
from .precision_tuner import get_precision_tuner
from .telemetry_layer import get_telemetry_layer
from .energy_controller import get_energy_controller
from .diagnostics import get_diagnostics
from .compatibility import get_compatibility_layer

class OptiCore:
    """Main OptiCore interface"""
    
    def __init__(self):
        # Initialize all components
        self.core_manager = get_core_manager()
        self.memory_allocator = get_memory_allocator()
        self.pooling_engine = get_pooling_engine()
        self.activation_checkpoint = get_activation_checkpoint()
        self.precision_tuner = get_precision_tuner()
        self.telemetry_layer = get_telemetry_layer()
        self.energy_controller = get_energy_controller()
        self.diagnostics = get_diagnostics()
        self.compatibility_layer = get_compatibility_layer()
        
        self.is_initialized = False
        self.is_monitoring = False
        
        print("🚀 MAHIA OptiCore main interface initialized")
        
    def initialize(self, start_monitoring: bool = True) -> bool:
        """
        Initialize OptiCore system.
        
        Args:
            start_monitoring: Whether to start monitoring automatically
            
        Returns:
            bool: True if successful
        """
        try:
            # Start core manager
            self.core_manager.start()
            
            # Start monitoring if requested
            if start_monitoring:
                self.start_monitoring()
                
            self.is_initialized = True
            self.diagnostics.log_message("INFO", "OptiCore initialized successfully", "opticore")
            print("✅ OptiCore initialization complete")
            return True
            
        except Exception as e:
            self.diagnostics.log_message("ERROR", f"OptiCore initialization failed: {e}", "opticore")
            print(f"❌ OptiCore initialization failed: {e}")
            return False
            
    def shutdown(self):
        """Shutdown OptiCore system"""
        try:
            # Stop monitoring
            self.stop_monitoring()
            
            # Stop core manager
            self.core_manager.stop()
            
            self.is_initialized = False
            self.diagnostics.log_message("INFO", "OptiCore shutdown complete", "opticore")
            print("🛑 OptiCore shutdown complete")
            
        except Exception as e:
            self.diagnostics.log_message("ERROR", f"OptiCore shutdown error: {e}", "opticore")
            print(f"❌ OptiCore shutdown error: {e}")
            
    def start_monitoring(self):
        """Start all monitoring systems"""
        if self.is_monitoring:
            print("⚠️  Monitoring is already running")
            return
            
        try:
            self.memory_allocator.start_monitoring()
            self.telemetry_layer.start_monitoring()
            self.is_monitoring = True
            self.diagnostics.log_message("INFO", "Monitoring started", "opticore")
            print("📈 Monitoring started")
            
        except Exception as e:
            self.diagnostics.log_message("ERROR", f"Failed to start monitoring: {e}", "opticore")
            print(f"❌ Failed to start monitoring: {e}")
            
    def stop_monitoring(self):
        """Stop all monitoring systems"""
        if not self.is_monitoring:
            print("⚠️  Monitoring is not running")
            return
            
        try:
            self.memory_allocator.stop_monitoring()
            self.telemetry_layer.stop_monitoring()
            self.is_monitoring = False
            self.diagnostics.log_message("INFO", "Monitoring stopped", "opticore")
            print("⏹️  Monitoring stopped")
            
        except Exception as e:
            self.diagnostics.log_message("ERROR", f"Failed to stop monitoring: {e}", "opticore")
            print(f"❌ Failed to stop monitoring: {e}")
            
    def get_system_status(self) -> Dict[str, Any]:
        """
        Get overall system status.
        
        Returns:
            Dict with system status information
        """
        return {
            "initialized": self.is_initialized,
            "monitoring": self.is_monitoring,
            "core_manager_stats": self.core_manager.get_stats(),
            "memory_allocator_stats": self.memory_allocator.get_stats(),
            "pooling_engine_stats": self.pooling_engine.get_stats(),
            "activation_checkpoint_stats": self.activation_checkpoint.get_stats(),
            "precision_tuner_stats": self.precision_tuner.get_stats(),
            "energy_controller_stats": self.energy_controller.get_stats(),
            "telemetry_metrics": len(self.telemetry_layer.get_metric_names()),
            "uptime_seconds": time.time() - getattr(self, '_start_time', time.time())
        }
        
    def export_diagnostics(self, base_path: str = "opticore_export") -> bool:
        """
        Export all diagnostics data.
        
        Args:
            base_path: Base path for export files
            
        Returns:
            bool: True if successful
        """
        try:
            timestamp = int(time.time())
            base_filename = f"{base_path}_{timestamp}"
            
            # Export to JSON
            json_success = self.diagnostics.export_to_json(f"{base_filename}.json")
            
            # Export metrics to CSV
            csv_success = self.diagnostics.export_to_csv(f"{base_filename}_metrics.csv")
            
            # Export to Prometheus format
            prom_success = self.diagnostics.export_to_prometheus(f"{base_filename}_prometheus.txt")
            
            if json_success and csv_success and prom_success:
                self.diagnostics.log_message("INFO", "Diagnostics exported successfully", "opticore")
                print(f"💾 Diagnostics exported to {base_filename}*")
                return True
            else:
                self.diagnostics.log_message("WARNING", "Some diagnostics exports failed", "opticore")
                print("⚠️  Some diagnostics exports failed")
                return False
                
        except Exception as e:
            self.diagnostics.log_message("ERROR", f"Diagnostics export failed: {e}", "opticore")
            print(f"❌ Diagnostics export failed: {e}")
            return False
            
    @contextmanager
    def optimization_context(self):
        """Context manager for optimization-aware operations"""
        start_time = time.time()
        try:
            # Record start metrics
            start_power = self.telemetry_layer.get_latest_metric("gpu_0_power")
            start_power_value = start_power[1] if start_power else 0.0
            
            yield self
            
            # Record end metrics and calculate optimization impact
            end_time = time.time()
            end_power = self.telemetry_layer.get_latest_metric("gpu_0_power")
            end_power_value = end_power[1] if end_power else 0.0
            
            # Calculate energy efficiency
            time_elapsed = end_time - start_time
            if time_elapsed > 0:
                self.energy_controller.calculate_efficiency_score(
                    (start_power_value + end_power_value) / 2,  # Average power
                    1.0 / time_elapsed,  # Performance (operations per second)
                    time_elapsed
                )
                
        except Exception as e:
            self.diagnostics.log_message("ERROR", f"Optimization context error: {e}", "opticore")
            raise e

# Global OptiCore instance
_opticore = None

__all__ = [
    "OptiCore",
    "get_opticore",
    "initialize_opticore",
    "shutdown_opticore",
    "opticore_memory",
    "opticore_pooling",
    "opticore_checkpoint",
    "opticore_precision",
    "opticore_telemetry",
    "opticore_energy",
    "opticore_diagnostics"
]

def get_opticore() -> OptiCore:
    """Get the global OptiCore instance"""
    global _opticore
    if _opticore is None:
        _opticore = OptiCore()
    return _opticore

def initialize_opticore(start_monitoring: bool = True) -> bool:
    """Initialize OptiCore system"""
    return get_opticore().initialize(start_monitoring)

def shutdown_opticore():
    """Shutdown OptiCore system"""
    if _opticore is not None:
        _opticore.shutdown()

# Convenience access to individual components
def opticore_memory():
    """Get memory allocator"""
    return get_opticore().memory_allocator
    
def opticore_pooling():
    """Get pooling engine"""
    return get_opticore().pooling_engine
    
def opticore_checkpoint():
    """Get activation checkpoint controller"""
    return get_opticore().activation_checkpoint
    
def opticore_precision():
    """Get precision tuner"""
    return get_opticore().precision_tuner
    
def opticore_telemetry():
    """Get telemetry layer"""
    return get_opticore().telemetry_layer
    
def opticore_energy():
    """Get energy controller"""
    return get_opticore().energy_controller
    
def opticore_diagnostics():
    """Get diagnostics"""
    return get_opticore().diagnostics

if __name__ == "__main__":
    # Example usage
    opticore = get_opticore()
    
    # Initialize system
    if opticore.initialize():
        # Get system status
        status = opticore.get_system_status()
        print(f"System Status: {status}")
        
        # Run some operations in optimization context
        with opticore.optimization_context():
            # Simulate some work
            time.sleep(1)
            
        # Export diagnostics
        opticore.export_diagnostics()
        
        # Shutdown
        opticore.shutdown()