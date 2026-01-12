"""
Compatibility Layer for MAHIA OptiCore
Provides Triton-compatible API for seamless integration.
"""

from typing import Dict, Any, Optional, List, Callable, Union
import functools

# Import OptiCore components
from .core_manager import get_core_manager
from .memory_allocator import get_memory_allocator
from .pooling_engine import get_pooling_engine
from .activation_checkpoint import get_activation_checkpoint
from .precision_tuner import get_precision_tuner
from .telemetry_layer import get_telemetry_layer
from .energy_controller import get_energy_controller

class OptiCoreCompatibilityLayer:
    """Triton-compatible API for OptiCore"""
    
    def __init__(self):
        # Initialize all OptiCore components
        self.core_manager = get_core_manager()
        self.memory_allocator = get_memory_allocator()
        self.pooling_engine = get_pooling_engine()
        self.activation_checkpoint = get_activation_checkpoint()
        self.precision_tuner = get_precision_tuner()
        self.telemetry_layer = get_telemetry_layer()
        self.energy_controller = get_energy_controller()
        
        print("🔄 OptiCore Compatibility Layer initialized")
        
    # Memory Management API (Triton-compatible)
    def memory_allocate(self, size: int, device: str = "cuda") -> Any:
        """
        Allocate memory (Triton-compatible).
        
        Args:
            size: Size in bytes
            device: Target device
            
        Returns:
            Allocated memory block
        """
        return self.memory_allocator.allocate(size, device)
        
    def memory_deallocate(self, block: Any) -> bool:
        """
        Deallocate memory (Triton-compatible).
        
        Args:
            block: Memory block to deallocate
            
        Returns:
            bool: True if successful
        """
        return self.memory_allocator.deallocate(block)
        
    # Kernel Launch API (Triton-compatible)
    def launch_kernel(self, kernel_fn: Callable, grid: tuple, 
                     args: tuple, kwargs: Optional[Dict[str, Any]] = None) -> Any:
        """
        Launch kernel (Triton-compatible).
        
        Args:
            kernel_fn: Kernel function to execute
            grid: Grid dimensions
            args: Positional arguments
            kwargs: Keyword arguments
            
        Returns:
            Kernel execution result
        """
        # Dispatch through CoreManager for proper scheduling
        result_container = []
        
        def kernel_wrapper():
            try:
                result = kernel_fn(*args, **(kwargs or {}))
                result_container.append(result)
            except Exception as e:
                result_container.append(e)
                
        self.core_manager.dispatch("kernel_launch", kernel_wrapper)
        
        # Wait for completion (in real implementation, this would be async)
        import time
        time.sleep(0.001)  # Simulate kernel execution time
        
        if result_container and isinstance(result_container[0], Exception):
            raise result_container[0]
            
        return result_container[0] if result_container else None
        
    # Autograd Support API
    def checkpoint_layer(self, layer_id: str, activations: Any, 
                        metadata: Optional[Dict[str, Any]] = None) -> bool:
        """
        Checkpoint layer activations.
        
        Args:
            layer_id: Layer identifier
            activations: Activations to checkpoint
            metadata: Additional metadata
            
        Returns:
            bool: True if successful
        """
        return self.activation_checkpoint.checkpoint(layer_id, activations, metadata)
        
    def restore_layer(self, layer_id: str, recompute_fn: Optional[Callable] = None) -> Any:
        """
        Restore layer activations.
        
        Args:
            layer_id: Layer identifier
            recompute_fn: Function to recompute if not found
            
        Returns:
            Restored activations
        """
        return self.activation_checkpoint.restore(layer_id, recompute_fn)
        
    # Precision Management API
    def tune_precision(self, loss: float, gradients: List[Any], 
                      force_precision: Optional[str] = None) -> str:
        """
        Tune precision based on loss and gradients.
        
        Args:
            loss: Current loss value
            gradients: List of gradients
            force_precision: Force specific precision
            
        Returns:
            str: Recommended precision
        """
        return self.precision_tuner.tune_precision(loss, gradients, force_precision)
        
    # Telemetry API
    def get_metric(self, metric_name: str) -> Any:
        """
        Get telemetry metric (Triton-compatible).
        
        Args:
            metric_name: Name of metric
            
        Returns:
            Metric value
        """
        latest = self.telemetry_layer.get_latest_metric(metric_name)
        return latest[1] if latest else None
        
    def record_metric(self, metric_name: str, value: Union[float, int]):
        """
        Record telemetry metric.
        
        Args:
            metric_name: Name of metric
            value: Metric value
        """
        self.telemetry_layer.record_metric(metric_name, value)
        
    # Energy Management API
    def optimize_energy_step(self, batch_time: float, batch_size: int, 
                           current_power: float) -> Dict[str, Any]:
        """
        Optimize energy efficiency for current step.
        
        Args:
            batch_time: Time to process batch
            batch_size: Number of samples
            current_power: Current power consumption
            
        Returns:
            Dict with optimization recommendations
        """
        return self.energy_controller.optimize_step(batch_time, batch_size, current_power)
        
    # Context Manager for backward compatibility
    def __enter__(self):
        """Context manager entry"""
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        pass

# Global compatibility layer instance
_compatibility_layer = None

def get_compatibility_layer() -> OptiCoreCompatibilityLayer:
    """Get the global compatibility layer instance"""
    global _compatibility_layer
    if _compatibility_layer is None:
        _compatibility_layer = OptiCoreCompatibilityLayer()
    return _compatibility_layer

# Convenience functions for direct API access
def memory_allocate(size: int, device: str = "cuda") -> Any:
    """Allocate memory"""
    return get_compatibility_layer().memory_allocate(size, device)
    
def memory_deallocate(block: Any) -> bool:
    """Deallocate memory"""
    return get_compatibility_layer().memory_deallocate(block)
    
def launch_kernel(kernel_fn: Callable, grid: tuple, 
                 args: tuple, kwargs: Optional[Dict[str, Any]] = None) -> Any:
    """Launch kernel"""
    return get_compatibility_layer().launch_kernel(kernel_fn, grid, args, kwargs)
    
def checkpoint_layer(layer_id: str, activations: Any, 
                    metadata: Optional[Dict[str, Any]] = None) -> bool:
    """Checkpoint layer"""
    return get_compatibility_layer().checkpoint_layer(layer_id, activations, metadata)
    
def restore_layer(layer_id: str, recompute_fn: Optional[Callable] = None) -> Any:
    """Restore layer"""
    return get_compatibility_layer().restore_layer(layer_id, recompute_fn)
    
def tune_precision(loss: float, gradients: List[Any], 
                  force_precision: Optional[str] = None) -> str:
    """Tune precision"""
    return get_compatibility_layer().tune_precision(loss, gradients, force_precision)
    
def get_metric(metric_name: str) -> Any:
    """Get metric"""
    return get_compatibility_layer().get_metric(metric_name)
    
def record_metric(metric_name: str, value: Union[float, int]):
    """Record metric"""
    get_compatibility_layer().record_metric(metric_name, value)
    
def optimize_energy_step(batch_time: float, batch_size: int, 
                        current_power: float) -> Dict[str, Any]:
    """Optimize energy step"""
    return get_compatibility_layer().optimize_energy_step(batch_time, batch_size, current_power)

# Decorator for Triton kernel compatibility
def jit(func):
    """
    Decorator for Triton kernel compatibility.
    In OptiCore, this provides optimized execution through CoreManager.
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        # Get compatibility layer
        compat_layer = get_compatibility_layer()
        
        # Execute through CoreManager for proper scheduling
        return compat_layer.launch_kernel(func, (1,), args, kwargs)
        
    return wrapper

if __name__ == "__main__":
    # Example usage
    compat = get_compatibility_layer()
    
    # Allocate memory
    buffer = compat.memory_allocate(1024 * 1024)  # 1MB
    print(f"Allocated buffer: {buffer is not None}")
    
    # Deallocate memory
    success = compat.memory_deallocate(buffer)
    print(f"Deallocation success: {success}")
    
    # Record a metric
    compat.record_metric("test_metric", 42.5)
    
    # Get metric
    value = compat.get_metric("test_metric")
    print(f"Retrieved metric: {value}")