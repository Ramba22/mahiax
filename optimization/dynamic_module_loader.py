"""
Dynamic Module Loader for MAHIA-X
Implements dynamic loading/unloading of modules for efficient resource usage
"""

import importlib
import importlib.util
import sys
import os
import time
import gc
from typing import Dict, Any, Optional, List, Callable
from collections import OrderedDict
from datetime import datetime
import threading

class ModuleManager:
    """Manages dynamic loading and unloading of modules"""
    
    def __init__(self, max_modules: int = 100):
        self.max_modules = max_modules
        self.loaded_modules = OrderedDict()  # module_name -> module_object
        self.module_usage = {}  # module_name -> usage_stats
        self.module_dependencies = {}  # module_name -> list of dependencies
        self.lock = threading.Lock()
        
    def load_module(self, module_name: str, module_path: Optional[str] = None) -> Any:
        """Dynamically load a module"""
        with self.lock:
            # Check if already loaded
            if module_name in self.loaded_modules:
                # Move to end to mark as recently used
                self.loaded_modules.move_to_end(module_name)
                self._update_usage_stats(module_name, 'load')
                return self.loaded_modules[module_name]
                
            try:
                module = None
                # Load module
                if module_path:
                    # Load from specific path
                    spec = importlib.util.spec_from_file_location(module_name, module_path)
                    if spec and spec.loader:
                        module = importlib.util.module_from_spec(spec)
                        spec.loader.exec_module(module)
                else:
                    # Load from standard import
                    module = importlib.import_module(module_name)
                    
                # Store module
                if module is not None:
                    self.loaded_modules[module_name] = module
                self._update_usage_stats(module_name, 'load')
                
                # Manage cache size
                self._manage_cache_size()
                
                return module if module is not None else None
                
            except Exception as e:
                print(f"Error loading module {module_name}: {e}")
                return None
                
    def unload_module(self, module_name: str) -> bool:
        """Unload a module to free resources"""
        with self.lock:
            if module_name in self.loaded_modules:
                # Remove from loaded modules
                del self.loaded_modules[module_name]
                
                # Remove from sys.modules if present
                if module_name in sys.modules:
                    del sys.modules[module_name]
                    
                # Update usage stats
                self._update_usage_stats(module_name, 'unload')
                return True
            return False
            
    def get_module(self, module_name: str, module_path: Optional[str] = None) -> Any:
        """Get module (load if not already loaded)"""
        return self.load_module(module_name, module_path)
        
    def release_unused_modules(self, threshold_seconds: int = 300) -> int:
        """Release modules that haven't been used for a while"""
        with self.lock:
            current_time = time.time()
            modules_to_unload = []
            
            for module_name, stats in self.module_usage.items():
                last_used = stats.get('last_used', 0)
                if current_time - last_used > threshold_seconds:
                    modules_to_unload.append(module_name)
                    
            # Unload identified modules
            unloaded_count = 0
            for module_name in modules_to_unload:
                if self.unload_module(module_name):
                    unloaded_count += 1
                    
            return unloaded_count
            
    def _update_usage_stats(self, module_name: str, action: str):
        """Update module usage statistics"""
        if module_name not in self.module_usage:
            self.module_usage[module_name] = {
                'load_count': 0,
                'unload_count': 0,
                'last_used': time.time(),
                'first_loaded': time.time()
            }
            
        stats = self.module_usage[module_name]
        stats['last_used'] = time.time()
        
        if action == 'load':
            stats['load_count'] += 1
        elif action == 'unload':
            stats['unload_count'] += 1
            
    def _manage_cache_size(self):
        """Manage cache size by removing least recently used modules"""
        if len(self.loaded_modules) > self.max_modules:
            # Remove least recently used module
            lru_module = next(iter(self.loaded_modules))
            self.unload_module(lru_module)
            
    def get_module_stats(self) -> Dict[str, Any]:
        """Get statistics about loaded modules"""
        return {
            'loaded_modules': list(self.loaded_modules.keys()),
            'usage_stats': self.module_usage.copy(),
            'cache_size': len(self.loaded_modules),
            'max_cache_size': self.max_modules
        }

class ResourceMonitor:
    """Monitors system resources and optimizes usage"""
    
    def __init__(self):
        self.monitoring = False
        self.monitor_thread = None
        self.resource_stats = {
            'memory_usage': 0,
            'cpu_usage': 0,
            'disk_usage': 0
        }
        self.optimization_callbacks = []
        
    def start_monitoring(self, interval: float = 5.0):
        """Start resource monitoring in background thread"""
        if not self.monitoring:
            self.monitoring = True
            self.monitor_thread = threading.Thread(
                target=self._monitor_loop, 
                args=(interval,),
                daemon=True
            )
            self.monitor_thread.start()
            
    def stop_monitoring(self):
        """Stop resource monitoring"""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join()
            
    def _monitor_loop(self, interval: float):
        """Main monitoring loop"""
        while self.monitoring:
            try:
                self._collect_resource_stats()
                self._trigger_optimizations()
                time.sleep(interval)
            except Exception as e:
                print(f"Error in resource monitoring: {e}")
                time.sleep(interval)
                
    def _collect_resource_stats(self):
        """Collect system resource statistics"""
        try:
            # Memory usage estimation (simplified)
            process_memory = len(gc.get_objects()) * 1000 if 'gc' in sys.modules else 0
            self.resource_stats['memory_usage'] = process_memory
            
            # CPU usage would require psutil or similar (not imported for simplicity)
            # For now, we'll use a placeholder
            self.resource_stats['cpu_usage'] = 0
            
            # Disk usage estimation (simplified)
            if os.name == 'nt':  # Windows
                disk_stats = os.statvfs('.') if hasattr(os, 'statvfs') else None
            else:
                disk_stats = os.statvfs('.') if hasattr(os, 'statvfs') else None
                
            self.resource_stats['disk_usage'] = disk_stats.f_bavail * disk_stats.f_frsize if disk_stats else 0
            
        except Exception as e:
            print(f"Error collecting resource stats: {e}")
            
    def _trigger_optimizations(self):
        """Trigger optimization callbacks based on resource usage"""
        # Check if optimizations should be triggered
        memory_high = self.resource_stats['memory_usage'] > 100000000  # 100MB threshold
        cpu_high = self.resource_stats['cpu_usage'] > 80.0  # 80% CPU threshold
        
        if memory_high or cpu_high:
            for callback in self.optimization_callbacks:
                try:
                    callback(self.resource_stats)
                except Exception as e:
                    print(f"Error in optimization callback: {e}")
                    
    def register_optimization_callback(self, callback: Callable):
        """Register a callback to be called when optimizations are needed"""
        self.optimization_callbacks.append(callback)
        
    def get_resource_stats(self) -> Dict[str, Any]:
        """Get current resource statistics"""
        return self.resource_stats.copy()

class DynamicModuleLoader:
    """Main interface for dynamic module loading with optimization"""
    
    def __init__(self):
        self.module_manager = ModuleManager()
        self.resource_monitor = ResourceMonitor()
        self.optimization_rules = []
        
        # Register resource monitor callback
        self.resource_monitor.register_optimization_callback(
            self._resource_optimization_callback
        )
        
    def load_module_on_demand(self, module_name: str, module_path: Optional[str] = None) -> Any:
        """Load module only when needed"""
        return self.module_manager.load_module(module_name, module_path)
        
    def unload_unused_modules(self, threshold_seconds: int = 300) -> int:
        """Unload modules that haven't been used recently"""
        return self.module_manager.release_unused_modules(threshold_seconds)
        
    def start_resource_monitoring(self, interval: float = 5.0):
        """Start monitoring system resources"""
        self.resource_monitor.start_monitoring(interval)
        
    def stop_resource_monitoring(self):
        """Stop monitoring system resources"""
        self.resource_monitor.stop_monitoring()
        
    def _resource_optimization_callback(self, resource_stats: Dict[str, Any]):
        """Callback for resource-based optimizations"""
        # If memory usage is high, unload unused modules
        if resource_stats['memory_usage'] > 100000000:  # 100MB
            unloaded = self.unload_unused_modules(120)  # Unload modules unused for 2 minutes
            if unloaded > 0:
                print(f"Unloaded {unloaded} modules to reduce memory usage")
                
    def add_optimization_rule(self, rule_func: Callable[[Dict[str, Any]], bool], 
                            action_func: Callable):
        """Add a custom optimization rule"""
        self.optimization_rules.append((rule_func, action_func))
        
    def apply_optimization_rules(self):
        """Apply custom optimization rules"""
        resource_stats = self.resource_monitor.get_resource_stats()
        
        for rule_func, action_func in self.optimization_rules:
            try:
                if rule_func(resource_stats):
                    action_func()
            except Exception as e:
                print(f"Error applying optimization rule: {e}")
                
    def get_system_status(self) -> Dict[str, Any]:
        """Get overall system status"""
        return {
            'modules': self.module_manager.get_module_stats(),
            'resources': self.resource_monitor.get_resource_stats(),
            'monitoring_active': self.resource_monitor.monitoring
        }

# MAHIA OptiCore - Core optimization system
class MAHIAOptiCore:
    """Core optimization system for MAHIA-X"""
    
    def __init__(self):
        self.dynamic_loader = DynamicModuleLoader()
        self.performance_profiles = {}
        self.optimization_history = []
        
    def optimize_for_task(self, task_type: str, required_modules: List[str]) -> Dict[str, Any]:
        """Optimize system for a specific task type"""
        optimization_result = {
            'task_type': task_type,
            'modules_loaded': [],
            'optimization_applied': False,
            'timestamp': datetime.now().isoformat()
        }
        
        # Load required modules
        for module_name in required_modules:
            module = self.dynamic_loader.load_module_on_demand(module_name)
            if module:
                optimization_result['modules_loaded'].append(module_name)
                
        # Apply task-specific optimizations
        if task_type == 'inference':
            # Optimize for inference (load minimal modules)
            self._optimize_for_inference()
            optimization_result['optimization_applied'] = True
        elif task_type == 'training':
            # Optimize for training (load compute modules)
            self._optimize_for_training()
            optimization_result['optimization_applied'] = True
        elif task_type == 'analysis':
            # Optimize for analysis (load data modules)
            self._optimize_for_analysis()
            optimization_result['optimization_applied'] = True
            
        # Record optimization
        self.optimization_history.append(optimization_result)
        return optimization_result
        
    def _optimize_for_inference(self):
        """Optimize system for inference tasks"""
        # Unload training-specific modules
        self.dynamic_loader.unload_unused_modules(60)  # 1 minute threshold
        
    def _optimize_for_training(self):
        """Optimize system for training tasks"""
        # Ensure compute modules are loaded
        pass
        
    def _optimize_for_analysis(self):
        """Optimize system for analysis tasks"""
        # Ensure data processing modules are loaded
        pass
        
    def get_optimization_report(self) -> Dict[str, Any]:
        """Get optimization report"""
        return {
            'system_status': self.dynamic_loader.get_system_status(),
            'optimization_history': self.optimization_history[-10:],  # Last 10 optimizations
            'performance_profiles': self.performance_profiles
        }

# Example usage
if __name__ == "__main__":
    import gc
    
    # Initialize optimization system
    opti_core = MAHIAOptiCore()
    
    # Start resource monitoring
    opti_core.dynamic_loader.start_resource_monitoring(2.0)
    
    # Optimize for inference task
    result = opti_core.optimize_for_task('inference', ['json', 're'])
    print("Optimization Result:", result)
    
    # Get system status
    status = opti_core.get_optimization_report()
    print("System Status:", status)
    
    # Stop monitoring
    opti_core.dynamic_loader.stop_resource_monitoring()