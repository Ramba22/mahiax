"""
Efficiency Optimizer for MAHIA-X
Implements dynamic module loading, resource management, and performance optimization
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Any, Optional, List, Tuple, Set
from collections import OrderedDict, defaultdict
import time
import threading
import gc
from datetime import datetime
import psutil
import os

# Conditional imports
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    
try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

class DynamicModuleManager:
    """Manages dynamic loading and unloading of model modules"""
    
    def __init__(self, max_modules: int = 20):
        """
        Initialize dynamic module manager
        
        Args:
            max_modules: Maximum number of modules to keep loaded
        """
        self.max_modules = max_modules
        self.loaded_modules = OrderedDict()
        self.module_usage = defaultdict(int)
        self.module_load_times = {}
        self.module_memory_usage = {}
        
        # Lock for thread safety
        self.lock = threading.Lock()
        
        print(f"✅ DynamicModuleManager initialized with max_modules: {max_modules}")
        
    def load_module(self, module_name: str, module_factory: callable) -> Optional[nn.Module]:
        """
        Load module dynamically
        
        Args:
            module_name: Name of module to load
            module_factory: Factory function to create module
            
        Returns:
            Loaded module or None if failed
        """
        with self.lock:
            # Check if module is already loaded
            if module_name in self.loaded_modules:
                self.module_usage[module_name] += 1
                # Move to end to mark as recently used
                self.loaded_modules.move_to_end(module_name)
                return self.loaded_modules[module_name]
                
            try:
                # Create module
                load_start = time.time()
                module = module_factory()
                load_time = time.time() - load_start
                
                # Store module
                self.loaded_modules[module_name] = module
                self.module_usage[module_name] = 1
                self.module_load_times[module_name] = load_time
                
                # Estimate memory usage
                if TORCH_AVAILABLE:
                    memory_usage = sum(p.numel() * p.element_size() for p in module.parameters())
                    self.module_memory_usage[module_name] = memory_usage
                    
                # Manage module cache size
                self._manage_cache_size()
                
                print(f"✅ Loaded module {module_name} in {load_time:.3f}s")
                return module
                
            except Exception as e:
                print(f"❌ Failed to load module {module_name}: {e}")
                return None
                
    def unload_module(self, module_name: str) -> bool:
        """
        Unload module to free resources
        
        Args:
            module_name: Name of module to unload
            
        Returns:
            True if successful, False otherwise
        """
        with self.lock:
            if module_name in self.loaded_modules:
                # Delete module
                del self.loaded_modules[module_name]
                
                # Remove usage tracking
                if module_name in self.module_usage:
                    del self.module_usage[module_name]
                if module_name in self.module_load_times:
                    del self.module_load_times[module_name]
                if module_name in self.module_memory_usage:
                    del self.module_memory_usage[module_name]
                    
                # Force garbage collection
                gc.collect()
                
                print(f"🗑️  Unloaded module {module_name}")
                return True
            return False
            
    def _manage_cache_size(self):
        """Manage module cache size to prevent memory overflow"""
        if len(self.loaded_modules) > self.max_modules:
            # Remove least recently used modules
            while len(self.loaded_modules) > self.max_modules:
                # Get least used module
                lru_module = next(iter(self.loaded_modules))
                self.unload_module(lru_module)
                
    def get_module_info(self) -> Dict[str, Any]:
        """
        Get information about loaded modules
        
        Returns:
            Module information dictionary
        """
        with self.lock:
            return {
                "loaded_modules": list(self.loaded_modules.keys()),
                "module_usage": dict(self.module_usage),
                "module_load_times": self.module_load_times,
                "module_memory_usage": self.module_memory_usage,
                "total_modules": len(self.loaded_modules),
                "max_modules": self.max_modules
            }
            
    def preload_modules(self, module_specs: List[Tuple[str, callable]]):
        """
        Preload multiple modules
        
        Args:
            module_specs: List of (module_name, factory_function) tuples
        """
        print(f"🔄 Preloading {len(module_specs)} modules...")
        
        for module_name, factory in module_specs:
            self.load_module(module_name, factory)
            
        print("✅ Module preloading completed")


class ResourceMonitor:
    """Monitors system resources and optimizes performance"""
    
    def __init__(self, memory_threshold: float = 0.8, cpu_threshold: float = 0.8):
        """
        Initialize resource monitor
        
        Args:
            memory_threshold: Memory usage threshold (0.0 to 1.0)
            cpu_threshold: CPU usage threshold (0.0 to 1.0)
        """
        self.memory_threshold = memory_threshold
        self.cpu_threshold = cpu_threshold
        self.monitoring_stats = {
            "peak_memory_usage": 0.0,
            "peak_cpu_usage": 0.0,
            "optimization_triggers": 0,
            "memory_reclaimed": 0
        }
        
        # Monitoring thread
        self.monitoring_thread = None
        self.monitoring_active = False
        
        print(f"✅ ResourceMonitor initialized")
        print(f"   Memory threshold: {memory_threshold:.1%}")
        print(f"   CPU threshold: {cpu_threshold:.1%}")
        
    def start_monitoring(self, interval: float = 5.0):
        """
        Start resource monitoring thread
        
        Args:
            interval: Monitoring interval in seconds
        """
        if self.monitoring_active:
            return
            
        self.monitoring_active = True
        self.monitoring_thread = threading.Thread(
            target=self._monitoring_worker,
            args=(interval,),
            daemon=True
        )
        self.monitoring_thread.start()
        print("✅ Resource monitoring started")
        
    def stop_monitoring(self):
        """Stop resource monitoring"""
        self.monitoring_active = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=2.0)
        print("✅ Resource monitoring stopped")
        
    def _monitoring_worker(self, interval: float):
        """Background monitoring worker"""
        while self.monitoring_active:
            try:
                # Check resources
                memory_usage = self._get_memory_usage()
                cpu_usage = self._get_cpu_usage()
                
                # Update peak values
                self.monitoring_stats["peak_memory_usage"] = max(
                    self.monitoring_stats["peak_memory_usage"],
                    memory_usage
                )
                self.monitoring_stats["peak_cpu_usage"] = max(
                    self.monitoring_stats["peak_cpu_usage"],
                    cpu_usage
                )
                
                # Trigger optimization if thresholds exceeded
                if memory_usage > self.memory_threshold or cpu_usage > self.cpu_threshold:
                    self._trigger_optimization(memory_usage, cpu_usage)
                    
                time.sleep(interval)
                
            except Exception as e:
                print(f"⚠️  Monitoring error: {e}")
                time.sleep(interval)
                
    def _get_memory_usage(self) -> float:
        """Get current memory usage as ratio"""
        if PSUTIL_AVAILABLE:
            return psutil.virtual_memory().percent / 100.0
        elif TORCH_AVAILABLE:
            # Fallback to torch memory info
            if torch.cuda.is_available():
                return torch.cuda.memory_allocated() / torch.cuda.get_device_properties(0).total_memory
            else:
                # Estimate based on system info
                try:
                    process = psutil.Process(os.getpid())
                    return process.memory_percent() / 100.0
                except:
                    return 0.5  # Default estimate
        else:
            return 0.5  # Default estimate
            
    def _get_cpu_usage(self) -> float:
        """Get current CPU usage as ratio"""
        if PSUTIL_AVAILABLE:
            return psutil.cpu_percent() / 100.0
        else:
            return 0.5  # Default estimate
            
    def _trigger_optimization(self, memory_usage: float, cpu_usage: float):
        """Trigger optimization based on resource usage"""
        self.monitoring_stats["optimization_triggers"] += 1
        print(f"⚡ Optimization triggered - Memory: {memory_usage:.1%}, CPU: {cpu_usage:.1%}")
        
        # Force garbage collection
        collected = gc.collect()
        self.monitoring_stats["memory_reclaimed"] += collected
        
        # Clear torch cache if available
        if TORCH_AVAILABLE and torch.cuda.is_available():
            torch.cuda.empty_cache()
            
    def get_resource_stats(self) -> Dict[str, Any]:
        """
        Get resource monitoring statistics
        
        Returns:
            Resource statistics dictionary
        """
        return {
            "timestamp": time.time(),
            "current_memory_usage": self._get_memory_usage(),
            "current_cpu_usage": self._get_cpu_usage(),
            "monitoring_stats": self.monitoring_stats,
            "monitoring_active": self.monitoring_active
        }


class ModularArchitecture:
    """Manages modular architecture with dynamic component loading"""
    
    def __init__(self, base_modules: Optional[List[str]] = None):
        """
        Initialize modular architecture
        
        Args:
            base_modules: List of base module names to always keep loaded
        """
        self.base_modules = base_modules or []
        self.dynamic_manager = DynamicModuleManager(max_modules=15)
        self.resource_monitor = ResourceMonitor(memory_threshold=0.85, cpu_threshold=0.85)
        
        # Module registry
        self.module_registry = {}
        self.module_dependencies = defaultdict(set)
        
        # Performance tracking
        self.performance_stats = {
            "modules_loaded": 0,
            "modules_unloaded": 0,
            "dynamic_loads": 0,
            "cache_hits": 0
        }
        
        # Start resource monitoring
        self.resource_monitor.start_monitoring(interval=3.0)
        
        print(f"✅ ModularArchitecture initialized")
        print(f"   Base modules: {len(self.base_modules)}")
        
    def register_module(self, module_name: str, factory_function: callable, 
                       dependencies: Optional[List[str]] = None):
        """
        Register a module in the system
        
        Args:
            module_name: Name of module
            factory_function: Factory function to create module
            dependencies: List of module dependencies
        """
        self.module_registry[module_name] = factory_function
        if dependencies:
            self.module_dependencies[module_name].update(dependencies)
            
        print(f"✅ Registered module: {module_name}")
        
    def get_module(self, module_name: str) -> Optional[nn.Module]:
        """
        Get module (load dynamically if needed)
        
        Args:
            module_name: Name of module to get
            
        Returns:
            Module instance or None
        """
        # Check if module is registered
        if module_name not in self.module_registry:
            print(f"❌ Module {module_name} not registered")
            return None
            
        # Check if already loaded
        module_info = self.dynamic_manager.get_module_info()
        if module_name in module_info["loaded_modules"]:
            self.performance_stats["cache_hits"] += 1
            return self.dynamic_manager.loaded_modules[module_name]
            
        # Load module dynamically
        self.performance_stats["dynamic_loads"] += 1
        factory = self.module_registry[module_name]
        module = self.dynamic_manager.load_module(module_name, factory)
        
        if module is not None:
            self.performance_stats["modules_loaded"] += 1
            
        return module
        
    def unload_unused_modules(self, keep_modules: Optional[List[str]] = None):
        """
        Unload modules that are not currently needed
        
        Args:
            keep_modules: List of modules to keep loaded
        """
        if keep_modules is None:
            keep_modules = self.base_modules
            
        module_info = self.dynamic_manager.get_module_info()
        loaded_modules = module_info["loaded_modules"]
        
        # Unload modules not in keep list
        for module_name in loaded_modules:
            if module_name not in keep_modules:
                success = self.dynamic_manager.unload_module(module_name)
                if success:
                    self.performance_stats["modules_unloaded"] += 1
                    
    def get_system_efficiency_stats(self) -> Dict[str, Any]:
        """
        Get comprehensive system efficiency statistics
        
        Returns:
            Efficiency statistics dictionary
        """
        module_info = self.dynamic_manager.get_module_info()
        resource_stats = self.resource_monitor.get_resource_stats()
        
        return {
            "timestamp": time.time(),
            "performance_stats": self.performance_stats,
            "module_info": module_info,
            "resource_stats": resource_stats,
            "total_modules_registered": len(self.module_registry),
            "base_modules": self.base_modules
        }
        
    def optimize_system(self) -> Dict[str, Any]:
        """
        Perform system-wide optimization
        
        Returns:
            Optimization results
        """
        start_time = time.time()
        
        # Get current state
        initial_stats = self.get_system_efficiency_stats()
        
        # Unload unused modules
        self.unload_unused_modules()
        
        # Trigger resource optimization
        memory_usage = self.resource_monitor._get_memory_usage()
        cpu_usage = self.resource_monitor._get_cpu_usage()
        self.resource_monitor._trigger_optimization(memory_usage, cpu_usage)
        
        # Get final state
        final_stats = self.get_system_efficiency_stats()
        
        optimization_time = time.time() - start_time
        
        return {
            "optimization_time": optimization_time,
            "initial_stats": initial_stats,
            "final_stats": final_stats,
            "modules_freed": initial_stats["module_info"]["total_modules"] - final_stats["module_info"]["total_modules"],
            "timestamp": time.time()
        }
        
    def export_efficiency_report(self, filepath: str) -> bool:
        """
        Export efficiency report to file
        
        Args:
            filepath: Path to export report
            
        Returns:
            True if successful, False otherwise
        """
        try:
            report = {
                "generated_at": datetime.now().isoformat(),
                "efficiency_stats": self.get_system_efficiency_stats(),
                "module_registry": list(self.module_registry.keys()),
                "base_modules": self.base_modules
            }
            
            import json
            with open(filepath, 'w') as f:
                json.dump(report, f, indent=2, default=str)
                
            print(f"✅ Efficiency report exported to {filepath}")
            return True
        except Exception as e:
            print(f"❌ Failed to export efficiency report: {e}")
            return False


# Example module factories for demonstration
def create_text_encoder():
    """Factory function for text encoder module"""
    if TORCH_AVAILABLE:
        return nn.Linear(768, 768)
    return None
    
def create_image_processor():
    """Factory function for image processor module"""
    if TORCH_AVAILABLE:
        return nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(64, 512)
        )
    return None
    
def create_audio_analyzer():
    """Factory function for audio analyzer module"""
    if TORCH_AVAILABLE:
        return nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128)
        )
    return None


def demo_efficiency_optimizer():
    """Demonstrate efficiency optimizer functionality"""
    print("🚀 Demonstrating Efficiency Optimizer...")
    print("=" * 50)
    
    # Create modular architecture
    mod_arch = ModularArchitecture(base_modules=["text_encoder"])
    print("✅ Created modular architecture")
    
    # Register modules
    print("\n🔧 Registering modules...")
    mod_arch.register_module("text_encoder", create_text_encoder)
    mod_arch.register_module("image_processor", create_image_processor)
    mod_arch.register_module("audio_analyzer", create_audio_analyzer)
    
    # Load modules dynamically
    print("\n🔄 Loading modules dynamically...")
    
    # Load text encoder (should be cached)
    text_module = mod_arch.get_module("text_encoder")
    print(f"   Text encoder loaded: {text_module is not None}")
    
    # Load image processor
    image_module = mod_arch.get_module("image_processor")
    print(f"   Image processor loaded: {image_module is not None}")
    
    # Load audio analyzer
    audio_module = mod_arch.get_module("audio_analyzer")
    print(f"   Audio analyzer loaded: {audio_module is not None}")
    
    # Show system stats
    print("\n📊 System Efficiency Statistics:")
    stats = mod_arch.get_system_efficiency_stats()
    print(f"   Modules loaded: {stats['module_info']['total_modules']}")
    print(f"   Dynamic loads: {stats['performance_stats']['dynamic_loads']}")
    print(f"   Cache hits: {stats['performance_stats']['cache_hits']}")
    print(f"   Memory usage: {stats['resource_stats']['current_memory_usage']:.1%}")
    print(f"   CPU usage: {stats['resource_stats']['current_cpu_usage']:.1%}")
    
    # Simulate some usage
    print("\n⚡ Simulating module usage...")
    for i in range(5):
        # Alternate between modules
        if i % 2 == 0:
            mod_arch.get_module("text_encoder")
        else:
            mod_arch.get_module("image_processor")
        time.sleep(0.1)
        
    # Optimize system
    print("\n⚙️  Optimizing system...")
    optimization_result = mod_arch.optimize_system()
    print(f"   Optimization completed in {optimization_result['optimization_time']:.3f}s")
    print(f"   Modules freed: {optimization_result['modules_freed']}")
    
    # Show final stats
    final_stats = mod_arch.get_system_efficiency_stats()
    print(f"   Final modules loaded: {final_stats['module_info']['total_modules']}")
    
    # Export report
    report_success = mod_arch.export_efficiency_report("efficiency_report.json")
    print(f"   Report export: {'SUCCESS' if report_success else 'FAILED'}")
    
    # Cleanup
    mod_arch.resource_monitor.stop_monitoring()
    
    print("\n" + "=" * 50)
    print("EFFICIENCY OPTIMIZER DEMO SUMMARY")
    print("=" * 50)
    print("Key Features Implemented:")
    print("  1. Dynamic module loading/unloading")
    print("  2. Resource monitoring and optimization")
    print("  3. Cache management and LRU eviction")
    print("  4. Performance statistics tracking")
    print("  5. System-wide optimization")
    print("\nBenefits:")
    print("  - Reduced memory footprint")
    print("  - Improved resource utilization")
    print("  - Dynamic scalability")
    print("  - Automated optimization")
    
    print("\n✅ Efficiency optimizer demonstration completed!")


if __name__ == "__main__":
    demo_efficiency_optimizer()