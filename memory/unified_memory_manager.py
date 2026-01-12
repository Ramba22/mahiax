"""
Unified Memory Manager for MAHIA-X
This module provides consistent tracking and management of VRAM, CPU-RAM, and activation memory.
"""

import os
import threading
import time
from typing import Dict, Any, Optional, List, Tuple
from collections import defaultdict

# Conditional imports with fallbacks
TORCH_AVAILABLE = False
PSUTIL_AVAILABLE = False
torch = None
psutil = None

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    pass

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    pass

class MemoryInfo:
    """Container for memory information"""
    
    def __init__(self, vram_used: float = 0.0, vram_total: float = 0.0, 
                 ram_used: float = 0.0, ram_total: float = 0.0,
                 activation_memory: float = 0.0):
        self.vram_used = vram_used
        self.vram_total = vram_total
        self.vram_percent = (vram_used / vram_total * 100) if vram_total > 0 else 0.0
        self.ram_used = ram_used
        self.ram_total = ram_total
        self.ram_percent = (ram_used / ram_total * 100) if ram_total > 0 else 0.0
        self.activation_memory = activation_memory
        self.timestamp = time.time()
    
    def __str__(self):
        return (f"MemoryInfo(VRAM: {self.vram_used:.2f}/{self.vram_total:.2f}GB ({self.vram_percent:.1f}%), "
                f"RAM: {self.ram_used:.2f}/{self.ram_total:.2f}GB ({self.ram_percent:.1f}%), "
                f"Activation: {self.activation_memory:.2f}GB)")

class UnifiedMemoryManager:
    """Unified Memory Manager for consistent tracking of VRAM, CPU-RAM, and activation memory"""
    
    def __init__(self, log_interval: int = 10):
        self.log_interval = log_interval
        self.is_monitoring = False
        self.monitoring_thread = None
        self.memory_logs = []
        self.peak_memory = MemoryInfo()
        self.current_memory = MemoryInfo()
        self.lock = threading.Lock()
        
        # Try to get total system memory
        self.system_ram_total = 0.0
        if PSUTIL_AVAILABLE and psutil is not None:
            try:
                mem = psutil.virtual_memory()
                self.system_ram_total = mem.total / (1024**3)  # Convert to GB
            except Exception:
                pass
            
        print("🧠 UnifiedMemoryManager initialized")
        
    def get_torch_memory_info(self) -> Tuple[float, float]:
        """Get PyTorch memory information (VRAM used, VRAM total) in GB"""
        if not TORCH_AVAILABLE or torch is None:
            return 0.0, 0.0
            
        try:
            if torch.cuda.is_available():
                device = torch.cuda.current_device()
                vram_used = torch.cuda.memory_allocated(device) / (1024**3)  # Convert to GB
                vram_total = torch.cuda.get_device_properties(device).total_memory / (1024**3)
                return vram_used, vram_total
            else:
                return 0.0, 0.0
        except Exception:
            return 0.0, 0.0
    
    def get_system_memory_info(self) -> Tuple[float, float]:
        """Get system memory information (RAM used, RAM total) in GB"""
        if not PSUTIL_AVAILABLE or psutil is None:
            return 0.0, self.system_ram_total
            
        try:
            mem = psutil.virtual_memory()
            ram_used = mem.used / (1024**3)  # Convert to GB
            ram_total = mem.total / (1024**3)
            return ram_used, ram_total
        except Exception:
            return 0.0, self.system_ram_total
    
    def update_memory_info(self):
        """Update current memory information"""
        vram_used, vram_total = self.get_torch_memory_info()
        ram_used, ram_total = self.get_system_memory_info()
        
        with self.lock:
            self.current_memory = MemoryInfo(
                vram_used=vram_used,
                vram_total=vram_total,
                ram_used=ram_used,
                ram_total=ram_total,
                activation_memory=0.0  # This would be updated by the model
            )
            
            # Update peak memory if current is higher
            if self.current_memory.vram_percent > self.peak_memory.vram_percent:
                self.peak_memory = MemoryInfo(
                    vram_used=self.current_memory.vram_used,
                    vram_total=self.current_memory.vram_total,
                    ram_used=self.current_memory.ram_used,
                    ram_total=self.current_memory.ram_total,
                    activation_memory=self.current_memory.activation_memory
                )
    
    def log_memory_info(self):
        """Log current memory information"""
        self.update_memory_info()
        
        with self.lock:
            self.memory_logs.append(self.current_memory)
            
            # Keep only last 1000 entries to prevent memory bloat
            if len(self.memory_logs) > 1000:
                self.memory_logs = self.memory_logs[-1000:]
            
            print(f"💾 Memory Status: {self.current_memory}")
    
    def start_monitoring(self):
        """Start periodic memory monitoring in a background thread"""
        if self.is_monitoring:
            print("⚠️  Memory monitoring is already running")
            return
            
        self.is_monitoring = True
        self.monitoring_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitoring_thread.start()
        print("📈 Started memory monitoring")
    
    def stop_monitoring(self):
        """Stop periodic memory monitoring"""
        self.is_monitoring = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=2.0)
        print("⏹️  Stopped memory monitoring")
    
    def _monitor_loop(self):
        """Background monitoring loop"""
        while self.is_monitoring:
            self.log_memory_info()
            time.sleep(self.log_interval)
    
    def get_current_memory(self) -> MemoryInfo:
        """Get current memory information"""
        with self.lock:
            return self.current_memory
    
    def get_peak_memory(self) -> MemoryInfo:
        """Get peak memory information"""
        with self.lock:
            return self.peak_memory
    
    def get_memory_logs(self) -> List[MemoryInfo]:
        """Get memory logs"""
        with self.lock:
            return self.memory_logs.copy()
    
    def reset_peak_memory(self):
        """Reset peak memory tracking"""
        with self.lock:
            self.peak_memory = MemoryInfo()
        print("🔄 Peak memory reset")
    
    def set_activation_memory(self, activation_memory_gb: float):
        """Set activation memory usage"""
        with self.lock:
            self.current_memory.activation_memory = activation_memory_gb
            if activation_memory_gb > self.peak_memory.activation_memory:
                self.peak_memory.activation_memory = activation_memory_gb
    
    def get_memory_summary(self) -> Dict[str, Any]:
        """Get a summary of memory usage"""
        current = self.get_current_memory()
        peak = self.get_peak_memory()
        
        return {
            "current": {
                "vram_used_gb": current.vram_used,
                "vram_total_gb": current.vram_total,
                "vram_percent": current.vram_percent,
                "ram_used_gb": current.ram_used,
                "ram_total_gb": current.ram_total,
                "ram_percent": current.ram_percent,
                "activation_memory_gb": current.activation_memory,
                "timestamp": current.timestamp
            },
            "peak": {
                "vram_used_gb": peak.vram_used,
                "vram_total_gb": peak.vram_total,
                "vram_percent": peak.vram_percent,
                "ram_used_gb": peak.ram_used,
                "ram_total_gb": peak.ram_total,
                "ram_percent": peak.ram_percent,
                "activation_memory_gb": peak.activation_memory,
                "timestamp": peak.timestamp
            },
            "log_count": len(self.memory_logs)
        }

# Global instance
_memory_manager = None

def get_memory_manager() -> UnifiedMemoryManager:
    """Get the global memory manager instance"""
    global _memory_manager
    if _memory_manager is None:
        _memory_manager = UnifiedMemoryManager()
    return _memory_manager

if __name__ == "__main__":
    # Example usage
    manager = get_memory_manager()
    manager.start_monitoring()
    
    # Simulate some work
    time.sleep(5)
    
    # Print summary
    summary = manager.get_memory_summary()
    print("\n📊 Memory Summary:")
    print(f"Current VRAM: {summary['current']['vram_used_gb']:.2f}/{summary['current']['vram_total_gb']:.2f}GB ({summary['current']['vram_percent']:.1f}%)")
    print(f"Peak VRAM: {summary['peak']['vram_used_gb']:.2f}/{summary['peak']['vram_total_gb']:.2f}GB ({summary['peak']['vram_percent']:.1f}%)")
    
    manager.stop_monitoring()