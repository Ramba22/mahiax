"""
Memory Fragmentation Test for MAHIA-X
This module monitors CUDA allocation/free patterns to detect memory fragmentation.
"""

import time
import threading
from typing import Dict, Any, Optional, List, Tuple
from collections import defaultdict, deque

# Conditional imports with fallbacks
TORCH_AVAILABLE = False
torch = None

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    pass

class MemoryFragmentationTest:
    """Memory Fragmentation Test for monitoring CUDA allocation patterns"""
    
    def __init__(self, window_size: int = 1000, fragmentation_threshold: float = 0.3, 
                 cache_flush_threshold: float = 0.9):
        self.window_size = window_size
        self.fragmentation_threshold = fragmentation_threshold
        self.cache_flush_threshold = cache_flush_threshold  # 90% fragmentation threshold for cache flush
        self.allocation_events = deque(maxlen=window_size)
        self.fragmentation_history = deque(maxlen=window_size)
        self.is_monitoring = False
        self.monitoring_thread = None
        self.lock = threading.Lock()
        self.flush_count = 0
        self.last_flush_time = 0
        
        print(f"🧱 MemoryFragmentationTest initialized:")
        print(f"   Window Size: {window_size}")
        print(f"   Fragmentation Threshold: {fragmentation_threshold*100:.1f}%")
        
    def record_allocation_event(self, allocated_bytes: int, reserved_bytes: int, 
                              free_bytes: int, timestamp: Optional[float] = None) -> Dict[str, Any]:
        """
        Record a memory allocation event.
        
        Args:
            allocated_bytes: Currently allocated memory in bytes
            reserved_bytes: Total reserved memory in bytes
            free_bytes: Free memory in the reserved block in bytes
            timestamp: Optional timestamp (defaults to current time)
            
        Returns:
            Dict containing fragmentation metrics
        """
        if timestamp is None:
            timestamp = time.time()
            
        # Calculate fragmentation metrics
        total_reserved = reserved_bytes
        total_allocated = allocated_bytes
        total_free = free_bytes
        
        # Fragmentation ratio: free memory that cannot be used due to fragmentation
        # This is calculated as the ratio of free memory to total reserved memory
        fragmentation_ratio = total_free / total_reserved if total_reserved > 0 else 0.0
        
        # Usable memory ratio: allocated memory to total reserved memory
        usable_ratio = total_allocated / total_reserved if total_reserved > 0 else 0.0
        
        # Wasted memory due to fragmentation
        wasted_bytes = total_free  # All free memory is potentially wasted due to fragmentation
        
        allocation_event = {
            "timestamp": timestamp,
            "allocated_bytes": allocated_bytes,
            "reserved_bytes": reserved_bytes,
            "free_bytes": free_bytes,
            "fragmentation_ratio": fragmentation_ratio,
            "usable_ratio": usable_ratio,
            "wasted_bytes": wasted_bytes,
            "is_fragmented": fragmentation_ratio > self.fragmentation_threshold
        }
        
        # Store event
        with self.lock:
            self.allocation_events.append(allocation_event)
            
            if allocation_event["is_fragmented"]:
                self.fragmentation_history.append(allocation_event)
        
        # Print warning if fragmentation is high
        if allocation_event["is_fragmented"]:
            print(f"⚠️  High Memory Fragmentation Detected: {fragmentation_ratio*100:.1f}% "
                  f"(Threshold: {self.fragmentation_threshold*100:.1f}%)")
            print(f"   Allocated: {allocated_bytes/(1024**3):.2f}GB | "
                  f"Reserved: {reserved_bytes/(1024**3):.2f}GB | "
                  f"Free: {free_bytes/(1024**3):.2f}GB")
        
        # Check if cache flush is needed
        if fragmentation_ratio > self.cache_flush_threshold:
            self._flush_cache()
            
        # Check if we should flush based on reserved memory usage
        reserved_utilization = total_allocated / total_reserved if total_reserved > 0 else 0
        if reserved_utilization < 0.1 and total_reserved > 1024**3:  # 1GB minimum
            self._flush_cache()
        
        return allocation_event
    
    def get_torch_memory_stats(self) -> Tuple[int, int, int]:
        """
        Get PyTorch CUDA memory statistics.
        
        Returns:
            Tuple of (allocated_bytes, reserved_bytes, free_bytes)
        """
        if not TORCH_AVAILABLE or torch is None or not torch.cuda.is_available():
            return 0, 0, 0
            
        try:
            # Get current device
            device = torch.cuda.current_device()
            
            # Get memory stats
            allocated_bytes = torch.cuda.memory_allocated(device)
            reserved_bytes = torch.cuda.memory_reserved(device)
            free_bytes = reserved_bytes - allocated_bytes
            
            return allocated_bytes, reserved_bytes, free_bytes
        except Exception:
            return 0, 0, 0
    
    def _flush_cache(self):
        """Flush CUDA cache when fragmentation is high"""
        if not TORCH_AVAILABLE or torch is None or not torch.cuda.is_available():
            return
            
        try:
            current_time = time.time()
            # Don't flush too frequently (at least 5 seconds between flushes)
            if current_time - self.last_flush_time < 5.0:
                return
                
            # Flush CUDA cache
            torch.cuda.empty_cache()
            self.flush_count += 1
            self.last_flush_time = current_time
            
            print(f"🔄 Automatic Cache Flush #{self.flush_count} - Fragmentation reduced")
        except Exception as e:
            print(f"⚠️  Cache flush failed: {e}")
    
    def check_current_fragmentation(self) -> Dict[str, Any]:
        """Check current memory fragmentation"""
        allocated, reserved, free = self.get_torch_memory_stats()
        return self.record_allocation_event(allocated, reserved, free)
    
    def start_auto_monitoring(self, interval_seconds: int = 2):
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
            self.check_current_fragmentation()
            time.sleep(interval_seconds)
    
    def get_fragmentation_history(self) -> List[Dict[str, Any]]:
        """Get history of fragmented memory events"""
        with self.lock:
            return list(self.fragmentation_history)
    
    def get_allocation_events(self) -> List[Dict[str, Any]]:
        """Get all allocation events"""
        with self.lock:
            return list(self.allocation_events)
    
    def clear_history(self):
        """Clear all history"""
        with self.lock:
            self.allocation_events.clear()
            self.fragmentation_history.clear()
        print("🗑️  Memory fragmentation history cleared")
    
    def get_fragmentation_statistics(self) -> Dict[str, Any]:
        """Calculate fragmentation statistics"""
        events = self.get_allocation_events()
        if not events:
            return {
                "total_events": 0,
                "fragmented_events": 0,
                "fragmentation_rate": 0.0,
                "avg_fragmentation_ratio": 0.0,
                "max_fragmentation_ratio": 0.0,
                "min_fragmentation_ratio": 0.0,
                "cache_flushes": self.flush_count
            }
        
        fragmented_events = [e for e in events if e["is_fragmented"]]
        fragmentation_ratios = [e["fragmentation_ratio"] for e in events]
        
        return {
            "total_events": len(events),
            "fragmented_events": len(fragmented_events),
            "fragmentation_rate": len(fragmented_events) / len(events) if events else 0.0,
            "avg_fragmentation_ratio": sum(fragmentation_ratios) / len(fragmentation_ratios) if fragmentation_ratios else 0.0,
            "max_fragmentation_ratio": max(fragmentation_ratios) if fragmentation_ratios else 0.0,
            "min_fragmentation_ratio": min(fragmentation_ratios) if fragmentation_ratios else 0.0,
            "cache_flushes": self.flush_count
        }
    
    def suggest_optimizations(self) -> List[str]:
        """Suggest optimizations based on fragmentation patterns"""
        stats = self.get_fragmentation_statistics()
        suggestions = []
        
        if stats["fragmentation_rate"] > 0.5:
            suggestions.append("🔧 High fragmentation rate detected. Consider using memory pooling.")
        elif stats["fragmentation_rate"] > 0.3:
            suggestions.append("⚠️  Moderate fragmentation rate. Monitor memory allocation patterns.")
            
        if stats["avg_fragmentation_ratio"] > 0.4:
            suggestions.append("🔧 Average fragmentation ratio is high. Consider batch size optimization.")
        elif stats["avg_fragmentation_ratio"] > 0.2:
            suggestions.append("⚠️  Average fragmentation ratio is moderate. Consider memory defragmentation.")
            
        if stats["max_fragmentation_ratio"] > 0.7:
            suggestions.append("🚨 Peak fragmentation is very high. Immediate optimization needed.")
            
        if not suggestions:
            suggestions.append("✅ Memory fragmentation levels are within acceptable ranges.")
            
        return suggestions
    
    def generate_report(self) -> str:
        """Generate a summary report of memory fragmentation"""
        stats = self.get_fragmentation_statistics()
        events = self.get_allocation_events()
        
        if not events:
            return "No memory allocation events recorded"
        
        report = f"""
🧱 Memory Fragmentation Report
=======================
Total Events: {stats['total_events']}
Fragmented Events: {stats['fragmented_events']}
Fragmentation Rate: {stats['fragmentation_rate']*100:.1f}%
Average Fragmentation: {stats['avg_fragmentation_ratio']*100:.1f}%
Max Fragmentation: {stats['max_fragmentation_ratio']*100:.1f}%
Min Fragmentation: {stats['min_fragmentation_ratio']*100:.1f}%

Memory Usage Summary:
"""
        
        # Show last 5 entries
        recent_events = events[-5:] if len(events) >= 5 else events
        for i, event in enumerate(recent_events):
            status = "FRAGMENTED" if event["is_fragmented"] else "NORMAL"
            report += f"  {i+1}. {time.ctime(event['timestamp'])} - {status}\n"
            report += f"     Allocated: {event['allocated_bytes']/(1024**3):.2f}GB | "
            report += f"Reserved: {event['reserved_bytes']/(1024**3):.2f}GB | "
            report += f"Free: {event['free_bytes']/(1024**3):.2f}GB\n"
            report += f"     Fragmentation: {event['fragmentation_ratio']*100:.1f}%\n"
        
        # Add suggestions
        suggestions = self.suggest_optimizations()
        report += f"\n💡 Optimization Suggestions:\n"
        for suggestion in suggestions:
            report += f"  {suggestion}\n"
        
        return report

# Global instance
_fragmentation_test = None

def get_fragmentation_test() -> MemoryFragmentationTest:
    """Get the global memory fragmentation test instance"""
    global _fragmentation_test
    if _fragmentation_test is None:
        _fragmentation_test = MemoryFragmentationTest()
    return _fragmentation_test

if __name__ == "__main__":
    # Example usage
    fragmentation_test = get_fragmentation_test()
    
    # Simulate some memory allocation events
    fragmentation_test.record_allocation_event(
        allocated_bytes=int(8 * 1024**3),  # 8GB allocated
        reserved_bytes=int(12 * 1024**3),  # 12GB reserved
        free_bytes=int(4 * 1024**3)        # 4GB free
    )
    
    fragmentation_test.record_allocation_event(
        allocated_bytes=int(10 * 1024**3), # 10GB allocated
        reserved_bytes=int(16 * 1024**3),  # 16GB reserved
        free_bytes=int(6 * 1024**3)        # 6GB free
    )
    
    # Print report
    print(fragmentation_test.generate_report())