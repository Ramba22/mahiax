"""
Memory Allocator for MAHIA OptiCore
Dynamic memory management with real-time monitoring and Torch CUDA Graphs support.
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

class MemoryAllocator:
    """Dynamic memory management with real-time monitoring"""
    
    def __init__(self, monitoring_interval: float = 1.0):
        self.monitoring_interval = monitoring_interval
        self.is_monitoring = False
        self.monitoring_thread = None
        self.memory_blocks = {}  # track allocated blocks
        self.free_blocks = defaultdict(list)  # track free blocks by size
        self.stats = {
            "allocated_bytes": 0,
            "free_bytes": 0,
            "peak_allocated": 0,
            "allocations": 0,
            "deallocations": 0
        }
        self.lock = threading.Lock()
        
        print(f"💾 MemoryAllocator initialized with monitoring interval: {monitoring_interval}s")
        
    def allocate(self, size: int, device: str = "cuda") -> Any:
        """
        Allocate memory block.
        
        Args:
            size: Size of memory block in bytes
            device: Target device ("cuda" or "cpu")
            
        Returns:
            Allocated tensor or memory block
        """
        with self.lock:
            # Try to find a suitable free block
            suitable_block = None
            suitable_size = None
            
            for block_size, blocks in self.free_blocks.items():
                if block_size >= size and blocks:
                    suitable_block = blocks.pop()
                    suitable_size = block_size
                    break
            
            if suitable_block is not None:
                # Reuse existing block
                print(f"🔄 Reusing memory block of size {suitable_size} for request of {size}")
                self.stats["free_bytes"] -= suitable_size if suitable_size is not None else 0
                self.stats["allocated_bytes"] += size
                
                # Track the allocated block
                block_id = id(suitable_block)
                self.memory_blocks[block_id] = {
                    "size": size,
                    "actual_size": suitable_size,
                    "device": device,
                    "timestamp": time.time()
                }
                
                # Update peak allocation
                if self.stats["allocated_bytes"] > self.stats["peak_allocated"]:
                    self.stats["peak_allocated"] = self.stats["allocated_bytes"]
                    
                return suitable_block
            else:
                # Allocate new block
                if TORCH_AVAILABLE and torch is not None:
                    try:
                        if device == "cuda" and torch.cuda.is_available():
                            tensor = torch.empty(size // 4, dtype=torch.float32, device="cuda")  # 4 bytes per float32
                        else:
                            tensor = torch.empty(size // 4, dtype=torch.float32, device="cpu")
                        
                        self.stats["allocated_bytes"] += size
                        self.stats["allocations"] += 1
                        
                        # Track the allocated block
                        block_id = id(tensor)
                        self.memory_blocks[block_id] = {
                            "size": size,
                            "actual_size": size,
                            "device": device,
                            "timestamp": time.time()
                        }
                        
                        # Update peak allocation
                        if self.stats["allocated_bytes"] > self.stats["peak_allocated"]:
                            self.stats["peak_allocated"] = self.stats["allocated_bytes"]
                            
                        print(f"🆕 Allocated new memory block of size {size}")
                        return tensor
                    except Exception as e:
                        print(f"❌ Memory allocation failed: {e}")
                        return None
                else:
                    # Fallback for when PyTorch is not available
                    try:
                        import numpy as np
                        array = np.empty(size // 4, dtype=np.float32)
                        self.stats["allocated_bytes"] += size
                        self.stats["allocations"] += 1
                        
                        # Track the allocated block
                        block_id = id(array)
                        self.memory_blocks[block_id] = {
                            "size": size,
                            "actual_size": size,
                            "device": "cpu",
                            "timestamp": time.time()
                        }
                        
                        # Update peak allocation
                        if self.stats["allocated_bytes"] > self.stats["peak_allocated"]:
                            self.stats["peak_allocated"] = self.stats["allocated_bytes"]
                            
                        print(f"🆕 Allocated new memory block of size {size}")
                        return array
                    except Exception as e:
                        print(f"❌ Memory allocation failed: {e}")
                        return None
    
    def deallocate(self, block: Any) -> bool:
        """
        Deallocate memory block.
        
        Args:
            block: Memory block to deallocate
            
        Returns:
            bool: True if successful, False otherwise
        """
        block_id = id(block)
        
        with self.lock:
            if block_id in self.memory_blocks:
                block_info = self.memory_blocks.pop(block_id)
                size = block_info["actual_size"]
                device = block_info["device"]
                
                # Add to free blocks
                self.free_blocks[size].append(block)
                self.stats["free_bytes"] += size
                self.stats["allocated_bytes"] -= block_info["size"]
                self.stats["deallocations"] += 1
                
                print(f"🆓 Deallocated memory block of size {size}")
                return True
            else:
                print("⚠️  Attempted to deallocate unknown memory block")
                return False
    
    def start_monitoring(self):
        """Start real-time memory monitoring"""
        if self.is_monitoring:
            print("⚠️  Memory monitoring is already running")
            return
            
        self.is_monitoring = True
        self.monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitoring_thread.start()
        print("📈 Started memory monitoring")
        
    def stop_monitoring(self):
        """Stop real-time memory monitoring"""
        self.is_monitoring = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=2.0)
        print("⏹️  Stopped memory monitoring")
        
    def _monitoring_loop(self):
        """Background monitoring loop"""
        while self.is_monitoring:
            self._collect_memory_stats()
            time.sleep(self.monitoring_interval)
            
    def _collect_memory_stats(self):
        """Collect memory statistics"""
        if TORCH_AVAILABLE and torch is not None and torch.cuda.is_available():
            try:
                device = torch.cuda.current_device()
                allocated = torch.cuda.memory_allocated(device)
                reserved = torch.cuda.memory_reserved(device)
                free = reserved - allocated
                
                print(f"📊 GPU Memory - Allocated: {allocated/(1024**3):.2f}GB, "
                      f"Reserved: {reserved/(1024**3):.2f}GB, "
                      f"Free: {free/(1024**3):.2f}GB")
            except Exception as e:
                print(f"❌ Error collecting GPU memory stats: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get memory allocator statistics"""
        with self.lock:
            return self.stats.copy()
            
    def clear_stats(self):
        """Clear statistics"""
        with self.lock:
            self.stats = {
                "allocated_bytes": 0,
                "free_bytes": 0,
                "peak_allocated": 0,
                "allocations": 0,
                "deallocations": 0
            }
        print("🗑️  Memory allocator statistics cleared")
        
    def defragment(self):
        """Perform memory defragmentation"""
        with self.lock:
            # Simple defragmentation: clear free blocks
            freed_memory = sum(sum(block_info["actual_size"] for block_info in 
                                 [{"actual_size": size} for size in self.free_blocks[size_list]]) 
                              for size_list in self.free_blocks)
            
            self.free_blocks.clear()
            self.stats["free_bytes"] = 0
            
            print(f"🧹 Memory defragmentation completed. Freed {freed_memory} bytes")

# Global instance
_memory_allocator = None

def get_memory_allocator() -> MemoryAllocator:
    """Get the global memory allocator instance"""
    global _memory_allocator
    if _memory_allocator is None:
        _memory_allocator = MemoryAllocator()
    return _memory_allocator

if __name__ == "__main__":
    # Example usage
    allocator = get_memory_allocator()
    
    # Allocate some memory
    block1 = allocator.allocate(1024 * 1024)  # 1MB
    block2 = allocator.allocate(2 * 1024 * 1024)  # 2MB
    
    # Print stats
    print(f"📊 Stats: {allocator.get_stats()}")
    
    # Deallocate
    allocator.deallocate(block1)
    allocator.deallocate(block2)
    
    # Print stats again
    print(f"📊 Stats: {allocator.get_stats()}")