"""
Pooling Engine for MAHIA OptiCore
Shared memory pools with hash-based buffer matching for MoE and Attention blocks.
"""

import time
import threading
import hashlib
from typing import Dict, Any, Optional, List, Tuple
from collections import defaultdict

class PoolingEngine:
    """Shared memory pools with hash-based buffer matching"""
    
    def __init__(self, max_pool_size: int = 100):
        self.max_pool_size = max_pool_size
        self.pools = defaultdict(list)  # pools by buffer type
        self.buffer_metadata = {}  # metadata for each buffer
        self.lock = threading.Lock()
        self.stats = {
            "buffers_created": 0,
            "buffers_reused": 0,
            "pool_overflows": 0
        }
        
        print(f"🧩 PoolingEngine initialized with max pool size: {max_pool_size}")
        
    def _generate_buffer_hash(self, shape: Tuple[int, ...], dtype: str, device: str) -> str:
        """
        Generate a hash for buffer identification.
        
        Args:
            shape: Buffer shape
            dtype: Buffer data type
            device: Target device
            
        Returns:
            str: Hash string for buffer identification
        """
        hash_input = f"{shape}_{dtype}_{device}"
        return hashlib.md5(hash_input.encode()).hexdigest()
        
    def get_buffer(self, shape: Tuple[int, ...], dtype: str = "float32", 
                   device: str = "cuda") -> Any:
        """
        Get a buffer from the pool or create a new one.
        
        Args:
            shape: Buffer shape
            dtype: Buffer data type
            device: Target device
            
        Returns:
            Buffer object
        """
        buffer_hash = self._generate_buffer_hash(shape, dtype, device)
        
        with self.lock:
            # Try to get buffer from pool
            if buffer_hash in self.pools and self.pools[buffer_hash]:
                buffer = self.pools[buffer_hash].pop()
                self.stats["buffers_reused"] += 1
                
                print(f"🔄 Reusing buffer from pool: {buffer_hash[:8]}...")
                return buffer
            
            # Create new buffer
            buffer = self._create_buffer(shape, dtype, device)
            if buffer is not None:
                self.stats["buffers_created"] += 1
                # Store metadata
                buffer_id = id(buffer)
                self.buffer_metadata[buffer_id] = {
                    "hash": buffer_hash,
                    "shape": shape,
                    "dtype": dtype,
                    "device": device,
                    "timestamp": time.time()
                }
                print(f"🆕 Created new buffer: {buffer_hash[:8]}...")
                return buffer
            else:
                print("❌ Failed to create buffer")
                return None
                
    def _create_buffer(self, shape: Tuple[int, ...], dtype: str, device: str) -> Any:
        """
        Create a new buffer.
        
        Args:
            shape: Buffer shape
            dtype: Buffer data type
            device: Target device
            
        Returns:
            Buffer object or None if failed
        """
        # Conditional imports
        try:
            import torch
            import numpy as np
            
            # Create buffer based on device
            if device == "cuda":
                if torch.cuda.is_available():
                    if dtype == "float32":
                        return torch.empty(shape, dtype=torch.float32, device="cuda")
                    elif dtype == "float16":
                        return torch.empty(shape, dtype=torch.float16, device="cuda")
                    elif dtype == "int32":
                        return torch.empty(shape, dtype=torch.int32, device="cuda")
                    else:
                        return torch.empty(shape, dtype=torch.float32, device="cuda")
                else:
                    # Fallback to CPU
                    if dtype == "float32":
                        return torch.empty(shape, dtype=torch.float32, device="cpu")
                    elif dtype == "float16":
                        return torch.empty(shape, dtype=torch.float16, device="cpu")
                    elif dtype == "int32":
                        return torch.empty(shape, dtype=torch.int32, device="cpu")
                    else:
                        return torch.empty(shape, dtype=torch.float32, device="cpu")
            else:
                # CPU buffer
                if dtype == "float32":
                    return np.empty(shape, dtype=np.float32)
                elif dtype == "float16":
                    return np.empty(shape, dtype=np.float16)
                elif dtype == "int32":
                    return np.empty(shape, dtype=np.int32)
                else:
                    return np.empty(shape, dtype=np.float32)
        except ImportError:
            try:
                import numpy as np
                if dtype == "float32":
                    return np.empty(shape, dtype=np.float32)
                elif dtype == "float16":
                    return np.empty(shape, dtype=np.float16)
                elif dtype == "int32":
                    return np.empty(shape, dtype=np.int32)
                else:
                    return np.empty(shape, dtype=np.float32)
            except ImportError:
                print("❌ No available libraries for buffer creation")
                return None
                
    def return_buffer(self, buffer: Any) -> bool:
        """
        Return a buffer to the pool.
        
        Args:
            buffer: Buffer to return
            
        Returns:
            bool: True if successful, False otherwise
        """
        buffer_id = id(buffer)
        
        with self.lock:
            if buffer_id in self.buffer_metadata:
                metadata = self.buffer_metadata[buffer_id]
                buffer_hash = metadata["hash"]
                
                # Check if pool is full
                if len(self.pools[buffer_hash]) < self.max_pool_size:
                    self.pools[buffer_hash].append(buffer)
                    print(f"📥 Returned buffer to pool: {buffer_hash[:8]}...")
                    return True
                else:
                    self.stats["pool_overflows"] += 1
                    print(f"⚠️  Pool overflow for buffer: {buffer_hash[:8]}...")
                    # Buffer will be garbage collected
                    return False
            else:
                print("⚠️  Attempted to return unknown buffer")
                return False
                
    def get_stats(self) -> Dict[str, Any]:
        """Get pooling engine statistics"""
        with self.lock:
            pool_stats = {hash_key: len(buffers) for hash_key, buffers in self.pools.items()}
            return {
                **self.stats,
                "pools": pool_stats,
                "total_pooled_buffers": sum(len(buffers) for buffers in self.pools.values())
            }
            
    def clear_stats(self):
        """Clear statistics"""
        with self.lock:
            self.stats = {
                "buffers_created": 0,
                "buffers_reused": 0,
                "pool_overflows": 0
            }
        print("🗑️  Pooling engine statistics cleared")
        
    def clear_pools(self):
        """Clear all pools"""
        with self.lock:
            self.pools.clear()
            self.buffer_metadata.clear()
        print("🧨 All pools cleared")

# Global instance
_pooling_engine = None

def get_pooling_engine() -> PoolingEngine:
    """Get the global pooling engine instance"""
    global _pooling_engine
    if _pooling_engine is None:
        _pooling_engine = PoolingEngine()
    return _pooling_engine

if __name__ == "__main__":
    # Example usage
    engine = get_pooling_engine()
    
    # Get some buffers
    buffer1 = engine.get_buffer((10, 10), "float32", "cpu")
    buffer2 = engine.get_buffer((10, 10), "float32", "cpu")
    
    # Return buffers to pool
    engine.return_buffer(buffer1)
    engine.return_buffer(buffer2)
    
    # Get buffer again (should reuse)
    buffer3 = engine.get_buffer((10, 10), "float32", "cpu")
    
    # Print stats
    print(f"📊 Stats: {engine.get_stats()}")