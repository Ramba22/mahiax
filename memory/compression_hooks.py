"""
Compression Hooks for Checkpoints in MAHIA-X
This module provides compression hooks for checkpoints using zstd-compressed state dicts.
"""

import time
import threading
import io
import pickle
from typing import Dict, Any, Optional, Callable, List
from collections import defaultdict

# Conditional imports with fallbacks
TRY_ZSTD = True
zstd = None

try:
    import zstd
except ImportError:
    TRY_ZSTD = False

class CompressionHooks:
    """Compression Hooks for checkpoint compression and decompression"""
    
    def __init__(self, compression_level: int = 3):
        self.compression_level = compression_level
        self.compression_stats = []
        self.lock = threading.Lock()
        
        if TRY_ZSTD and zstd is not None:
            print(f"⚙️  CompressionHooks initialized with zstd compression level {compression_level}")
        else:
            print("⚠️  CompressionHooks initialized without zstd (not available)")
        
    def compress_state_dict(self, state_dict: Dict[str, Any]) -> bytes:
        """
        Compress a state dictionary using zstd compression.
        
        Args:
            state_dict: Dictionary containing model state
            
        Returns:
            bytes: Compressed state dictionary
        """
        start_time = time.time()
        
        # Serialize the state dict
        serialized = pickle.dumps(state_dict)
        original_size = len(serialized)
        
        # Compress if zstd is available
        if TRY_ZSTD and zstd is not None:
            try:
                compressed = zstd.compress(serialized, self.compression_level)
                compression_ratio = original_size / len(compressed) if len(compressed) > 0 else 1.0
                compression_time = time.time() - start_time
                
                # Store stats
                with self.lock:
                    self.compression_stats.append({
                        "timestamp": start_time,
                        "operation": "compress",
                        "original_size": original_size,
                        "compressed_size": len(compressed),
                        "compression_ratio": compression_ratio,
                        "compression_time": compression_time,
                        "success": True
                    })
                
                print(f"🗜️  Compressed state dict: {original_size/(1024**2):.2f}MB -> {len(compressed)/(1024**2):.2f}MB "
                      f"(ratio: {compression_ratio:.2f}:1, time: {compression_time:.3f}s)")
                
                return compressed
            except Exception as e:
                print(f"❌ Compression failed: {e}")
                # Store stats for failed compression
                with self.lock:
                    self.compression_stats.append({
                        "timestamp": start_time,
                        "operation": "compress",
                        "original_size": original_size,
                        "compressed_size": original_size,
                        "compression_ratio": 1.0,
                        "compression_time": time.time() - start_time,
                        "success": False,
                        "error": str(e)
                    })
        
        # Return uncompressed if compression failed or not available
        compression_time = time.time() - start_time
        with self.lock:
            self.compression_stats.append({
                "timestamp": start_time,
                "operation": "compress",
                "original_size": original_size,
                "compressed_size": original_size,
                "compression_ratio": 1.0,
                "compression_time": compression_time,
                "success": False,
                "reason": "zstd not available" if not (TRY_ZSTD and zstd is not None) else "compression failed"
            })
        
        print(f"⚠️  State dict not compressed: {original_size/(1024**2):.2f}MB (time: {compression_time:.3f}s)")
        return serialized
    
    def decompress_state_dict(self, compressed_data: bytes) -> Dict[str, Any]:
        """
        Decompress a state dictionary.
        
        Args:
            compressed_data: Compressed state dictionary data
            
        Returns:
            Dict[str, Any]: Decompressed state dictionary
        """
        start_time = time.time()
        compressed_size = len(compressed_data)
        
        # Try to decompress if zstd is available
        if TRY_ZSTD and zstd is not None:
            try:
                # Try to decompress - if it fails, it might be uncompressed data
                try:
                    decompressed = zstd.decompress(compressed_data)
                    was_compressed = True
                except Exception:
                    # If decompression fails, assume it's uncompressed
                    decompressed = compressed_data
                    was_compressed = False
                
                # Deserialize
                state_dict = pickle.loads(decompressed)
                decompression_time = time.time() - start_time
                
                # Store stats
                with self.lock:
                    self.compression_stats.append({
                        "timestamp": start_time,
                        "operation": "decompress",
                        "compressed_size": compressed_size,
                        "decompressed_size": len(decompressed),
                        "decompression_time": decompression_time,
                        "success": True,
                        "was_compressed": was_compressed
                    })
                
                if was_compressed:
                    ratio = len(decompressed) / compressed_size if compressed_size > 0 else 1.0
                    print(f"🎉 Decompressed state dict: {compressed_size/(1024**2):.2f}MB -> {len(decompressed)/(1024**2):.2f}MB "
                          f"(ratio: {ratio:.2f}:1, time: {decompression_time:.3f}s)")
                else:
                    print(f"📦 Loaded uncompressed state dict: {len(decompressed)/(1024**2):.2f}MB "
                          f"(time: {decompression_time:.3f}s)")
                
                return state_dict
            except Exception as e:
                print(f"❌ Decompression failed: {e}")
                # Store stats for failed decompression
                with self.lock:
                    self.compression_stats.append({
                        "timestamp": start_time,
                        "operation": "decompress",
                        "compressed_size": compressed_size,
                        "decompression_time": time.time() - start_time,
                        "success": False,
                        "error": str(e)
                    })
                raise e
        
        # If zstd is not available, assume uncompressed data
        try:
            state_dict = pickle.loads(compressed_data)
            decompression_time = time.time() - start_time
            
            # Store stats
            with self.lock:
                self.compression_stats.append({
                    "timestamp": start_time,
                    "operation": "decompress",
                    "compressed_size": compressed_size,
                    "decompressed_size": len(compressed_data),
                    "decompression_time": decompression_time,
                    "success": True,
                    "was_compressed": False
                })
            
            print(f"📦 Loaded uncompressed state dict: {len(compressed_data)/(1024**2):.2f}MB "
                  f"(time: {decompression_time:.3f}s)")
            
            return state_dict
        except Exception as e:
            print(f"❌ Failed to load state dict: {e}")
            # Store stats for failed decompression
            with self.lock:
                self.compression_stats.append({
                    "timestamp": start_time,
                    "operation": "decompress",
                    "compressed_size": compressed_size,
                    "decompression_time": time.time() - start_time,
                    "success": False,
                    "error": str(e)
                })
            raise e
    
    def get_compression_stats(self) -> List[Dict[str, Any]]:
        """Get compression statistics"""
        with self.lock:
            return self.compression_stats.copy()
    
    def clear_stats(self):
        """Clear compression statistics"""
        with self.lock:
            self.compression_stats.clear()
        print("🗑️  Compression statistics cleared")
    
    def get_compression_summary(self) -> Dict[str, Any]:
        """Get a summary of compression statistics"""
        stats = self.get_compression_stats()
        if not stats:
            return {
                "total_operations": 0,
                "compression_operations": 0,
                "decompression_operations": 0,
                "successful_operations": 0,
                "failed_operations": 0,
                "avg_compression_ratio": 0.0,
                "avg_compression_time": 0.0,
                "avg_decompression_time": 0.0
            }
        
        compression_ops = [s for s in stats if s["operation"] == "compress"]
        decompression_ops = [s for s in stats if s["operation"] == "decompress"]
        successful_ops = [s for s in stats if s["success"]]
        failed_ops = [s for s in stats if not s["success"]]
        
        compression_ratios = [s["compression_ratio"] for s in compression_ops if s["success"] and "compression_ratio" in s]
        compression_times = [s["compression_time"] for s in compression_ops if s["success"] and "compression_time" in s]
        decompression_times = [s["decompression_time"] for s in decompression_ops if s["success"] and "decompression_time" in s]
        
        return {
            "total_operations": len(stats),
            "compression_operations": len(compression_ops),
            "decompression_operations": len(decompression_ops),
            "successful_operations": len(successful_ops),
            "failed_operations": len(failed_ops),
            "avg_compression_ratio": sum(compression_ratios) / len(compression_ratios) if compression_ratios else 0.0,
            "avg_compression_time": sum(compression_times) / len(compression_times) if compression_times else 0.0,
            "avg_decompression_time": sum(decompression_times) / len(decompression_times) if decompression_times else 0.0
        }
    
    def save_compressed_checkpoint(self, state_dict: Dict[str, Any], filepath: str) -> bool:
        """
        Save a compressed checkpoint to file.
        
        Args:
            state_dict: Dictionary containing model state
            filepath: Path to save the checkpoint
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            compressed_data = self.compress_state_dict(state_dict)
            with open(filepath, 'wb') as f:
                f.write(compressed_data)
            print(f"💾 Compressed checkpoint saved to {filepath}")
            return True
        except Exception as e:
            print(f"❌ Failed to save compressed checkpoint: {e}")
            return False
    
    def load_compressed_checkpoint(self, filepath: str) -> Dict[str, Any]:
        """
        Load a compressed checkpoint from file.
        
        Args:
            filepath: Path to the checkpoint file
            
        Returns:
            Dict[str, Any]: Loaded state dictionary
        """
        try:
            with open(filepath, 'rb') as f:
                compressed_data = f.read()
            state_dict = self.decompress_state_dict(compressed_data)
            print(f"📂 Compressed checkpoint loaded from {filepath}")
            return state_dict
        except Exception as e:
            print(f"❌ Failed to load compressed checkpoint: {e}")
            raise e
    
    def generate_report(self) -> str:
        """Generate a summary report of compression statistics"""
        summary = self.get_compression_summary()
        stats = self.get_compression_stats()
        
        if not stats:
            return "No compression operations recorded"
        
        report = f"""
⚙️  Compression Hooks Report
====================
Total Operations: {summary['total_operations']}
  Compression: {summary['compression_operations']}
  Decompression: {summary['decompression_operations']}
  Successful: {summary['successful_operations']}
  Failed: {summary['failed_operations']}

Performance Metrics:
  Average Compression Ratio: {summary['avg_compression_ratio']:.2f}:1
  Average Compression Time: {summary['avg_compression_time']:.3f}s
  Average Decompression Time: {summary['avg_decompression_time']:.3f}s

Recent Operations:
"""
        
        # Show last 5 entries
        recent_stats = stats[-5:] if len(stats) >= 5 else stats
        for i, stat in enumerate(recent_stats):
            op_type = "COMPRESS" if stat["operation"] == "compress" else "DECOMPRESS"
            status = "✅" if stat["success"] else "❌"
            report += f"  {i+1}. {time.ctime(stat['timestamp'])} - {status} {op_type}\n"
            
            if stat["operation"] == "compress" and stat["success"]:
                report += f"     Size: {stat['original_size']/(1024**2):.2f}MB -> {stat['compressed_size']/(1024**2):.2f}MB "
                report += f"(ratio: {stat['compression_ratio']:.2f}:1)\n"
                report += f"     Time: {stat['compression_time']:.3f}s\n"
            elif stat["operation"] == "decompress" and stat["success"]:
                report += f"     Size: {stat['compressed_size']/(1024**2):.2f}MB -> {stat['decompressed_size']/(1024**2):.2f}MB\n"
                report += f"     Time: {stat['decompression_time']:.3f}s\n"
                if stat.get("was_compressed", False):
                    report += f"     Was Compressed: Yes\n"
                else:
                    report += f"     Was Compressed: No\n"
        
        return report

# Global instance
_compression_hooks = None

def get_compression_hooks() -> CompressionHooks:
    """Get the global compression hooks instance"""
    global _compression_hooks
    if _compression_hooks is None:
        _compression_hooks = CompressionHooks()
    return _compression_hooks

if __name__ == "__main__":
    # Example usage
    compression_hooks = get_compression_hooks()
    
    # Simulate a state dict
    sample_state_dict = {
        "layer1.weight": [1.0, 2.0, 3.0],
        "layer1.bias": [0.1, 0.2],
        "layer2.weight": [[1.0, 2.0], [3.0, 4.0]],
        "layer2.bias": [0.05, 0.15]
    }
    
    # Compress and decompress
    compressed = compression_hooks.compress_state_dict(sample_state_dict)
    decompressed = compression_hooks.decompress_state_dict(compressed)
    
    # Verify they're the same
    print(f"Original == Decompressed: {sample_state_dict == decompressed}")
    
    # Print report
    print(compression_hooks.generate_report())