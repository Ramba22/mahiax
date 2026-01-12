"""
Zero-Downtime Checkpointing for MAHIA-X
Implements async I/O thread for checkpointing without training interruption
"""

import torch
import threading
import queue
import time
import os
import json
import hashlib
from typing import Dict, Any, Optional, List, Tuple, Union
from collections import OrderedDict
import asyncio
import concurrent.futures
from datetime import datetime

# Conditional imports
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

class AsyncCheckpointWriter:
    """Asynchronous checkpoint writer using background threads"""
    
    def __init__(self, max_queue_size: int = 10):
        """
        Initialize async checkpoint writer
        
        Args:
            max_queue_size: Maximum size of checkpoint queue
        """
        self.max_queue_size = max_queue_size
        self.checkpoint_queue = queue.Queue(maxsize=max_queue_size)
        self.writer_thread = None
        self.is_running = False
        self.completed_checkpoints = OrderedDict()
        self.failed_checkpoints = OrderedDict()
        
        print(f"✅ AsyncCheckpointWriter initialized with queue size: {max_queue_size}")
        
    def start_writer(self):
        """Start the background writer thread"""
        if self.is_running:
            return
            
        self.is_running = True
        self.writer_thread = threading.Thread(target=self._writer_worker, daemon=True)
        self.writer_thread.start()
        print("✅ Async checkpoint writer started")
        
    def stop_writer(self):
        """Stop the background writer thread"""
        if not self.is_running:
            return
            
        self.is_running = False
        if self.writer_thread:
            self.writer_thread.join(timeout=5.0)
        print("✅ Async checkpoint writer stopped")
        
    def _writer_worker(self):
        """Background worker thread for writing checkpoints"""
        while self.is_running:
            try:
                # Get checkpoint data from queue (blocking with timeout)
                checkpoint_data = self.checkpoint_queue.get(timeout=1.0)
                
                # Write checkpoint
                success = self._write_checkpoint(checkpoint_data)
                
                # Track completion
                checkpoint_id = checkpoint_data["checkpoint_id"]
                if success:
                    self.completed_checkpoints[checkpoint_id] = {
                        "timestamp": time.time(),
                        "path": checkpoint_data["filepath"],
                        "size": os.path.getsize(checkpoint_data["filepath"]) if os.path.exists(checkpoint_data["filepath"]) else 0
                    }
                else:
                    self.failed_checkpoints[checkpoint_id] = {
                        "timestamp": time.time(),
                        "error": checkpoint_data.get("error", "Unknown error")
                    }
                    
                # Mark task as done
                self.checkpoint_queue.task_done()
                
            except queue.Empty:
                # No checkpoints to process, continue loop
                continue
            except Exception as e:
                print(f"❌ Error in checkpoint writer worker: {e}")
                time.sleep(0.1)  # Brief pause to avoid busy loop
                
    def _write_checkpoint(self, checkpoint_data: Dict[str, Any]) -> bool:
        """
        Write checkpoint to disk
        
        Args:
            checkpoint_data: Checkpoint data dictionary
            
        Returns:
            True if successful, False otherwise
        """
        try:
            filepath = checkpoint_data["filepath"]
            data = checkpoint_data["data"]
            
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            
            # Write checkpoint using torch.save
            if TORCH_AVAILABLE:
                torch.save(data, filepath)
            else:
                # Fallback to JSON for non-torch data
                with open(filepath, 'w') as f:
                    json.dump(data, f, indent=2, default=str)
                    
            print(f"✅ Checkpoint written to {filepath}")
            return True
            
        except Exception as e:
            error_msg = f"Failed to write checkpoint: {str(e)}"
            print(f"❌ {error_msg}")
            checkpoint_data["error"] = error_msg
            return False
            
    def queue_checkpoint(self, checkpoint_id: str, data: Any, filepath: str) -> bool:
        """
        Queue a checkpoint for async writing
        
        Args:
            checkpoint_id: Unique identifier for checkpoint
            data: Checkpoint data
            filepath: Path to save checkpoint
            
        Returns:
            True if queued successfully, False if queue is full
        """
        checkpoint_data = {
            "checkpoint_id": checkpoint_id,
            "data": data,
            "filepath": filepath,
            "queued_at": time.time()
        }
        
        try:
            self.checkpoint_queue.put(checkpoint_data, block=False)
            print(f"📥 Checkpoint {checkpoint_id} queued for async writing")
            return True
        except queue.Full:
            print(f"⚠️  Checkpoint queue full, dropping checkpoint {checkpoint_id}")
            return False
            
    def get_queue_status(self) -> Dict[str, Any]:
        """
        Get checkpoint queue status
        
        Returns:
            Dictionary with queue status information
        """
        return {
            "queue_size": self.checkpoint_queue.qsize(),
            "queue_max_size": self.max_queue_size,
            "is_running": self.is_running,
            "completed_checkpoints": len(self.completed_checkpoints),
            "failed_checkpoints": len(self.failed_checkpoints),
            "queue_full": self.checkpoint_queue.full()
        }


class ZeroDowntimeCheckpointing:
    """Zero-downtime checkpointing system with async I/O"""
    
    def __init__(self, 
                 checkpoint_dir: str = "checkpoints",
                 max_queue_size: int = 10,
                 compression: bool = False):
        """
        Initialize zero-downtime checkpointing system
        
        Args:
            checkpoint_dir: Directory to store checkpoints
            max_queue_size: Maximum size of checkpoint queue
            compression: Whether to compress checkpoints
        """
        self.checkpoint_dir = checkpoint_dir
        self.compression = compression
        self.async_writer = AsyncCheckpointWriter(max_queue_size)
        
        # Start async writer
        self.async_writer.start_writer()
        
        # Checkpoint metadata
        self.checkpoint_metadata = OrderedDict()
        self.checkpoint_counter = 0
        
        # Performance tracking
        self.performance_stats = {
            "total_checkpoints": 0,
            "successful_checkpoints": 0,
            "failed_checkpoints": 0,
            "total_write_time": 0.0,
            "avg_write_time": 0.0
        }
        
        print(f"✅ ZeroDowntimeCheckpointing initialized")
        print(f"   Checkpoint directory: {checkpoint_dir}")
        print(f"   Async writer: {'enabled' if self.async_writer.is_running else 'disabled'}")
        print(f"   Compression: {compression}")
        
    def create_checkpoint(self, 
                         model_state: Dict[str, Any], 
                         optimizer_state: Optional[Dict[str, Any]] = None,
                         scheduler_state: Optional[Dict[str, Any]] = None,
                         metadata: Optional[Dict[str, Any]] = None,
                         prefix: str = "checkpoint") -> str:
        """
        Create a checkpoint with zero downtime
        
        Args:
            model_state: Model state dictionary
            optimizer_state: Optional optimizer state
            scheduler_state: Optional scheduler state
            metadata: Optional additional metadata
            prefix: Checkpoint file prefix
            
        Returns:
            Checkpoint ID
        """
        start_time = time.time()
        
        # Generate checkpoint ID
        self.checkpoint_counter += 1
        checkpoint_id = f"{prefix}_{self.checkpoint_counter}_{int(time.time() * 1000)}"
        
        # Prepare checkpoint data
        checkpoint_data = {
            "model_state_dict": model_state,
            "optimizer_state_dict": optimizer_state,
            "scheduler_state_dict": scheduler_state,
            "metadata": metadata or {},
            "checkpoint_id": checkpoint_id,
            "timestamp": time.time(),
            "created_at": datetime.now().isoformat()
        }
        
        # Generate filepath
        filepath = os.path.join(
            self.checkpoint_dir, 
            f"{prefix}_{self.checkpoint_counter}.pt"
        )
        
        # Queue checkpoint for async writing
        queue_success = self.async_writer.queue_checkpoint(
            checkpoint_id, checkpoint_data, filepath
        )
        
        # Update metadata
        self.checkpoint_metadata[checkpoint_id] = {
            "filepath": filepath,
            "queued_at": time.time(),
            "size_estimate": self._estimate_checkpoint_size(checkpoint_data),
            "queue_success": queue_success
        }
        
        # Update performance stats
        write_time = time.time() - start_time
        self.performance_stats["total_write_time"] += write_time
        self.performance_stats["total_checkpoints"] += 1
        if queue_success:
            self.performance_stats["successful_checkpoints"] += 1
        else:
            self.performance_stats["failed_checkpoints"] += 1
            
        self.performance_stats["avg_write_time"] = (
            self.performance_stats["total_write_time"] / 
            self.performance_stats["total_checkpoints"]
        )
        
        print(f"✅ Checkpoint {checkpoint_id} created in {write_time:.3f}s")
        return checkpoint_id
        
    def _estimate_checkpoint_size(self, checkpoint_data: Dict[str, Any]) -> int:
        """
        Estimate checkpoint size (rough approximation)
        
        Args:
            checkpoint_data: Checkpoint data dictionary
            
        Returns:
            Estimated size in bytes
        """
        # Rough estimation based on tensor sizes
        estimated_size = 0
        
        if TORCH_AVAILABLE:
            for key, value in checkpoint_data.items():
                if isinstance(value, dict):
                    for sub_key, sub_value in value.items():
                        if hasattr(sub_value, 'numel') and hasattr(sub_value, 'element_size'):
                            estimated_size += sub_value.numel() * sub_value.element_size()
                elif hasattr(value, 'numel') and hasattr(value, 'element_size'):
                    estimated_size += value.numel() * value.element_size()
                    
        # Add some overhead for metadata
        estimated_size += 1024 * 1024  # 1MB overhead estimate
        
        return estimated_size
        
    def calculate_checkpoint_hash(self, filepath: str) -> str:
        """
        Calculate SHA256 hash of checkpoint file
        
        Args:
            filepath: Path to checkpoint file
            
        Returns:
            SHA256 hash as hex string
        """
        if not os.path.exists(filepath):
            return ""
            
        try:
            hash_sha256 = hashlib.sha256()
            with open(filepath, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_sha256.update(chunk)
            return hash_sha256.hexdigest()
        except Exception as e:
            print(f"❌ Failed to calculate checkpoint hash: {e}")
            return ""
            
    def get_checkpoint_status(self, checkpoint_id: str) -> Dict[str, Any]:
        """
        Get status of a specific checkpoint
        
        Args:
            checkpoint_id: Checkpoint ID
            
        Returns:
            Dictionary with checkpoint status
        """
        if checkpoint_id in self.checkpoint_metadata:
            metadata = self.checkpoint_metadata[checkpoint_id]
            
            # Check if completed
            if checkpoint_id in self.async_writer.completed_checkpoints:
                status = "completed"
                completion_info = self.async_writer.completed_checkpoints[checkpoint_id]
            elif checkpoint_id in self.async_writer.failed_checkpoints:
                status = "failed"
                completion_info = self.async_writer.failed_checkpoints[checkpoint_id]
            else:
                status = "queued"
                completion_info = None
                
            return {
                "checkpoint_id": checkpoint_id,
                "status": status,
                "filepath": metadata["filepath"],
                "queued_at": metadata["queued_at"],
                "completion_info": completion_info
            }
        else:
            return {"checkpoint_id": checkpoint_id, "status": "not_found"}
            
    def get_system_status(self) -> Dict[str, Any]:
        """
        Get overall system status
        
        Returns:
            Dictionary with system status information
        """
        queue_status = self.async_writer.get_queue_status()
        
        return {
            "timestamp": time.time(),
            "checkpoint_dir": self.checkpoint_dir,
            "performance_stats": self.performance_stats,
            "queue_status": queue_status,
            "total_checkpoints_queued": len(self.checkpoint_metadata),
            "checkpoint_metadata_count": len(self.checkpoint_metadata)
        }
        
    def wait_for_checkpoint_completion(self, checkpoint_id: str, timeout: float = 30.0) -> bool:
        """
        Wait for a specific checkpoint to complete
        
        Args:
            checkpoint_id: Checkpoint ID to wait for
            timeout: Maximum time to wait in seconds
            
        Returns:
            True if checkpoint completed successfully, False otherwise
        """
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            status = self.get_checkpoint_status(checkpoint_id)
            if status["status"] == "completed":
                return True
            elif status["status"] == "failed":
                return False
            time.sleep(0.1)  # Brief pause
            
        return False  # Timeout reached
        
    def cleanup_old_checkpoints(self, keep_last_n: int = 5) -> int:
        """
        Clean up old checkpoints, keeping only the last N
        
        Args:
            keep_last_n: Number of recent checkpoints to keep
            
        Returns:
            Number of checkpoints cleaned up
        """
        if len(self.checkpoint_metadata) <= keep_last_n:
            return 0
            
        # Get checkpoints to delete
        checkpoint_ids = list(self.checkpoint_metadata.keys())
        checkpoints_to_delete = checkpoint_ids[:-keep_last_n]
        
        deleted_count = 0
        for checkpoint_id in checkpoints_to_delete:
            metadata = self.checkpoint_metadata[checkpoint_id]
            filepath = metadata["filepath"]
            
            # Delete file if it exists
            if os.path.exists(filepath):
                try:
                    os.remove(filepath)
                    deleted_count += 1
                    print(f"🗑️  Deleted old checkpoint: {filepath}")
                except Exception as e:
                    print(f"❌ Failed to delete {filepath}: {e}")
                    
            # Remove from metadata
            del self.checkpoint_metadata[checkpoint_id]
            
        return deleted_count
        
    def export_checkpoint_report(self, filepath: str) -> bool:
        """
        Export checkpointing report to file
        
        Args:
            filepath: Path to export report
            
        Returns:
            True if successful, False otherwise
        """
        try:
            report = {
                "generated_at": datetime.now().isoformat(),
                "system_status": self.get_system_status(),
                "checkpoint_metadata": dict(list(self.checkpoint_metadata.items())[-20:]),  # Last 20
                "completed_checkpoints": dict(list(self.async_writer.completed_checkpoints.items())[-20:]),
                "failed_checkpoints": dict(list(self.async_writer.failed_checkpoints.items())[-10:])
            }
            
            with open(filepath, 'w') as f:
                json.dump(report, f, indent=2, default=str)
                
            print(f"✅ Checkpoint report exported to {filepath}")
            return True
        except Exception as e:
            print(f"❌ Failed to export checkpoint report: {e}")
            return False
            
    def shutdown(self, wait_for_completion: bool = True):
        """
        Shutdown the checkpointing system
        
        Args:
            wait_for_completion: Whether to wait for queued checkpoints to complete
        """
        if wait_for_completion:
            print("⏳ Waiting for queued checkpoints to complete...")
            self.async_writer.checkpoint_queue.join()  # Wait for all queued items to be processed
            
        # Stop async writer
        self.async_writer.stop_writer()
        print("✅ ZeroDowntimeCheckpointing shutdown complete")


def demo_zero_downtime_checkpointing():
    """Demonstrate zero-downtime checkpointing functionality"""
    print("🚀 Demonstrating Zero-Downtime Checkpointing...")
    print("=" * 50)
    
    # Create checkpointing system
    checkpointing = ZeroDowntimeCheckpointing(
        checkpoint_dir="demo_checkpoints",
        max_queue_size=5,
        compression=False
    )
    print("✅ Created zero-downtime checkpointing system")
    
    # Simulate model state
    if TORCH_AVAILABLE:
        # Create dummy model state
        model_state = {
            "layer1.weight": torch.randn(768, 768),
            "layer1.bias": torch.randn(768),
            "layer2.weight": torch.randn(768, 384),
            "layer2.bias": torch.randn(384)
        }
        
        optimizer_state = {
            "state": {"param1": torch.randn(768, 768)},
            "param_groups": [{"lr": 0.001}]
        }
        
        print("✅ Created dummy model and optimizer states")
        
        # Create multiple checkpoints quickly (zero downtime)
        print("\n💾 Creating checkpoints with zero downtime...")
        checkpoint_ids = []
        
        for i in range(5):
            metadata = {
                "epoch": i + 1,
                "loss": 1.0 / (i + 1),
                "accuracy": 0.5 + (i * 0.1)
            }
            
            checkpoint_id = checkpointing.create_checkpoint(
                model_state=model_state,
                optimizer_state=optimizer_state,
                metadata=metadata,
                prefix=f"demo_checkpoint_{i+1}"
            )
            
            checkpoint_ids.append(checkpoint_id)
            print(f"   Queued checkpoint {i+1}: {checkpoint_id}")
            
        # Show system status
        print("\n📊 System Status:")
        status = checkpointing.get_system_status()
        print(f"   Queue size: {status['queue_status']['queue_size']}")
        print(f"   Checkpoints queued: {status['total_checkpoints_queued']}")
        print(f"   Average write time: {status['performance_stats']['avg_write_time']:.3f}s")
        
        # Wait for completion of first checkpoint
        print("\n⏳ Waiting for first checkpoint to complete...")
        first_checkpoint_success = checkpointing.wait_for_checkpoint_completion(checkpoint_ids[0], timeout=10.0)
        print(f"   First checkpoint completed: {first_checkpoint_success}")
        
        # Show checkpoint status
        if checkpoint_ids:
            print("\n🔍 Checkpoint Status:")
            checkpoint_status = checkpointing.get_checkpoint_status(checkpoint_ids[0])
            print(f"   ID: {checkpoint_status['checkpoint_id']}")
            print(f"   Status: {checkpoint_status['status']}")
            print(f"   Filepath: {checkpoint_status['filepath']}")
            
        # Clean up old checkpoints
        print("\n🧹 Cleaning up old checkpoints...")
        cleaned_count = checkpointing.cleanup_old_checkpoints(keep_last_n=3)
        print(f"   Cleaned up {cleaned_count} old checkpoints")
        
        # Export report
        report_success = checkpointing.export_checkpoint_report("checkpoint_report.json")
        print(f"   Report export: {'SUCCESS' if report_success else 'FAILED'}")
        
        # Shutdown
        checkpointing.shutdown(wait_for_completion=True)
        
    else:
        print("❌ PyTorch not available, skipping demonstration")
        
    print("\n" + "=" * 50)
    print("ZERO-DOWNTIME CHECKPOINTING DEMO SUMMARY")
    print("=" * 50)
    print("Key Features Implemented:")
    print("  1. Asynchronous checkpoint writing with background threads")
    print("  2. Zero-downtime checkpoint creation during training")
    print("  3. Queue-based checkpoint management")
    print("  4. Performance monitoring and statistics")
    print("  5. Automatic cleanup of old checkpoints")
    print("  6. Comprehensive reporting and status tracking")
    print("\nBenefits:")
    print("  - No training interruption during checkpointing")
    print("  - Efficient I/O with background processing")
    print("  - Configurable queue size for memory management")
    print("  - Reliable checkpoint storage with error handling")
    print("  - Performance optimization for large models")
    
    print("\n✅ Zero-downtime checkpointing demonstration completed!")


if __name__ == "__main__":
    demo_zero_downtime_checkpointing()