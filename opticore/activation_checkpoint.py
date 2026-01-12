"""
Activation Checkpoint Controller for MAHIA OptiCore
Layer-selective caching with recomputation on demand and adaptive mode logic.
"""

import time
import threading
from typing import Dict, Any, Optional, List, Callable, Tuple
from collections import defaultdict

# Conditional imports with fallbacks
TORCH_AVAILABLE = False
torch = None

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    pass

class ActivationCheckpointController:
    """Layer-selective caching with recomputation on demand"""
    
    def __init__(self, max_cache_size: int = 100):
        self.max_cache_size = max_cache_size
        self.checkpoint_cache = {}  # cache for activations
        self.cache_metadata = {}  # metadata for cached activations
        self.telemetry_feedback = {}  # telemetry data for adaptive decisions
        self.lock = threading.Lock()
        self.stats = {
            "checkpoints_created": 0,
            "checkpoints_reused": 0,
            "checkpoints_evicted": 0,
            "recomputations": 0
        }
        
        print(f"🧠 ActivationCheckpointController initialized with max cache size: {max_cache_size}")
        
    def checkpoint(self, layer_id: str, activations: Any, 
                   metadata: Optional[Dict[str, Any]] = None) -> bool:
        """
        Create a checkpoint for activations.
        
        Args:
            layer_id: Identifier for the layer
            activations: Activations to checkpoint
            metadata: Additional metadata
            
        Returns:
            bool: True if successful, False otherwise
        """
        with self.lock:
            # Check if cache is full
            if len(self.checkpoint_cache) >= self.max_cache_size:
                # Evict oldest checkpoint
                oldest_key = min(self.cache_metadata.keys(), 
                               key=lambda k: self.cache_metadata[k]["timestamp"])
                del self.checkpoint_cache[oldest_key]
                del self.cache_metadata[oldest_key]
                self.stats["checkpoints_evicted"] += 1
                print(f"🗑️  Evicted oldest checkpoint: {oldest_key}")
            
            # Store checkpoint
            self.checkpoint_cache[layer_id] = activations
            self.cache_metadata[layer_id] = {
                "timestamp": time.time(),
                "metadata": metadata or {},
                "size": self._estimate_size(activations)
            }
            self.stats["checkpoints_created"] += 1
            
            print(f"💾 Checkpointed activations for layer: {layer_id}")
            return True
            
    def _estimate_size(self, activations: Any) -> int:
        """
        Estimate the size of activations in bytes.
        
        Args:
            activations: Activations to estimate size for
            
        Returns:
            int: Estimated size in bytes
        """
        if TORCH_AVAILABLE and torch is not None and isinstance(activations, torch.Tensor):
            return activations.element_size() * activations.nelement()
        elif hasattr(activations, 'nbytes'):
            return activations.nbytes
        else:
            # Fallback estimation
            return 0
            
    def restore(self, layer_id: str, recompute_fn: Optional[Callable] = None) -> Any:
        """
        Restore activations from checkpoint or recompute.
        
        Args:
            layer_id: Identifier for the layer
            recompute_fn: Function to recompute activations if not found
            
        Returns:
            Restored or recomputed activations
        """
        with self.lock:
            if layer_id in self.checkpoint_cache:
                # Restore from cache
                activations = self.checkpoint_cache[layer_id]
                self.stats["checkpoints_reused"] += 1
                print(f"🔄 Restored activations for layer: {layer_id}")
                return activations
            elif recompute_fn is not None:
                # Recompute activations
                try:
                    activations = recompute_fn()
                    self.stats["recomputations"] += 1
                    print(f"🧮 Recomputed activations for layer: {layer_id}")
                    return activations
                except Exception as e:
                    print(f"❌ Failed to recompute activations: {e}")
                    return None
            else:
                print(f"⚠️  No checkpoint or recompute function for layer: {layer_id}")
                return None
                
    def update_telemetry_feedback(self, layer_id: str, feedback: Dict[str, Any]):
        """
        Update telemetry feedback for adaptive decisions.
        
        Args:
            layer_id: Identifier for the layer
            feedback: Telemetry feedback data
        """
        with self.lock:
            if layer_id not in self.telemetry_feedback:
                self.telemetry_feedback[layer_id] = []
                
            self.telemetry_feedback[layer_id].append({
                "timestamp": time.time(),
                "feedback": feedback
            })
            
            # Keep only last 100 feedback entries
            if len(self.telemetry_feedback[layer_id]) > 100:
                self.telemetry_feedback[layer_id] = self.telemetry_feedback[layer_id][-100:]
                
    def should_checkpoint(self, layer_id: str, current_loss: float = 0.0, 
                         gradient_norm: float = 0.0) -> bool:
        """
        Determine if layer should be checkpointed based on adaptive logic.
        
        Args:
            layer_id: Identifier for the layer
            current_loss: Current loss value
            gradient_norm: Gradient norm
            
        Returns:
            bool: True if should checkpoint, False otherwise
        """
        # Simple adaptive logic - in a real implementation, this would be more sophisticated
        with self.lock:
            # If we have telemetry feedback, use it
            if layer_id in self.telemetry_feedback and self.telemetry_feedback[layer_id]:
                recent_feedback = self.telemetry_feedback[layer_id][-10:]  # Last 10 feedbacks
                avg_loss = sum(f["feedback"].get("loss", 0) for f in recent_feedback) / len(recent_feedback)
                
                # Checkpoint if loss is increasing or gradient norm is high
                if current_loss > avg_loss * 1.1 or gradient_norm > 1.0:
                    return True
                    
            # Default: checkpoint if cache is not full
            return len(self.checkpoint_cache) < self.max_cache_size
            
    def get_stats(self) -> Dict[str, Any]:
        """Get activation checkpoint statistics"""
        with self.lock:
            return self.stats.copy()
            
    def clear_stats(self):
        """Clear statistics"""
        with self.lock:
            self.stats = {
                "checkpoints_created": 0,
                "checkpoints_reused": 0,
                "checkpoints_evicted": 0,
                "recomputations": 0
            }
        print("🗑️  Activation checkpoint statistics cleared")
        
    def clear_cache(self):
        """Clear all checkpoints"""
        with self.lock:
            self.checkpoint_cache.clear()
            self.cache_metadata.clear()
            self.telemetry_feedback.clear()
        print("🧨 All checkpoints cleared")

# Global instance
_activation_checkpoint = None

def get_activation_checkpoint() -> ActivationCheckpointController:
    """Get the global activation checkpoint controller instance"""
    global _activation_checkpoint
    if _activation_checkpoint is None:
        _activation_checkpoint = ActivationCheckpointController()
    return _activation_checkpoint

if __name__ == "__main__":
    # Example usage
    controller = get_activation_checkpoint()
    
    # Simulate some activations
    sample_activations = [1.0, 2.0, 3.0, 4.0, 5.0]
    
    # Checkpoint activations
    controller.checkpoint("layer_1", sample_activations, {"epoch": 1, "batch": 10})
    
    # Restore activations
    restored = controller.restore("layer_1")
    print(f"Restored: {restored}")
    
    # Try to restore non-existent layer with recompute function
    def recompute():
        return [6.0, 7.0, 8.0, 9.0, 10.0]
        
    recomputed = controller.restore("layer_2", recompute)
    print(f"Recomputed: {recomputed}")
    
    # Print stats
    print(f"📊 Stats: {controller.get_stats()}")