"""
Deterministic Shuffling with Seed Logging for MAHIA-X
This module implements deterministic shuffling with seed logging for reproducibility.
"""

import random
import time
import json
import os
from typing import List, Dict, Any, Optional
from collections import OrderedDict

# Conditional imports with fallbacks
TORCH_AVAILABLE = False
torch = None

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    pass

class DeterministicShuffler:
    """Deterministic shuffling with seed logging for reproducibility"""
    
    def __init__(self, seed: int = 42, log_file: Optional[str] = None):
        """
        Initialize deterministic shuffler
        
        Args:
            seed: Random seed for shuffling
            log_file: File to log shuffle operations (optional)
        """
        self.seed = seed
        self.log_file = log_file
        self.shuffle_log = []
        
        # Set initial seed
        self._set_seed(seed)
        
        print(f"✅ DeterministicShuffler initialized with seed: {seed}")
        
    def _set_seed(self, seed: int):
        """Set random seed for all relevant libraries"""
        # Set Python random seed
        random.seed(seed)
        
        # Set NumPy seed if available
        try:
            import numpy as np
            np.random.seed(seed)
        except ImportError:
            pass
            
        # Set PyTorch seed if available
        if TORCH_AVAILABLE and torch is not None:
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(seed)
                torch.cuda.manual_seed_all(seed)
                
        # Set environment variable for hash consistency
        os.environ['PYTHONHASHSEED'] = str(seed)
        
    def shuffle_list(self, items: List[Any], seed: Optional[int] = None) -> List[Any]:
        """
        Shuffle a list deterministically
        
        Args:
            items: List to shuffle
            seed: Seed to use (uses instance seed if None)
            
        Returns:
            Shuffled list
        """
        # Use provided seed or instance seed
        shuffle_seed = seed if seed is not None else self.seed
        
        # Set seed for this shuffle operation
        self._set_seed(shuffle_seed)
        
        # Create a copy to avoid modifying original
        shuffled_items = list(items)
        
        # Shuffle using Python's random module
        random.shuffle(shuffled_items)
        
        # Log shuffle operation
        shuffle_record = {
            "seed": shuffle_seed,
            "timestamp": time.time(),
            "item_count": len(items),
            "operation": "shuffle_list"
        }
        self.shuffle_log.append(shuffle_record)
        
        # Save to log file if specified
        if self.log_file:
            self._save_log_to_file()
            
        return shuffled_items
        
    def shuffle_indices(self, indices: List[int], seed: Optional[int] = None) -> List[int]:
        """
        Shuffle indices deterministically
        
        Args:
            indices: List of indices to shuffle
            seed: Seed to use (uses instance seed if None)
            
        Returns:
            Shuffled indices
        """
        return self.shuffle_list(indices, seed)
        
    def get_permutation(self, n: int, seed: Optional[int] = None) -> List[int]:
        """
        Get a deterministic permutation of indices
        
        Args:
            n: Number of indices (0 to n-1)
            seed: Seed to use (uses instance seed if None)
            
        Returns:
            Permutation of indices
        """
        # Use provided seed or instance seed
        shuffle_seed = seed if seed is not None else self.seed
        
        # Set seed for this operation
        self._set_seed(shuffle_seed)
        
        # Create indices
        indices = list(range(n))
        
        # Shuffle indices
        random.shuffle(indices)
        
        # Log operation
        shuffle_record = {
            "seed": shuffle_seed,
            "timestamp": time.time(),
            "item_count": n,
            "operation": "get_permutation"
        }
        self.shuffle_log.append(shuffle_record)
        
        # Save to log file if specified
        if self.log_file:
            self._save_log_to_file()
            
        return indices
        
    def _save_log_to_file(self):
        """Save shuffle log to file"""
        if not self.log_file:
            return
            
        try:
            # Create directory if it doesn't exist
            log_dir = os.path.dirname(self.log_file)
            if log_dir and not os.path.exists(log_dir):
                os.makedirs(log_dir)
                
            # Write log to file
            with open(self.log_file, 'w') as f:
                json.dump(self.shuffle_log, f, indent=2)
        except Exception as e:
            print(f"⚠️  Failed to save shuffle log: {e}")
            
    def load_log_from_file(self, log_file: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Load shuffle log from file
        
        Args:
            log_file: File to load from (uses instance log_file if None)
            
        Returns:
            Shuffle log
        """
        file_path = log_file if log_file is not None else self.log_file
        if not file_path:
            return []
            
        try:
            with open(file_path, 'r') as f:
                loaded_log = json.load(f)
                return loaded_log
        except Exception as e:
            print(f"⚠️  Failed to load shuffle log: {e}")
            return []
            
    def get_shuffle_log(self) -> List[Dict[str, Any]]:
        """Get shuffle operation log"""
        return self.shuffle_log
        
    def clear_log(self):
        """Clear shuffle log"""
        self.shuffle_log = []
        
    def reset_seed(self, seed: int):
        """Reset seed and clear shuffle history"""
        self.seed = seed
        self._set_seed(seed)
        print(f"✅ Shuffle seed reset to {seed}")
        
    def get_current_seed(self) -> int:
        """Get current seed"""
        return self.seed


class ShuffleLogger:
    """Shuffle logger for tracking and reproducing shuffle operations"""
    
    def __init__(self, log_file: str = "shuffle_log.json"):
        """
        Initialize shuffle logger
        
        Args:
            log_file: File to log shuffle operations
        """
        self.log_file = log_file
        self.operation_log = OrderedDict()
        
    def log_shuffle_operation(self, operation_id: str, seed: int, 
                            item_count: int, operation_type: str, 
                            metadata: Optional[Dict[str, Any]] = None):
        """
        Log a shuffle operation
        
        Args:
            operation_id: Unique identifier for operation
            seed: Seed used for shuffling
            item_count: Number of items shuffled
            operation_type: Type of shuffle operation
            metadata: Additional metadata (optional)
        """
        log_entry = {
            "operation_id": operation_id,
            "seed": seed,
            "item_count": item_count,
            "operation_type": operation_type,
            "timestamp": time.time(),
            "metadata": metadata or {}
        }
        
        self.operation_log[operation_id] = log_entry
        self._save_log()
        
    def _save_log(self):
        """Save log to file"""
        try:
            # Keep only last 1000 operations to prevent file from growing too large
            if len(self.operation_log) > 1000:
                # Remove oldest entries
                keys_to_remove = list(self.operation_log.keys())[:len(self.operation_log) - 1000]
                for key in keys_to_remove:
                    del self.operation_log[key]
                    
            # Write log to file
            with open(self.log_file, 'w') as f:
                json.dump(dict(self.operation_log), f, indent=2)
        except Exception as e:
            print(f"⚠️  Failed to save shuffle log: {e}")
            
    def load_log(self) -> Dict[str, Dict[str, Any]]:
        """Load log from file"""
        try:
            if os.path.exists(self.log_file):
                with open(self.log_file, 'r') as f:
                    loaded_log = json.load(f)
                    self.operation_log = OrderedDict(loaded_log)
                    return dict(self.operation_log)
            return {}
        except Exception as e:
            print(f"⚠️  Failed to load shuffle log: {e}")
            return {}
            
    def get_operation(self, operation_id: str) -> Optional[Dict[str, Any]]:
        """Get specific operation by ID"""
        return self.operation_log.get(operation_id)
        
    def get_operations_by_type(self, operation_type: str) -> List[Dict[str, Any]]:
        """Get all operations of a specific type"""
        return [op for op in self.operation_log.values() if op.get("operation_type") == operation_type]
        
    def clear_log(self):
        """Clear log"""
        self.operation_log.clear()
        self._save_log()


def demo_deterministic_shuffling():
    """Demonstrate deterministic shuffling functionality"""
    print("🚀 Demonstrating Deterministic Shuffling with Seed Logging...")
    print("=" * 60)
    
    # Create deterministic shuffler
    shuffler = DeterministicShuffler(seed=123, log_file="shuffle_demo_log.json")
    print("✅ Created deterministic shuffler with seed: 123")
    
    # Create sample data
    sample_data = list(range(20))
    print(f"✅ Created sample data: {sample_data[:10]}...")
    
    # Shuffle data deterministically
    shuffled_data1 = shuffler.shuffle_list(sample_data)
    print(f"✅ First shuffle result: {shuffled_data1[:10]}...")
    
    # Reset seed and shuffle again - should be identical
    shuffler.reset_seed(123)
    shuffled_data2 = shuffler.shuffle_list(sample_data)
    print(f"✅ Second shuffle result: {shuffled_data2[:10]}...")
    
    # Check if results are identical
    if shuffled_data1 == shuffled_data2:
        print("✅ Verification successful - shuffles are identical with same seed")
    else:
        print("❌ Verification failed - shuffles are different with same seed")
        
    # Try with different seed
    shuffler.reset_seed(456)
    shuffled_data3 = shuffler.shuffle_list(sample_data)
    print(f"✅ Third shuffle (different seed): {shuffled_data3[:10]}...")
    
    # Check if results are different
    if shuffled_data1 != shuffled_data3:
        print("✅ Verification successful - shuffles are different with different seeds")
    else:
        print("❌ Verification failed - shuffles are identical with different seeds")
        
    # Test index shuffling
    indices = list(range(100))
    shuffled_indices = shuffler.shuffle_indices(indices)
    print(f"✅ Shuffled 100 indices, first 10: {shuffled_indices[:10]}")
    
    # Test permutation
    permutation = shuffler.get_permutation(50)
    print(f"✅ Generated permutation of 50 items, first 10: {permutation[:10]}")
    
    # Show shuffle log
    shuffle_log = shuffler.get_shuffle_log()
    print(f"✅ Shuffle operations logged: {len(shuffle_log)}")
    
    # Show log details
    for i, log_entry in enumerate(shuffle_log[-3:]):  # Show last 3 entries
        print(f"   Log entry {len(shuffle_log)-2+i}: Seed={log_entry['seed']}, Items={log_entry['item_count']}, Op={log_entry['operation']}")
    
    # Create shuffle logger
    logger = ShuffleLogger("detailed_shuffle_log.json")
    print("✅ Created detailed shuffle logger")
    
    # Log some operations
    logger.log_shuffle_operation("op1", 123, 1000, "dataset_shuffle", {"dataset": "train"})
    logger.log_shuffle_operation("op2", 456, 500, "dataset_shuffle", {"dataset": "validation"})
    
    # Retrieve operations
    train_ops = logger.get_operations_by_type("dataset_shuffle")
    print(f"✅ Logged {len(train_ops)} dataset shuffle operations")
    
    # Show current seed
    current_seed = shuffler.get_current_seed()
    print(f"✅ Current seed: {current_seed}")
    
    print("\n" + "=" * 60)
    print("DETERMINISTIC SHUFFLING DEMO SUMMARY")
    print("=" * 60)
    print("Key Features Implemented:")
    print("  1. Deterministic shuffling with fixed seeds")
    print("  2. Comprehensive seed logging")
    print("  3. Reproducible shuffle operations")
    print("  4. Multi-library seed management")
    print("  5. Detailed operation tracking")
    print("  6. File-based log persistence")
    print("\nBenefits:")
    print("  - Reproducible data processing pipelines")
    print("  - Consistent experimental results")
    print("  - Audit trail for shuffle operations")
    print("  - Cross-platform compatibility")
    print("  - Easy debugging and verification")
    
    print("\n✅ Deterministic Shuffling demonstration completed!")


if __name__ == "__main__":
    demo_deterministic_shuffling()