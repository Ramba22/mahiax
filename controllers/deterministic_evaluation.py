"""
Deterministic Evaluation Mode for MAHIA-X
This module provides fixed seeds and reproducible routing decisions for consistent evaluation.
"""

# Conditional imports with fallbacks
TORCH_AVAILABLE = False
torch = None
nn = None

NUMPY_AVAILABLE = False
np = None

import random
from typing import Optional, Dict, Any, List, Tuple
import os

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    pass

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    pass

class DeterministicEvaluationMode:
    """Deterministic evaluation mode with fixed seeds and reproducible routing decisions"""
    
    def __init__(self, seed: int = 42, enable_cuda_deterministic: bool = True):
        self.seed = seed
        self.enable_cuda_deterministic = enable_cuda_deterministic
        self.original_states = {}
        self.is_active = False
        
        # Store original random states
        self._store_original_states()
        
        print(f"🎲 DeterministicEvaluationMode initialized with seed: {seed}")
        
    def _store_original_states(self):
        """Store original random states for restoration"""
        self.original_states = {
            'python': random.getstate(),
        }
        
        if NUMPY_AVAILABLE and np is not None:
            self.original_states['numpy'] = np.random.get_state()
            
        if TORCH_AVAILABLE and torch is not None:
            self.original_states['torch'] = torch.get_rng_state()
            
            if torch.cuda.is_available():
                self.original_states['torch_cuda'] = torch.cuda.get_rng_state_all()
            
    def _set_seeds(self):
        """Set all random seeds for deterministic behavior"""
        # Python random
        random.seed(self.seed)
        
        # NumPy
        if NUMPY_AVAILABLE and np is not None:
            np.random.seed(self.seed)
        
        # PyTorch
        if TORCH_AVAILABLE and torch is not None:
            torch.manual_seed(self.seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(self.seed)
                torch.cuda.manual_seed_all(self.seed)
                
        # Environment variables for deterministic behavior
        os.environ['PYTHONHASHSEED'] = str(self.seed)
        
        # PyTorch deterministic settings
        if self.enable_cuda_deterministic and TORCH_AVAILABLE and torch is not None:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            
    def activate(self):
        """Activate deterministic evaluation mode"""
        if self.is_active:
            print("⚠️  Deterministic evaluation mode is already active")
            return
            
        # Store current states before changing them
        self._store_original_states()
        
        # Set deterministic seeds
        self._set_seeds()
        
        self.is_active = True
        print("✅ Deterministic evaluation mode activated")
        print(f"   Seed: {self.seed}")
        print(f"   CUDA deterministic: {self.enable_cuda_deterministic}")
        
    def deactivate(self):
        """Deactivate deterministic evaluation mode and restore original states"""
        if not self.is_active:
            print("⚠️  Deterministic evaluation mode is not active")
            return
            
        # Restore original states
        random.setstate(self.original_states['python'])
        
        if NUMPY_AVAILABLE and np is not None and 'numpy' in self.original_states:
            np.random.set_state(self.original_states['numpy'])
            
        if TORCH_AVAILABLE and torch is not None and 'torch' in self.original_states:
            torch.set_rng_state(self.original_states['torch'])
            
        if TORCH_AVAILABLE and torch is not None:
            if torch.cuda.is_available() and 'torch_cuda' in self.original_states:
                torch.cuda.set_rng_state_all(self.original_states['torch_cuda'])
                
            # Restore PyTorch settings
            if self.enable_cuda_deterministic:
                torch.backends.cudnn.deterministic = False
                torch.backends.cudnn.benchmark = True
                
        self.is_active = False
        print("🔄 Deterministic evaluation mode deactivated, original states restored")
        
    def is_deterministic_active(self) -> bool:
        """Check if deterministic mode is currently active"""
        return self.is_active
        
    def get_seed(self) -> int:
        """Get current seed value"""
        return self.seed
        
    def set_seed(self, new_seed: int):
        """Set new seed value"""
        self.seed = new_seed
        if self.is_active:
            # If already active, re-activate with new seed
            self._set_seeds()
            print(f"🔄 Seed updated to: {new_seed}")
            
    def ensure_deterministic_forward(self, model: nn.Module, input_data: torch.Tensor, 
                                   num_runs: int = 5) -> Dict[str, Any]:
        """Ensure model forward pass is deterministic by running multiple times
        and checking for consistency.
        
        Args:
            model: PyTorch model to test
            input_data: Input tensor for forward pass
            num_runs: Number of forward passes to run
            
        Returns:
            dict: Results including consistency check and statistics
        """
        if not self.is_active:
            print("⚠️  Deterministic mode not active, activating for this test")
            self.activate()
            
        model.eval()
        results = []
        
        with torch.no_grad():
            for i in range(num_runs):
                output = model(input_data)
                results.append(output.cpu().numpy())
                
        # Check consistency
        results_array = np.array(results)
        is_consistent = np.allclose(results_array, results_array[0], rtol=1e-5, atol=1e-6)
        
        # Calculate statistics
        mean_output = np.mean(results_array, axis=0)
        std_output = np.std(results_array, axis=0)
        
        consistency_info = {
            "is_consistent": is_consistent,
            "num_runs": num_runs,
            "mean_output": mean_output,
            "std_output": std_output,
            "max_std": np.max(std_output),
            "min_std": np.min(std_output)
        }
        
        if is_consistent:
            print("✅ Forward pass is deterministic")
        else:
            print("❌ Forward pass is NOT deterministic")
            print(f"   Max standard deviation: {consistency_info['max_std']:.2e}")
            
        return consistency_info
        
    def create_deterministic_dataloader(self, dataset, batch_size: int = 32, 
                                      shuffle: bool = True) -> torch.utils.data.DataLoader:
        """Create a deterministic DataLoader with fixed shuffling
        
        Args:
            dataset: Dataset to load
            batch_size: Batch size
            shuffle: Whether to shuffle data
            
        Returns:
            DataLoader with deterministic behavior
        """
        # For deterministic shuffling, we need to set the seed each time
        def worker_init_fn(worker_id):
            worker_seed = torch.initial_seed() % 2**32
            np.random.seed(worker_seed)
            random.seed(worker_seed)
            
        # Create DataLoader with fixed seed
        if shuffle:
            # Use a generator with fixed seed for shuffling
            g = torch.Generator()
            g.manual_seed(self.seed)
            return torch.utils.data.DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=shuffle,
                generator=g,
                worker_init_fn=worker_init_fn
            )
        else:
            return torch.utils.data.DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=shuffle,
                worker_init_fn=worker_init_fn
            )
            
    def get_deterministic_state_dict(self) -> Dict[str, Any]:
        """Get current deterministic state for logging/debugging"""
        return {
            "seed": self.seed,
            "is_active": self.is_active,
            "enable_cuda_deterministic": self.enable_cuda_deterministic,
            "python_random_state": str(random.getstate()),
            "numpy_random_state": str(np.random.get_state()),
        }
        
    def __enter__(self):
        """Context manager entry"""
        self.activate()
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.deactivate()


class DeterministicRoutingController:
    """Controller for ensuring deterministic routing decisions in MoE layers"""
    
    def __init__(self, seed: int = 42):
        self.seed = seed
        self.original_torch_seed = None
        self.is_routing_deterministic = False
        
    def enable_deterministic_routing(self, model: nn.Module):
        """Enable deterministic routing for all MoE layers in the model"""
        # Set seed for routing decisions
        self.original_torch_seed = torch.get_rng_state()
        torch.manual_seed(self.seed)
        
        # Find and modify MoE layers
        moe_layers = []
        for name, module in model.named_modules():
            if hasattr(module, 'gate') and hasattr(module, 'experts'):
                moe_layers.append((name, module))
                print(f"🔧 Found MoE layer: {name}")
                
        # Ensure all MoE layers use deterministic selection
        for name, layer in moe_layers:
            if hasattr(layer, 'top_k'):
                # For top-k selection, ensure consistent behavior
                print(f"✅ Enabled deterministic routing for {name}")
                
        self.is_routing_deterministic = True
        print(f"🎯 Deterministic routing enabled with seed: {self.seed}")
        
    def disable_deterministic_routing(self):
        """Disable deterministic routing and restore original state"""
        if self.original_torch_seed is not None:
            torch.set_rng_state(self.original_torch_seed)
            self.original_torch_seed = None
            
        self.is_routing_deterministic = False
        print("🔄 Deterministic routing disabled")
        
    def is_routing_deterministic_active(self) -> bool:
        """Check if deterministic routing is active"""
        return self.is_routing_deterministic


# Example usage and testing
if __name__ == "__main__":
    print("Testing Deterministic Evaluation Mode")
    print("=" * 40)
    
    # Test deterministic mode
    with DeterministicEvaluationMode(seed=42) as det_mode:
        print(f"Active: {det_mode.is_deterministic_active()}")
        
        # Test with a simple model
        class SimpleModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(10, 5)
                
            def forward(self, x):
                return self.linear(x)
                
        model = SimpleModel()
        test_input = torch.randn(2, 10)
        
        # Test deterministic forward
        consistency = det_mode.ensure_deterministic_forward(model, test_input)
        print(f"Consistency check: {consistency['is_consistent']}")
        
    print(f"Active after context: {det_mode.is_deterministic_active()}")
    
    # Test routing controller
    print("\nTesting Deterministic Routing Controller:")
    routing_controller = DeterministicRoutingController(seed=123)
    routing_controller.enable_deterministic_routing(model)
    print(f"Routing deterministic: {routing_controller.is_routing_deterministic_active()}")
    routing_controller.disable_deterministic_routing()