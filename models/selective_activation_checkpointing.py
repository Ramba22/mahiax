"""
Selective Activation Checkpointing for MAHIA-X
This module implements selective activation checkpointing that recomputes only memory-intensive layers.
"""

# Conditional imports with fallbacks
TORCH_AVAILABLE = False
torch = None
nn = None
checkpoint = None

try:
    import torch
    import torch.nn as nn
    from torch.utils.checkpoint import checkpoint
    TORCH_AVAILABLE = True
except ImportError:
    pass

from typing import Optional, Callable, Any, List, Tuple
import functools

class SelectiveActivationCheckpointing:
    """Selective activation checkpointing that recomputes only memory-intensive layers"""
    
    def __init__(self, memory_threshold: float = 0.8):
        """
        Initialize selective activation checkpointing
        
        Args:
            memory_threshold: Threshold for determining memory-intensive layers (0.0-1.0)
        """
        self.memory_threshold = memory_threshold
        self.checkpointed_modules = set()
        self.module_memory_usage = {}
        self.enabled = False
        
        if TORCH_AVAILABLE:
            print(f"✅ SelectiveActivationCheckpointing initialized with threshold: {memory_threshold}")
        else:
            print("⚠️  PyTorch not available, selective checkpointing disabled")
    
    def estimate_memory_usage(self, module, input_shape):
        """
        Estimate memory usage of a module
        
        Args:
            module: Module to estimate memory usage for
            input_shape: Shape of input tensor
            
        Returns:
            Estimated memory usage in MB
        """
        if not TORCH_AVAILABLE:
            return 0.0
            
        try:
            # Estimate based on parameter count and input/output sizes
            param_memory = sum(p.numel() * p.element_size() for p in module.parameters())
            
            # Estimate activation memory (simplified)
            activation_memory = 1
            for dim in input_shape:
                activation_memory *= dim
            activation_memory *= 4  # Assuming float32
            
            total_memory = param_memory + activation_memory
            return total_memory / (1024 * 1024)  # Convert to MB
        except:
            return 0.0
    
    def identify_memory_intensive_layers(self, model, input_shape):
        """
        Identify memory-intensive layers in the model
        
        Args:
            model: Model to analyze
            input_shape: Shape of input tensor
            
        Returns:
            List of names of memory-intensive modules
        """
        if not TORCH_AVAILABLE:
            return []
            
        memory_intensive_layers = []
        max_memory = 0
        
        # First pass: estimate memory usage for all modules
        for name, module in model.named_modules():
            if len(list(module.children())) == 0:  # Leaf modules only
                memory_usage = self.estimate_memory_usage(module, input_shape)
                self.module_memory_usage[name] = memory_usage
                max_memory = max(max_memory, memory_usage)
        
        # Second pass: identify layers above threshold
        threshold = self.memory_threshold * max_memory if max_memory > 0 else float('inf')
        
        for name, memory_usage in self.module_memory_usage.items():
            if memory_usage >= threshold:
                memory_intensive_layers.append(name)
                print(f"🔍 Memory-intensive layer identified: {name} ({memory_usage:.2f} MB)")
        
        return memory_intensive_layers
    
    def apply_selective_checkpointing(self, model, input_shape):
        """
        Apply selective checkpointing to memory-intensive layers
        
        Args:
            model: Model to apply checkpointing to
            input_shape: Shape of input tensor
            
        Returns:
            Model with selective checkpointing applied
        """
        if not TORCH_AVAILABLE:
            return model
            
        print("⚙️  Applying selective activation checkpointing...")
        
        # Identify memory-intensive layers
        memory_intensive_layers = self.identify_memory_intensive_layers(model, input_shape)
        self.checkpointed_modules = set(memory_intensive_layers)
        
        if not memory_intensive_layers:
            print("⚠️  No memory-intensive layers identified for checkpointing")
            return model
        
        # Apply checkpointing wrapper to identified layers
        for name, module in model.named_modules():
            if name in memory_intensive_layers:
                # Wrap the module with checkpointing
                wrapped_module = CheckpointedModuleWrapper(module)
                # Replace the module in the model
                self._replace_module(model, name, wrapped_module)
                print(f"✅ Applied checkpointing to: {name}")
        
        self.enabled = True
        print(f"✅ Selective activation checkpointing applied to {len(memory_intensive_layers)} layers")
        return model
    
    def _replace_module(self, model, module_name, new_module):
        """
        Replace a module in the model with a new module
        
        Args:
            model: Model containing the module
            module_name: Name of the module to replace
            new_module: New module to replace with
        """
        if not TORCH_AVAILABLE:
            return
            
        # Split the module name to get parent and child names
        name_parts = module_name.split('.')
        parent = model
        for part in name_parts[:-1]:
            parent = getattr(parent, part)
        setattr(parent, name_parts[-1], new_module)
    
    def get_checkpointing_stats(self) -> dict:
        """
        Get statistics about applied checkpointing
        
        Returns:
            Dictionary with checkpointing statistics
        """
        return {
            "enabled": self.enabled,
            "checkpointed_modules_count": len(self.checkpointed_modules),
            "checkpointed_modules": list(self.checkpointed_modules),
            "module_memory_usage": self.module_memory_usage
        }
    
    def disable(self):
        """Disable selective activation checkpointing"""
        self.enabled = False
        print("🚫 Selective activation checkpointing disabled")


class CheckpointedModuleWrapper(nn.Module):
    """Wrapper for modules that applies checkpointing during forward pass"""
    
    def __init__(self, module):
        """
        Initialize checkpointed module wrapper
        
        Args:
            module: Module to wrap with checkpointing
        """
        super().__init__()
        self.module = module
        
    def forward(self, *args, **kwargs):
        """
        Forward pass with checkpointing
        """
        if not TORCH_AVAILABLE:
            return self.module(*args, **kwargs)
            
        # Apply checkpointing if available and in training mode
        if torch.is_grad_enabled() and self.training and checkpoint is not None:
            try:
                # Create a function that only takes the module and its inputs
                def module_fn(module, *inputs, **kwinputs):
                    return module(*inputs, **kwinputs)
                
                # Apply checkpointing
                return checkpoint(module_fn, self.module, *args, **kwargs, use_reentrant=False)
            except Exception as e:
                # Fallback to regular forward pass if checkpointing fails
                print(f"⚠️  Checkpointing failed, using regular forward pass: {e}")
                return self.module(*args, **kwargs)
        else:
            # Regular forward pass during evaluation or if checkpointing not available
            return self.module(*args, **kwargs)


def demo_selective_checkpointing():
    """Demonstrate selective activation checkpointing"""
    if not TORCH_AVAILABLE:
        print("❌ PyTorch not available for selective checkpointing demo")
        return
        
    print("🚀 Demonstrating Selective Activation Checkpointing...")
    print("=" * 60)
    
    # Create a sample model with varying layer sizes
    class SampleModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.layer1 = nn.Linear(128, 256)
            self.layer2 = nn.Linear(256, 512)  # Memory-intensive
            self.layer3 = nn.Linear(512, 256)
            self.layer4 = nn.Linear(256, 1024)  # Memory-intensive
            self.layer5 = nn.Linear(1024, 128)
            self.activation = nn.ReLU()
            
        def forward(self, x):
            x = self.activation(self.layer1(x))
            x = self.activation(self.layer2(x))
            x = self.activation(self.layer3(x))
            x = self.activation(self.layer4(x))
            x = self.layer5(x)
            return x
    
    # Create model and checkpointing system
    model = SampleModel()
    checkpointing = SelectiveActivationCheckpointing(memory_threshold=0.7)
    
    print("✅ Created sample model and checkpointing system")
    
    # Apply selective checkpointing
    input_shape = (32, 128)  # Batch size 32, feature size 128
    checkpointed_model = checkpointing.apply_selective_checkpointing(model, input_shape)
    
    # Show statistics
    stats = checkpointing.get_checkpointing_stats()
    print(f"✅ Checkpointing statistics:")
    print(f"   Enabled: {stats['enabled']}")
    print(f"   Checkpointed modules: {stats['checkpointed_modules_count']}")
    for module_name in stats['checkpointed_modules']:
        memory_usage = stats['module_memory_usage'].get(module_name, 0)
        print(f"   - {module_name}: {memory_usage:.2f} MB")
    
    # Test forward pass
    dummy_input = torch.randn(32, 128)
    try:
        output = checkpointed_model(dummy_input)
        print(f"✅ Forward pass successful: {output.shape}")
    except Exception as e:
        print(f"❌ Forward pass failed: {e}")
    
    print("\n" + "=" * 60)
    print("SELECTIVE ACTIVATION CHECKPOINTING DEMO SUMMARY")
    print("=" * 60)
    print("Key Features Implemented:")
    print("  1. Memory usage estimation for model layers")
    print("  2. Automatic identification of memory-intensive layers")
    print("  3. Selective checkpointing application")
    print("  4. Runtime memory optimization")
    print("  5. Fallback mechanisms for robustness")
    print("\nBenefits:")
    print("  - Reduced memory footprint during training")
    print("  - Selective recomputation of only expensive layers")
    print("  - Automatic optimization based on layer characteristics")
    print("  - Graceful degradation when checkpointing fails")
    
    print("\n✅ Selective Activation Checkpointing demonstration completed!")


if __name__ == "__main__":
    demo_selective_checkpointing()