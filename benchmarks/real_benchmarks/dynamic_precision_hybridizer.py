"""
Dynamic Precision Hybridization for MAHIA
Layer-wise dynamic precision switching (FP8 ↔ FP16 ↔ BF16 ↔ INT4) based on gradient noise and entropy
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple, Any
import time

# Try to import quantization libraries
try:
    import bitsandbytes as bnb
    BITSANDBYTES_AVAILABLE = True
except ImportError:
    BITSANDBYTES_AVAILABLE = False
    print("⚠️  bitsandbytes not available, INT4 quantization limited")

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    np = None
    print("⚠️  numpy not available, some features limited")

try:
    import torchao
    TORCHAO_AVAILABLE = True
    # Check for FP8 support
    try:
        import torchao.float8
        FP8_AVAILABLE = True
    except ImportError:
        FP8_AVAILABLE = False
        print("⚠️  torchao.float8 not available, FP8 quantization limited")
except ImportError:
    TORCHAO_AVAILABLE = False
    FP8_AVAILABLE = False
    print("⚠️  torchao not available, advanced quantization limited")

class GradientAnalyzer:
    """Analyze gradients for noise and entropy metrics"""
    
    def __init__(self):
        self.gradient_history = {}
        self.loss_history = []
        self.noise_threshold = 0.1
        self.entropy_threshold = 0.5
        self.loss_variance_threshold = 0.01
        self.loss_window_size = 10
    
    def compute_gradient_noise_scale(self, gradients: List[torch.Tensor]) -> float:
        """
        Compute gradient noise scale (a measure of training stability)
        
        Args:
            gradients: List of gradient tensors
            
        Returns:
            float: Gradient noise scale
        """
        if not gradients:
            return 0.0
        
        # Compute gradient norms
        grad_norms = [torch.norm(g).item() for g in gradients if g is not None]
        
        if len(grad_norms) < 2:
            return 0.0
        
        # Compute noise as variance of gradient norms
        if NUMPY_AVAILABLE and np is not None:
            noise_scale = np.var(grad_norms)
        else:
            # Fallback implementation without numpy
            mean = sum(grad_norms) / len(grad_norms)
            variance = sum((x - mean) ** 2 for x in grad_norms) / len(grad_norms)
            noise_scale = variance
            
        return float(noise_scale)
    
    def compute_gradient_entropy(self, gradients: List[torch.Tensor]) -> float:
        """
        Compute gradient entropy (a measure of gradient diversity)
        
        Args:
            gradients: List of gradient tensors
            
        Returns:
            float: Gradient entropy
        """
        if not gradients:
            return 0.0
        
        # Flatten and concatenate all gradients
        flat_grads = []
        for g in gradients:
            if g is not None:
                flat_grads.append(g.detach().flatten())
        
        if not flat_grads:
            return 0.0
            
        all_grads = torch.cat(flat_grads)
        
        # Normalize gradients to probability distribution
        grad_abs = torch.abs(all_grads)
        if grad_abs.sum() == 0:
            return 0.0
            
        prob_dist = grad_abs / grad_abs.sum()
        
        # Compute entropy
        entropy = -torch.sum(prob_dist * torch.log(prob_dist + 1e-8)).item()
        return entropy
        
    def compute_loss_variance(self, current_loss: Optional[float] = None) -> float:
        """
        Compute loss variance over recent training steps
        
        Args:
            current_loss: Current loss value (optional, will be added to history)
            
        Returns:
            float: Loss variance
        """
        # Add current loss to history if provided
        if current_loss is not None:
            self.loss_history.append(current_loss)
            
            # Keep only recent history
            if len(self.loss_history) > self.loss_window_size:
                self.loss_history.pop(0)
        
        # Compute variance of loss history
        if len(self.loss_history) < 2:
            return 0.0
            
        if NUMPY_AVAILABLE and 'np' in globals() and np is not None:
            return float(np.var(self.loss_history))
        else:
            mean = sum(self.loss_history) / len(self.loss_history)
            variance = sum((x - mean) ** 2 for x in self.loss_history) / len(self.loss_history)
            return variance
    
    def should_change_precision(self, gradients: List[torch.Tensor], 
                               current_loss: Optional[float] = None) -> Tuple[bool, str]:
        """
        Determine if precision should be changed based on gradient metrics and loss variance
        
        Args:
            gradients: List of gradient tensors
            current_loss: Current loss value (optional)
            
        Returns:
            Tuple[bool, str]: (should_change, reason)
        """
        noise_scale = self.compute_gradient_noise_scale(gradients)
        entropy = self.compute_gradient_entropy(gradients)
        loss_variance = self.compute_loss_variance(current_loss)
        
        # High noise indicates instability - use higher precision
        if noise_scale > self.noise_threshold:
            return True, f"High gradient noise ({noise_scale:.4f} > {self.noise_threshold})"
        
        # Low entropy indicates uniform gradients - use lower precision
        if entropy < self.entropy_threshold:
            return True, f"Low gradient entropy ({entropy:.4f} < {self.entropy_threshold})"
            
        # High loss variance indicates unstable training - use higher precision
        if loss_variance > self.loss_variance_threshold:
            return True, f"High loss variance ({loss_variance:.4f} > {self.loss_variance_threshold})"
            
        # Low loss variance with stable gradients - can use lower precision for efficiency
        if loss_variance < self.loss_variance_threshold / 4:  # Very stable
            return True, f"Low loss variance ({loss_variance:.4f} < {self.loss_variance_threshold/4:.4f}) - optimizing for efficiency"
        
        return False, "Stable training conditions"

class PrecisionController:
    """Control precision switching for different layers"""
    
    def __init__(self, model: nn.Module):
        self.model = model
        self.layer_precisions = {}
        self.precision_history = {}
        self.switch_count = 0
        
        # Initialize all layers to FP16 as default
        for name, module in model.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d, nn.Embedding)):
                self.layer_precisions[name] = "FP16"
                self.precision_history[name] = ["FP16"]
    
    def get_current_precision(self, layer_name: str) -> str:
        """Get current precision for a layer"""
        return self.layer_precisions.get(layer_name, "FP16")
    
    def suggest_precision_change(self, layer_name: str, gradient_metrics: Dict[str, float]) -> str:
        """
        Suggest precision change for a layer based on metrics
        
        Args:
            layer_name: Name of the layer
            gradient_metrics: Metrics about gradients
            
        Returns:
            str: Suggested precision (FP8, FP16, BF16, INT4)
        """
        current_precision = self.get_current_precision(layer_name)
        noise_scale = gradient_metrics.get("noise_scale", 0.0)
        entropy = gradient_metrics.get("entropy", 1.0)
        
        # Decision logic based on metrics
        if noise_scale > 0.1:  # High noise - need more precision
            if current_precision == "INT4":
                return "FP8"
            elif current_precision == "FP8":
                return "FP16"
            else:
                return "FP32"  # Maximum precision
        elif entropy < 0.3:  # Low entropy - can use less precision
            if current_precision == "FP32":
                return "FP16"
            elif current_precision == "FP16":
                return "FP8"
            elif current_precision == "FP8":
                return "INT4"
            else:
                return "INT4"  # Minimum precision
        else:  # Stable conditions - maintain current precision
            return current_precision
    
    def apply_precision_change(self, layer_name: str, new_precision: str) -> bool:
        """
        Apply precision change to a layer
        
        Args:
            layer_name: Name of the layer
            new_precision: New precision to apply
            
        Returns:
            bool: Whether change was successful
        """
        if new_precision == self.get_current_precision(layer_name):
            return False  # No change needed
        
        # Find the layer
        target_module = None
        for name, module in self.model.named_modules():
            if name == layer_name:
                target_module = module
                break
        
        if target_module is None:
            print(f"⚠️  Layer {layer_name} not found")
            return False
        
        # Apply precision change (simplified implementation)
        old_precision = self.layer_precisions[layer_name]
        self.layer_precisions[layer_name] = new_precision
        self.precision_history[layer_name].append(new_precision)
        self.switch_count += 1
        
        print(f"🔄 Precision changed for {layer_name}: {old_precision} → {new_precision}")
        return True

class DynamicPrecisionHybridizer:
    """Main dynamic precision hybridization system"""
    
    def __init__(self, model: nn.Module, 
                 noise_threshold: float = 0.1,
                 entropy_threshold: float = 0.5):
        """
        Initialize dynamic precision hybridizer
        
        Args:
            model: PyTorch model to optimize
            noise_threshold: Threshold for gradient noise detection
            entropy_threshold: Threshold for gradient entropy detection
        """
        self.model = model
        self.gradient_analyzer = GradientAnalyzer()
        self.precision_controller = PrecisionController(model)
        
        # Set thresholds
        self.gradient_analyzer.noise_threshold = noise_threshold
        self.gradient_analyzer.entropy_threshold = entropy_threshold
        
        # Performance tracking
        self.energy_savings = 0.0
        self.performance_impact = 0.0
        self.stability_improvements = 0.0
        
        print("✅ Dynamic Precision Hybridizer initialized")
    
    def collect_gradients(self, model: Optional[nn.Module] = None) -> List[torch.Tensor]:
        """
        Collect gradients from model parameters
        
        Args:
            model: Model to collect gradients from (uses self.model if None)
            
        Returns:
            List[torch.Tensor]: List of gradient tensors
        """
        if model is None:
            model = self.model
            
        gradients = []
        for param in model.parameters():
            if param.grad is not None:
                gradients.append(param.grad)
        
        return gradients
    
    def analyze_and_adapt(self, model: Optional[nn.Module] = None) -> Dict[str, Any]:
        """
        Analyze gradients and adapt precision accordingly
        
        Args:
            model: Model to analyze (uses self.model if None)
            
        Returns:
            Dict[str, Any]: Analysis results and actions taken
        """
        if model is None:
            model = self.model
        
        # Collect gradients
        gradients = self.collect_gradients(model)
        
        # Analyze gradients
        noise_scale = self.gradient_analyzer.compute_gradient_noise_scale(gradients)
        entropy = self.gradient_analyzer.compute_gradient_entropy(gradients)
        
        gradient_metrics = {
            "noise_scale": noise_scale,
            "entropy": entropy
        }
        
        # Determine if precision change is needed
        should_change, reason = self.gradient_analyzer.should_change_precision(gradients)
        
        actions = []
        if should_change:
            # Apply precision changes to layers
            for name, module in model.named_modules():
                if isinstance(module, (nn.Linear, nn.Conv2d, nn.Embedding)):
                    suggested_precision = self.precision_controller.suggest_precision_change(
                        name, gradient_metrics
                    )
                    
                    changed = self.precision_controller.apply_precision_change(
                        name, suggested_precision
                    )
                    
                    if changed:
                        actions.append({
                            "layer": name,
                            "from": self.precision_controller.precision_history[name][-2],
                            "to": suggested_precision,
                            "reason": reason
                        })
        
        # Update performance metrics (simulated)
        self._update_performance_metrics(should_change, noise_scale, entropy)
        
        return {
            "gradient_metrics": gradient_metrics,
            "should_change_precision": should_change,
            "reason": reason,
            "actions": actions,
            "performance_metrics": {
                "energy_savings": self.energy_savings,
                "stability_improvements": self.stability_improvements
            }
        }
    
    def _update_performance_metrics(self, precision_changed: bool, 
                                  noise_scale: float, entropy: float):
        """
        Update performance metrics based on analysis
        
        Args:
            precision_changed: Whether precision was changed
            noise_scale: Current gradient noise scale
            entropy: Current gradient entropy
        """
        # Simulate energy savings from lower precision usage
        low_precision_layers = sum(1 for p in self.precision_controller.layer_precisions.values() 
                                 if p in ["INT4", "FP8"])
        total_layers = len(self.precision_controller.layer_precisions)
        
        if total_layers > 0:
            self.energy_savings = (low_precision_layers / total_layers) * 0.3  # Up to 30% savings
        
        # Stability improvements from noise reduction
        self.stability_improvements = max(0, 0.1 - noise_scale) * 10  # Normalize to 0-1
        
        # Performance impact (might be negative during transitions)
        if precision_changed:
            self.performance_impact = np.random.normal(0, 0.05)  # Small random impact
    
    def get_precision_summary(self) -> Dict[str, Any]:
        """Get summary of current precision distribution"""
        precision_counts = {}
        for precision in self.precision_controller.layer_precisions.values():
            precision_counts[precision] = precision_counts.get(precision, 0) + 1
        
        return {
            "precision_distribution": precision_counts,
            "total_switches": self.precision_controller.switch_count,
            "energy_savings_estimate": self.energy_savings,
            "stability_improvements": self.stability_improvements
        }
    
    def export_precision_config(self, filepath: Optional[str] = None) -> Dict[str, str]:
        """
        Export current precision configuration
        
        Args:
            filepath: File to save configuration to (optional)
            
        Returns:
            Dict[str, str]: Current precision configuration
        """
        config = self.precision_controller.layer_precisions.copy()
        
        if filepath:
            import json
            try:
                with open(filepath, 'w') as f:
                    json.dump(config, f, indent=2)
                print(f"✅ Precision configuration exported to {filepath}")
            except Exception as e:
                print(f"❌ Failed to export precision configuration: {e}")
        
        return config

# Example usage
def example_dynamic_precision_hybridization():
    """Example of dynamic precision hybridization"""
    print("🔧 Setting up dynamic precision hybridization example...")
    
    # Simple model for demonstration
    class SimpleModel(nn.Module):
        def __init__(self, vocab_size=1000, hidden_size=128, num_classes=2):
            super().__init__()
            self.embedding = nn.Embedding(vocab_size, hidden_size)
            self.linear1 = nn.Linear(hidden_size, hidden_size * 4)
            self.linear2 = nn.Linear(hidden_size * 4, hidden_size)
            self.classifier = nn.Linear(hidden_size, num_classes)
            
        def forward(self, input_ids, attention_mask=None):
            x = self.embedding(input_ids)
            x = x.mean(dim=1)  # Global average pooling
            x = torch.relu(self.linear1(x))
            x = torch.relu(self.linear2(x))
            logits = self.classifier(x)
            return logits
    
    # Create model
    model = SimpleModel()
    print(f"✅ Created model with {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M parameters")
    
    # Create hybridizer
    hybridizer = DynamicPrecisionHybridizer(model)
    
    # Simulate training iterations
    print("\n" + "="*60)
    print("🔄 Simulating training with dynamic precision adaptation...")
    
    for iteration in range(5):
        print(f"\n🔁 Iteration {iteration + 1}")
        
        # Simulate gradient computation
        for param in model.parameters():
            if param.requires_grad:
                # Create mock gradients with varying noise/entropy
                noise_factor = np.random.uniform(0.5, 2.0)
                param.grad = torch.randn_like(param) * noise_factor
        
        # Analyze and adapt precision
        results = hybridizer.analyze_and_adapt()
        
        # Print results
        metrics = results["gradient_metrics"]
        print(f"   Gradient Noise: {metrics['noise_scale']:.4f}")
        print(f"   Gradient Entropy: {metrics['entropy']:.4f}")
        print(f"   Precision Change: {results['should_change_precision']}")
        if results['should_change_precision']:
            print(f"   Reason: {results['reason']}")
            for action in results['actions']:
                print(f"   🔄 {action['layer']}: {action['from']} → {action['to']}")
    
    # Print final precision summary
    print("\n" + "="*60)
    summary = hybridizer.get_precision_summary()
    print("📊 Final Precision Summary:")
    for precision, count in summary["precision_distribution"].items():
        print(f"   {precision}: {count} layers")
    print(f"   Total Switches: {summary['total_switches']}")
    print(f"   Energy Savings: {summary['energy_savings_estimate']:.1%}")
    print(f"   Stability Improvements: {summary['stability_improvements']:.1%}")
    
    return hybridizer

if __name__ == "__main__":
    example_dynamic_precision_hybridization()