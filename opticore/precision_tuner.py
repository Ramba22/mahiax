"""
Precision Tuner for MAHIA OptiCore
Analysis of gradient entropy and stability with dynamic switching between precisions.
"""

import time
import threading
from typing import Dict, Any, Optional, List, Tuple
from collections import defaultdict, deque

# Conditional imports with fallbacks
TORCH_AVAILABLE = False
torch = None
NUMPY_AVAILABLE = False
np = None

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    pass

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    pass

class PrecisionTuner:
    """Dynamic precision tuning based on gradient analysis"""
    
    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self.gradient_history = deque(maxlen=window_size)
        self.loss_history = deque(maxlen=window_size)
        self.precision_history = deque(maxlen=window_size)
        self.stability_scores = deque(maxlen=window_size)
        self.current_precision = "fp32"  # Default precision
        self.safe_mode = False
        self.lock = threading.Lock()
        self.stats = {
            "precision_switches": 0,
            "safe_mode_activations": 0,
            "stability_violations": 0
        }
        
        print(f"🧮 PrecisionTuner initialized with window size: {window_size}")
        print(f"   Default precision: {self.current_precision}")
        
    def analyze_gradients(self, gradients: List[Any]) -> Dict[str, float]:
        """
        Analyze gradients for entropy and stability.
        
        Args:
            gradients: List of gradient tensors
            
        Returns:
            Dict with analysis metrics
        """
        if not gradients:
            return {"entropy": 0.0, "norm": 0.0, "stability": 1.0}
            
        # Calculate gradient norm
        grad_norm = 0.0
        grad_elements = 0
        
        for grad in gradients:
            if TORCH_AVAILABLE and torch is not None and isinstance(grad, torch.Tensor):
                grad_norm += torch.norm(grad).item()
                grad_elements += grad.nelement()
            elif NUMPY_AVAILABLE and np is not None and hasattr(grad, 'shape'):
                grad_norm += np.linalg.norm(grad)
                grad_elements += grad.size
            else:
                # Fallback for other types
                try:
                    grad_norm += float(sum(x**2 for x in grad)) ** 0.5
                    grad_elements += len(grad)
                except:
                    pass
        
        # Calculate entropy (simplified)
        avg_grad_magnitude = grad_norm / grad_elements if grad_elements > 0 else 0.0
        entropy = self._calculate_entropy(avg_grad_magnitude)
        
        # Calculate stability score
        stability = self._calculate_stability(avg_grad_magnitude)
        
        analysis = {
            "entropy": entropy,
            "norm": grad_norm,
            "stability": stability,
            "avg_magnitude": avg_grad_magnitude
        }
        
        # Store in history
        with self.lock:
            self.gradient_history.append(analysis)
            
        return analysis
        
    def _calculate_entropy(self, avg_magnitude: float) -> float:
        """
        Calculate gradient entropy (simplified).
        
        Args:
            avg_magnitude: Average gradient magnitude
            
        Returns:
            float: Entropy value
        """
        # Simplified entropy calculation
        if avg_magnitude <= 0:
            return 0.0
        return min(1.0, max(0.0, 1.0 - avg_magnitude))  # Inverse relationship
        
    def _calculate_stability(self, avg_magnitude: float) -> float:
        """
        Calculate stability score.
        
        Args:
            avg_magnitude: Average gradient magnitude
            
        Returns:
            float: Stability score (0.0 to 1.0)
        """
        # Simple stability measure - lower magnitude = more stable
        return max(0.0, 1.0 - avg_magnitude)
        
    def tune_precision(self, loss: float, gradients: List[Any], 
                      force_precision: Optional[str] = None) -> str:
        """
        Tune precision based on loss and gradient analysis.
        
        Args:
            loss: Current loss value
            gradients: List of gradient tensors
            force_precision: Force specific precision (optional)
            
        Returns:
            str: Recommended precision
        """
        with self.lock:
            # Store loss
            self.loss_history.append(loss)
            
            # If forced precision, use it
            if force_precision:
                if force_precision != self.current_precision:
                    self.stats["precision_switches"] += 1
                    print(f"🔧 Forced precision change: {self.current_precision} → {force_precision}")
                self.current_precision = force_precision
                self.precision_history.append(force_precision)
                return force_precision
                
            # Analyze gradients
            analysis = self.analyze_gradients(gradients)
            
            # Check if we should enter safe mode
            if self._should_enter_safe_mode(analysis, loss):
                if not self.safe_mode:
                    self.safe_mode = True
                    self.stats["safe_mode_activations"] += 1
                    print("🛡️  Entering safe mode - switching to FP32")
                self.current_precision = "fp32"
            else:
                # Normal precision tuning
                self.safe_mode = False
                new_precision = self._select_precision(analysis, loss)
                if new_precision != self.current_precision:
                    self.stats["precision_switches"] += 1
                    print(f"🔄 Precision change: {self.current_precision} → {new_precision} "
                          f"(entropy: {analysis['entropy']:.3f}, stability: {analysis['stability']:.3f})")
                self.current_precision = new_precision
                
            self.precision_history.append(self.current_precision)
            return self.current_precision
            
    def _should_enter_safe_mode(self, analysis: Dict[str, float], loss: float) -> bool:
        """
        Determine if safe mode should be activated.
        
        Args:
            analysis: Gradient analysis results
            loss: Current loss value
            
        Returns:
            bool: True if safe mode should be activated
        """
        # Check for instability
        if analysis["stability"] < 0.3:
            self.stats["stability_violations"] += 1
            return True
            
        # Check for exploding gradients
        if analysis["norm"] > 100.0:
            self.stats["stability_violations"] += 1
            return True
            
        # Check for loss explosion
        if len(self.loss_history) > 10:
            recent_losses = list(self.loss_history)[-10:]
            if loss > max(recent_losses) * 2.0:
                self.stats["stability_violations"] += 1
                return True
                
        return False
        
    def _select_precision(self, analysis: Dict[str, float], loss: float) -> str:
        """
        Select precision based on analysis.
        
        Args:
            analysis: Gradient analysis results
            loss: Current loss value
            
        Returns:
            str: Selected precision
        """
        entropy = analysis["entropy"]
        stability = analysis["stability"]
        
        # High entropy (complex gradients) -> higher precision
        if entropy > 0.7:
            return "fp32"
        # Medium entropy and good stability -> medium precision
        elif entropy > 0.3 and stability > 0.7:
            return "fp16"
        # Low entropy and good stability -> lower precision
        elif stability > 0.8:
            return "fp8"
        # Default to FP16 for balanced performance
        else:
            return "fp16"
            
    def get_current_precision(self) -> str:
        """Get current precision setting"""
        with self.lock:
            return self.current_precision
            
    def is_safe_mode(self) -> bool:
        """Check if safe mode is active"""
        with self.lock:
            return self.safe_mode
            
    def get_stats(self) -> Dict[str, Any]:
        """Get precision tuner statistics"""
        with self.lock:
            return self.stats.copy()
            
    def clear_stats(self):
        """Clear statistics"""
        with self.lock:
            self.stats = {
                "precision_switches": 0,
                "safe_mode_activations": 0,
                "stability_violations": 0
            }
        print("🗑️  Precision tuner statistics cleared")
        
    def get_precision_history(self) -> List[str]:
        """Get precision change history"""
        with self.lock:
            return list(self.precision_history)

# Global instance
_precision_tuner = None

def get_precision_tuner() -> PrecisionTuner:
    """Get the global precision tuner instance"""
    global _precision_tuner
    if _precision_tuner is None:
        _precision_tuner = PrecisionTuner()
    return _precision_tuner

if __name__ == "__main__":
    # Example usage
    tuner = get_precision_tuner()
    
    # Simulate some gradients and loss values
    sample_gradients = [[0.1, 0.2, 0.3], [0.05, 0.15, 0.25]]
    sample_loss = 0.5
    
    # Tune precision
    precision = tuner.tune_precision(sample_loss, sample_gradients)
    print(f"Recommended precision: {precision}")
    
    # Force precision change
    precision = tuner.tune_precision(sample_loss, sample_gradients, force_precision="fp32")
    print(f"Forced precision: {precision}")
    
    # Print stats
    print(f"📊 Stats: {tuner.get_stats()}")