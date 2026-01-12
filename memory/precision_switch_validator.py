"""
Precision Switch Validator for MAHIA-X
This module measures FP32 vs FP16 vs FP8 divergence to validate precision switching.
"""

import time
from typing import Dict, Any, Optional, List, Tuple
from collections import defaultdict

# Conditional imports with fallbacks
TORCH_AVAILABLE = False
torch = None
nn = None

NUMPY_AVAILABLE = False
np = None

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

class PrecisionSwitchValidator:
    """Precision Switch Validator for measuring FP32 vs FP16 vs FP8 divergence"""
    
    def __init__(self, tolerance_fp32_fp16: float = 1e-3, tolerance_fp32_fp8: float = 1e-2):
        self.tolerance_fp32_fp16 = tolerance_fp32_fp16
        self.tolerance_fp32_fp8 = tolerance_fp32_fp8
        self.validation_results = []
        self.lock = threading.Lock() if 'threading' in globals() else None
        
        print(f"🧮 PrecisionSwitchValidator initialized with tolerances:")
        print(f"   FP32 vs FP16: {tolerance_fp32_fp16}")
        print(f"   FP32 vs FP8: {tolerance_fp32_fp8}")
        
    def calculate_divergence(self, tensor1: Any, tensor2: Any, method: str = "mse") -> float:
        """
        Calculate divergence between two tensors using specified method.
        
        Args:
            tensor1: First tensor (reference)
            tensor2: Second tensor (comparison)
            method: Method to use ('mse', 'mae', 'max_diff')
            
        Returns:
            float: Divergence value
        """
        if not TORCH_AVAILABLE or torch is None:
            # Fallback to NumPy if PyTorch is not available
            if not NUMPY_AVAILABLE or np is None:
                return 0.0
                
            # Convert to NumPy arrays if needed
            if hasattr(tensor1, 'numpy'):
                arr1 = tensor1.numpy() if hasattr(tensor1, 'numpy') else np.array(tensor1)
            else:
                arr1 = np.array(tensor1)
                
            if hasattr(tensor2, 'numpy'):
                arr2 = tensor2.numpy() if hasattr(tensor2, 'numpy') else np.array(tensor2)
            else:
                arr2 = np.array(tensor2)
            
            # Calculate divergence
            if method == "mse":
                return float(np.mean((arr1 - arr2) ** 2))
            elif method == "mae":
                return float(np.mean(np.abs(arr1 - arr2)))
            elif method == "max_diff":
                return float(np.max(np.abs(arr1 - arr2)))
            else:
                return float(np.mean((arr1 - arr2) ** 2))
        
        # Use PyTorch if available
        if not isinstance(tensor1, torch.Tensor):
            tensor1 = torch.tensor(tensor1, dtype=torch.float32)
        if not isinstance(tensor2, torch.Tensor):
            tensor2 = torch.tensor(tensor2, dtype=torch.float32)
            
        # Ensure same device and dtype for comparison
        if tensor1.device != tensor2.device:
            tensor2 = tensor2.to(tensor1.device)
            
        # Calculate divergence
        if method == "mse":
            return float(torch.mean((tensor1 - tensor2) ** 2).item())
        elif method == "mae":
            return float(torch.mean(torch.abs(tensor1 - tensor2)).item())
        elif method == "max_diff":
            return float(torch.max(torch.abs(tensor1 - tensor2)).item())
        else:
            return float(torch.mean((tensor1 - tensor2) ** 2).item())
    
    def validate_precision_switch(self, fp32_result: Any, fp16_result: Any, 
                                 fp8_result: Any = None) -> Dict[str, Any]:
        """
        Validate precision switching by comparing results across different precisions.
        
        Args:
            fp32_result: Result from FP32 computation (reference)
            fp16_result: Result from FP16 computation
            fp8_result: Result from FP8 computation (optional)
            
        Returns:
            Dict containing validation results
        """
        timestamp = time.time()
        
        # Calculate divergences
        fp32_fp16_divergence = self.calculate_divergence(fp32_result, fp16_result, "mse")
        fp32_fp16_mae = self.calculate_divergence(fp32_result, fp16_result, "mae")
        fp32_fp16_max_diff = self.calculate_divergence(fp32_result, fp16_result, "max_diff")
        
        validation_result = {
            "timestamp": timestamp,
            "fp32_fp16": {
                "mse": fp32_fp16_divergence,
                "mae": fp32_fp16_mae,
                "max_diff": fp32_fp16_max_diff,
                "within_tolerance": fp32_fp16_divergence <= self.tolerance_fp32_fp16
            }
        }
        
        # Add FP8 validation if provided
        if fp8_result is not None:
            fp32_fp8_divergence = self.calculate_divergence(fp32_result, fp8_result, "mse")
            fp32_fp8_mae = self.calculate_divergence(fp32_result, fp8_result, "mae")
            fp32_fp8_max_diff = self.calculate_divergence(fp32_result, fp8_result, "max_diff")
            
            validation_result["fp32_fp8"] = {
                "mse": fp32_fp8_divergence,
                "mae": fp32_fp8_mae,
                "max_diff": fp32_fp8_max_diff,
                "within_tolerance": fp32_fp8_divergence <= self.tolerance_fp32_fp8
            }
        
        # Store result
        if self.lock:
            with self.lock:
                self.validation_results.append(validation_result)
                
                # Keep only last 1000 entries to prevent memory bloat
                if len(self.validation_results) > 1000:
                    self.validation_results = self.validation_results[-1000:]
        else:
            self.validation_results.append(validation_result)
            
            # Keep only last 1000 entries to prevent memory bloat
            if len(self.validation_results) > 1000:
                self.validation_results = self.validation_results[-1000:]
        
        # Print summary
        print(f"🧮 Precision Validation Results:")
        print(f"   FP32 vs FP16 - MSE: {fp32_fp16_divergence:.6f} "
              f"(tolerance: {self.tolerance_fp32_fp16}) - "
              f"{'✅ PASS' if fp32_fp16_divergence <= self.tolerance_fp32_fp16 else '❌ FAIL'}")
              
        if fp8_result is not None:
            print(f"   FP32 vs FP8  - MSE: {fp32_fp8_divergence:.6f} "
                  f"(tolerance: {self.tolerance_fp32_fp8}) - "
                  f"{'✅ PASS' if fp32_fp8_divergence <= self.tolerance_fp32_fp8 else '❌ FAIL'}")
        else:
            fp32_fp8_divergence = 0.0
        
        return validation_result
    
    def get_validation_results(self) -> List[Dict[str, Any]]:
        """Get all validation results"""
        if self.lock:
            with self.lock:
                return self.validation_results.copy()
        else:
            return self.validation_results.copy()
    
    def clear_validation_results(self):
        """Clear all validation results"""
        if self.lock:
            with self.lock:
                self.validation_results.clear()
        else:
            self.validation_results.clear()
        print("🗑️  Validation results cleared")
    
    def generate_report(self) -> str:
        """Generate a summary report of precision validation results"""
        results = self.get_validation_results()
        if not results:
            return "No validation results available"
        
        total_validations = len(results)
        fp32_fp16_pass = sum(1 for r in results if r["fp32_fp16"]["within_tolerance"])
        fp32_fp16_fail = total_validations - fp32_fp16_pass
        
        report = f"""
🧮 Precision Switch Validation Report
================================
Total Validations: {total_validations}
FP32 vs FP16:
  Passed: {fp32_fp16_pass} ({fp32_fp16_pass/total_validations*100:.1f}%)
  Failed: {fp32_fp16_fail} ({fp32_fp16_fail/total_validations*100:.1f}%)

Tolerance Settings:
  FP32 vs FP16: {self.tolerance_fp32_fp16}
"""
        
        # Add FP8 stats if available
        fp8_results = [r for r in results if "fp32_fp8" in r]
        if fp8_results:
            fp32_fp8_pass = sum(1 for r in fp8_results if r["fp32_fp8"]["within_tolerance"])
            fp32_fp8_fail = len(fp8_results) - fp32_fp8_pass
            report += f"FP32 vs FP8:\n  Passed: {fp32_fp8_pass} ({fp32_fp8_pass/len(fp8_results)*100:.1f}%)\n  Failed: {fp32_fp8_fail} ({fp32_fp8_fail/len(fp8_results)*100:.1f}%)\n"
            report += f"Tolerance Settings:\n  FP32 vs FP8: {self.tolerance_fp32_fp8}\n"
        
        # Show recent validations
        recent_results = results[-5:] if len(results) >= 5 else results
        report += f"\nRecent Validations:\n"
        for i, result in enumerate(recent_results):
            report += f"  {i+1}. {time.ctime(result['timestamp'])}\n"
            report += f"     FP32 vs FP16: MSE={result['fp32_fp16']['mse']:.6f} "
            report += f"({'PASS' if result['fp32_fp16']['within_tolerance'] else 'FAIL'})\n"
            if "fp32_fp8" in result:
                report += f"     FP32 vs FP8:  MSE={result['fp32_fp8']['mse']:.6f} "
                report += f"({'PASS' if result['fp32_fp8']['within_tolerance'] else 'FAIL'})\n"
        
        return report

# Global instance
_precision_validator = None

def get_precision_validator() -> PrecisionSwitchValidator:
    """Get the global precision switch validator instance"""
    global _precision_validator
    if _precision_validator is None:
        _precision_validator = PrecisionSwitchValidator()
    return _precision_validator

# Import threading at the end to avoid circular import issues
import threading

if __name__ == "__main__":
    # Example usage
    validator = get_precision_validator()
    
    # Simulate some validation results
    # In practice, these would be actual tensor results from different precisions
    fp32_result = [1.0, 2.0, 3.0, 4.0, 5.0]
    fp16_result = [1.0001, 2.0002, 2.9998, 4.0003, 4.9999]
    fp8_result = [1.01, 2.02, 2.98, 4.03, 4.99]
    
    validator.validate_precision_switch(fp32_result, fp16_result, fp8_result)
    
    # Print report
    print(validator.generate_report())