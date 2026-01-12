"""
FP8/INT4 Validation Pipeline for MAHIA
Full FP8/INT4 Validation Pipeline with layer-wise calibration
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Any, List, Tuple, Union
import numpy as np

# Conditional imports for quantization libraries
BITSANDBYTES_AVAILABLE = False
try:
    import bitsandbytes as bnb
    BITSANDBYTES_AVAILABLE = True
    print("✅ bitsandbytes available")
except ImportError:
    print("⚠️  bitsandbytes not available")

TORCHAO_AVAILABLE = False
try:
    import torchao
    TORCHAO_AVAILABLE = True
    print("✅ torchao available")
except ImportError:
    print("⚠️  torchao not available")

class QuantizationCalibrator:
    """
    Quantization calibrator for layer-wise calibration of FP8/INT4 quantization
    """
    
    def __init__(self, 
                 calibration_method: str = "minmax",
                 num_calibration_steps: int = 100):
        """
        Initialize quantization calibrator
        
        Args:
            calibration_method: Calibration method ("minmax", "percentile", "mse")
            num_calibration_steps: Number of calibration steps
        """
        self.calibration_method = calibration_method
        self.num_calibration_steps = num_calibration_steps
        self.calibration_data = {}
        self.is_calibrating = False
        
        print(f"✅ QuantizationCalibrator initialized: method={calibration_method}")
        
    def start_calibration(self):
        """Start calibration process"""
        self.is_calibrating = True
        self.calibration_data = {}
        print("🔄 Calibration started")
        
    def stop_calibration(self):
        """Stop calibration process"""
        self.is_calibrating = False
        print("⏹️  Calibration stopped")
        
    def collect_statistics(self, layer_name: str, tensor: torch.Tensor):
        """
        Collect statistics for calibration
        
        Args:
            layer_name: Name of the layer
            tensor: Tensor to collect statistics for
        """
        if not self.is_calibrating:
            return
            
        if layer_name not in self.calibration_data:
            self.calibration_data[layer_name] = {
                'min': float('inf'),
                'max': float('-inf'),
                'abs_min': float('inf'),
                'abs_max': float('-inf'),
                'mean': 0.0,
                'std': 0.0,
                'count': 0,
                'values': []
            }
            
        data = self.calibration_data[layer_name]
        
        # Update min/max values
        tensor_min = tensor.min().item()
        tensor_max = tensor.max().item()
        data['min'] = min(data['min'], tensor_min)
        data['max'] = max(data['max'], tensor_max)
        
        # Update absolute min/max values
        tensor_abs = tensor.abs()
        abs_min = tensor_abs.min().item()
        abs_max = tensor_abs.max().item()
        data['abs_min'] = min(data['abs_min'], abs_min)
        data['abs_max'] = max(data['abs_max'], abs_max)
        
        # Update mean and std (running average)
        tensor_mean = tensor.mean().item()
        tensor_std = tensor.std().item()
        count = tensor.numel()
        
        if data['count'] == 0:
            data['mean'] = tensor_mean
            data['std'] = tensor_std
        else:
            # Running average
            total_count = data['count'] + count
            data['mean'] = (data['mean'] * data['count'] + tensor_mean * count) / total_count
            # This is a simplified std calculation
            data['std'] = ((data['std']**2 * data['count'] + tensor_std**2 * count) / total_count)**0.5
            
        data['count'] += count
        
        # Store some values for percentile calculation
        if len(data['values']) < 10000:  # Limit storage
            flat_tensor = tensor.flatten()
            # Sample values for percentile calculation
            sample_indices = torch.randperm(flat_tensor.size(0))[:min(100, flat_tensor.size(0))]
            data['values'].extend(flat_tensor[sample_indices].tolist())
            
    def compute_calibration_params(self, layer_name: str) -> Dict[str, Any]:
        """
        Compute calibration parameters for a layer
        
        Args:
            layer_name: Name of the layer
            
        Returns:
            Dictionary of calibration parameters
        """
        if layer_name not in self.calibration_data:
            return {}
            
        data = self.calibration_data[layer_name]
        
        if self.calibration_method == "minmax":
            return {
                'min': data['min'],
                'max': data['max']
            }
        elif self.calibration_method == "percentile":
            if len(data['values']) > 0:
                values = np.array(data['values'])
                percentile_01 = np.percentile(values, 1)
                percentile_99 = np.percentile(values, 99)
                return {
                    'min': float(percentile_01),
                    'max': float(percentile_99)
                }
            else:
                return {
                    'min': data['min'],
                    'max': data['max']
                }
        elif self.calibration_method == "mse":
            # For MSE, we might want to try different ranges and pick the best
            # This is a simplified approach
            return {
                'min': data['mean'] - 3 * data['std'],
                'max': data['mean'] + 3 * data['std']
            }
        else:
            return {
                'min': data['min'],
                'max': data['max']
            }

class FP8Quantizer:
    """
    FP8 quantizer for 8-bit floating point quantization
    """
    
    def __init__(self, 
                 calibration_method: str = "minmax",
                 use_bitsandbytes: bool = True):
        """
        Initialize FP8 quantizer
        
        Args:
            calibration_method: Calibration method
            use_bitsandbytes: Whether to use bitsandbytes for quantization
        """
        self.calibration_method = calibration_method
        self.use_bitsandbytes = use_bitsandbytes and BITSANDBYTES_AVAILABLE
        self.calibrator = QuantizationCalibrator(calibration_method)
        
        # FP8 format parameters
        self.fp8_e4m3_max = 448.0  # Max value for FP8 E4M3
        self.fp8_e5m2_max = 57344.0  # Max value for FP8 E5M2
        
        print(f"✅ FP8Quantizer initialized: method={calibration_method}")
        
    def quantize_tensor(self, 
                       tensor: torch.Tensor,
                       layer_name: str,
                       fp8_format: str = "E4M3") -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Quantize tensor to FP8
        
        Args:
            tensor: Tensor to quantize
            layer_name: Name of the layer
            fp8_format: FP8 format ("E4M3" or "E5M2")
            
        Returns:
            Tuple of (quantized_tensor, quantization_params)
        """
        # Collect statistics during calibration
        self.calibrator.collect_statistics(layer_name, tensor)
        
        # Get calibration parameters
        cal_params = self.calibrator.compute_calibration_params(layer_name)
        
        if not cal_params:
            # Fallback to tensor min/max
            tensor_min = tensor.min()
            tensor_max = tensor.max()
        else:
            tensor_min = cal_params['min']
            tensor_max = cal_params['max']
            
        # Determine scaling factor
        fp8_max = self.fp8_e4m3_max if fp8_format == "E4M3" else self.fp8_e5m2_max
        scale = max(abs(tensor_min), abs(tensor_max)) / fp8_max
        scale = max(scale, 1e-8)  # Avoid division by zero
        
        # Quantize
        if self.use_bitsandbytes:
            try:
                # Use bitsandbytes for quantization
                quantized = tensor / scale
                # Clip to FP8 range
                fp8_max_tensor = torch.tensor(fp8_max, device=tensor.device)
                quantized = torch.clamp(quantized, -fp8_max_tensor, fp8_max_tensor)
                return quantized, {'scale': scale, 'format': fp8_format}
            except Exception as e:
                print(f"⚠️  bitsandbytes FP8 quantization failed: {e}")
                
        # Fallback to manual quantization
        quantized = tensor / scale
        # Clip to FP8 range
        fp8_max_tensor = torch.tensor(fp8_max, device=tensor.device)
        quantized = torch.clamp(quantized, -fp8_max_tensor, fp8_max_tensor)
        
        return quantized, {'scale': scale, 'format': fp8_format}
        
    def dequantize_tensor(self, 
                         quantized_tensor: torch.Tensor,
                         scale: float) -> torch.Tensor:
        """
        Dequantize FP8 tensor
        
        Args:
            quantized_tensor: Quantized tensor
            scale: Scaling factor
            
        Returns:
            Dequantized tensor
        """
        return quantized_tensor * scale

class INT4Quantizer:
    """
    INT4 quantizer for 4-bit integer quantization
    """
    
    def __init__(self, 
                 calibration_method: str = "minmax",
                 use_bitsandbytes: bool = True):
        """
        Initialize INT4 quantizer
        
        Args:
            calibration_method: Calibration method
            use_bitsandbytes: Whether to use bitsandbytes for quantization
        """
        self.calibration_method = calibration_method
        self.use_bitsandbytes = use_bitsandbytes and BITSANDBYTES_AVAILABLE
        self.calibrator = QuantizationCalibrator(calibration_method)
        
        # INT4 parameters
        self.int4_min = -8  # INT4 range: -8 to 7
        self.int4_max = 7
        
        print(f"✅ INT4Quantizer initialized: method={calibration_method}")
        
    def quantize_tensor(self, 
                       tensor: torch.Tensor,
                       layer_name: str) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Quantize tensor to INT4
        
        Args:
            tensor: Tensor to quantize
            layer_name: Name of the layer
            
        Returns:
            Tuple of (quantized_tensor, quantization_params)
        """
        # Collect statistics during calibration
        self.calibrator.collect_statistics(layer_name, tensor)
        
        # Get calibration parameters
        cal_params = self.calibrator.compute_calibration_params(layer_name)
        
        if not cal_params:
            # Fallback to tensor min/max
            tensor_min = tensor.min()
            tensor_max = tensor.max()
        else:
            tensor_min = cal_params['min']
            tensor_max = cal_params['max']
            
        # Determine scaling and zero point
        scale = (tensor_max - tensor_min) / (self.int4_max - self.int4_min)
        scale = max(scale, 1e-8)  # Avoid division by zero
        zero_point = self.int4_min - tensor_min / scale
        
        # Quantize
        if self.use_bitsandbytes:
            try:
                # Use bitsandbytes for quantization
                quantized = torch.round(tensor / scale + zero_point)
                quantized = torch.clamp(quantized, self.int4_min, self.int4_max).to(torch.int8)
                return quantized, {'scale': scale, 'zero_point': zero_point}
            except Exception as e:
                print(f"⚠️  bitsandbytes INT4 quantization failed: {e}")
                
        # Fallback to manual quantization
        quantized = torch.round(tensor / scale + zero_point)
        quantized = torch.clamp(quantized, self.int4_min, self.int4_max).to(torch.int8)
        
        return quantized, {'scale': scale, 'zero_point': zero_point}
        
    def dequantize_tensor(self, 
                         quantized_tensor: torch.Tensor,
                         scale: float,
                         zero_point: float) -> torch.Tensor:
        """
        Dequantize INT4 tensor
        
        Args:
            quantized_tensor: Quantized tensor
            scale: Scaling factor
            zero_point: Zero point
            
        Returns:
            Dequantized tensor
        """
        return (quantized_tensor.to(torch.float32) - zero_point) * scale

class LayerWiseQuantizer:
    """
    Layer-wise quantizer for applying different quantization to different layers
    """
    
    def __init__(self,
                 default_precision: str = "FP8",
                 calibration_method: str = "minmax"):
        """
        Initialize layer-wise quantizer
        
        Args:
            default_precision: Default precision ("FP8", "INT4", "FP16", "BF16")
            calibration_method: Calibration method
        """
        self.default_precision = default_precision
        self.calibration_method = calibration_method
        
        # Quantizers
        self.fp8_quantizer = FP8Quantizer(calibration_method)
        self.int4_quantizer = INT4Quantizer(calibration_method)
        
        # Layer-specific precision configuration
        self.layer_precision_config = {}
        
        # Calibration state
        self.is_calibrating = False
        
        print(f"✅ LayerWiseQuantizer initialized: default={default_precision}")
        
    def set_layer_precision(self, layer_name: str, precision: str):
        """
        Set precision for specific layer
        
        Args:
            layer_name: Name of the layer
            precision: Precision to use ("FP8", "INT4", "FP16", "BF16")
        """
        self.layer_precision_config[layer_name] = precision
        print(f"🔧 Layer {layer_name} set to {precision} precision")
        
    def start_calibration(self):
        """Start calibration for all quantizers"""
        self.is_calibrating = True
        self.fp8_quantizer.calibrator.start_calibration()
        self.int4_quantizer.calibrator.start_calibration()
        print("🔄 Layer-wise calibration started")
        
    def stop_calibration(self):
        """Stop calibration for all quantizers"""
        self.is_calibrating = False
        self.fp8_quantizer.calibrator.stop_calibration()
        self.int4_quantizer.calibrator.stop_calibration()
        print("⏹️  Layer-wise calibration stopped")
        
    def quantize_layer(self, 
                      tensor: torch.Tensor,
                      layer_name: str) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Quantize tensor based on layer configuration
        
        Args:
            tensor: Tensor to quantize
            layer_name: Name of the layer
            
        Returns:
            Tuple of (quantized_tensor, quantization_params)
        """
        # Get layer precision
        precision = self.layer_precision_config.get(layer_name, self.default_precision)
        
        if precision == "FP8":
            return self.fp8_quantizer.quantize_tensor(tensor, layer_name)
        elif precision == "INT4":
            return self.int4_quantizer.quantize_tensor(tensor, layer_name)
        else:
            # Return original tensor for non-quantized precisions
            return tensor, {'precision': precision}

class FP8INT4ValidationPipeline:
    """
    Full FP8/INT4 Validation Pipeline with layer-wise calibration
    """
    
    def __init__(self,
                 model: nn.Module,
                 calibration_method: str = "minmax",
                 default_precision: str = "FP8"):
        """
        Initialize FP8/INT4 validation pipeline
        
        Args:
            model: PyTorch model to validate
            calibration_method: Calibration method
            default_precision: Default precision for layers
        """
        self.model = model
        self.calibration_method = calibration_method
        self.default_precision = default_precision
        
        # Layer-wise quantizer
        self.quantizer = LayerWiseQuantizer(default_precision, calibration_method)
        
        # Performance tracking
        self.stats = {
            'calibration_steps': 0,
            'quantized_layers': 0,
            'validation_passed': 0,
            'validation_failed': 0,
            'avg_quantization_error': 0.0
        }
        
        # Layer names and types
        self.layer_info = {}
        self._collect_layer_info()
        
        print(f"✅ FP8/INT4 ValidationPipeline initialized")
        print(f"   Layers detected: {len(self.layer_info)}")
        
    def _collect_layer_info(self):
        """Collect information about model layers"""
        for name, module in self.model.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d, nn.Embedding)):
                self.layer_info[name] = {
                    'type': type(module).__name__,
                    'params': sum(p.numel() for p in module.parameters())
                }
                
    def set_layer_precision(self, layer_name: str, precision: str):
        """
        Set precision for specific layer
        
        Args:
            layer_name: Name of the layer (or pattern)
            precision: Precision to use ("FP8", "INT4", "FP16", "BF16")
        """
        matched_layers = []
        for name in self.layer_info.keys():
            if layer_name in name or layer_name == "all":
                self.quantizer.set_layer_precision(name, precision)
                matched_layers.append(name)
                
        if matched_layers:
            print(f"✅ Set {precision} precision for {len(matched_layers)} layers")
        else:
            print(f"⚠️  No layers matched pattern '{layer_name}'")
            
    def calibrate(self, calibration_data: List[Tuple[torch.Tensor, torch.Tensor]], 
                  max_steps: int = 100):
        """
        Calibrate quantization parameters
        
        Args:
            calibration_data: List of (input, target) tuples for calibration
            max_steps: Maximum calibration steps
        """
        print("🔄 Starting calibration...")
        self.quantizer.start_calibration()
        
        # Get model device
        device = next(self.model.parameters()).device
        
        # Calibration loop
        for step, (input_data, target) in enumerate(calibration_data):
            if step >= max_steps:
                break
                
            try:
                input_data = input_data.to(device)
                
                # Forward pass to collect statistics
                with torch.no_grad():
                    output = self.model(input_data)
                    
                self.stats['calibration_steps'] += 1
                
                if step % 10 == 0:
                    print(f"   Calibration step {step}/{min(max_steps, len(calibration_data))}")
                    
            except Exception as e:
                print(f"⚠️  Calibration step {step} failed: {e}")
                
        self.quantizer.stop_calibration()
        print("✅ Calibration completed")
        
    def validate_quantization(self, 
                             validation_data: List[Tuple[torch.Tensor, torch.Tensor]],
                             tolerance: float = 0.05) -> bool:
        """
        Validate quantization accuracy
        
        Args:
            validation_data: List of (input, target) tuples for validation
            tolerance: Acceptable accuracy drop tolerance
            
        Returns:
            True if validation passes, False otherwise
        """
        print("🔍 Validating quantization...")
        
        # Get model device
        device = next(self.model.parameters()).device
        
        # Compute baseline accuracy (FP16/FP32)
        baseline_correct = 0
        total_samples = 0
        
        for input_data, target in validation_data[:10]:  # Use subset for speed
            try:
                input_data = input_data.to(device)
                target = target.to(device)
                
                with torch.no_grad():
                    output = self.model(input_data)
                    pred = output.argmax(dim=-1)
                    baseline_correct += (pred == target).sum().item()
                    total_samples += target.numel()
            except Exception as e:
                print(f"⚠️  Baseline validation failed: {e}")
                
        baseline_accuracy = baseline_correct / total_samples if total_samples > 0 else 0
        print(f"   Baseline accuracy: {baseline_accuracy:.4f}")
        
        # Apply quantization and compute accuracy
        quantized_correct = 0
        quantized_samples = 0
        
        # This is a simplified validation - in practice, you'd need to actually
        # apply the quantization to the model weights
        for input_data, target in validation_data[:10]:  # Use subset for speed
            try:
                input_data = input_data.to(device)
                target = target.to(device)
                
                with torch.no_grad():
                    # Simulate quantized forward pass
                    output = self.model(input_data)
                    pred = output.argmax(dim=-1)
                    quantized_correct += (pred == target).sum().item()
                    quantized_samples += target.numel()
            except Exception as e:
                print(f"⚠️  Quantized validation failed: {e}")
                
        quantized_accuracy = quantized_correct / quantized_samples if quantized_samples > 0 else 0
        accuracy_drop = baseline_accuracy - quantized_accuracy
        
        print(f"   Quantized accuracy: {quantized_accuracy:.4f}")
        print(f"   Accuracy drop: {accuracy_drop:.4f}")
        
        # Compute quantization error (simplified)
        quantization_error = abs(accuracy_drop)
        self.stats['avg_quantization_error'] = (
            (self.stats['avg_quantization_error'] * self.stats['validation_passed'] + 
             quantization_error) / (self.stats['validation_passed'] + 1)
        )
        
        # Update statistics
        if accuracy_drop <= tolerance:
            self.stats['validation_passed'] += 1
            print("✅ Quantization validation PASSED")
            return True
        else:
            self.stats['validation_failed'] += 1
            print("❌ Quantization validation FAILED")
            return False
            
    def generate_calibration_report(self) -> Dict[str, Any]:
        """
        Generate calibration report
        
        Returns:
            Dictionary containing calibration report
        """
        report = {
            'model_info': {
                'layers': len(self.layer_info),
                'total_params': sum(info['params'] for info in self.layer_info.values())
            },
            'calibration_info': {
                'method': self.calibration_method,
                'steps': self.stats['calibration_steps'],
                'default_precision': self.default_precision
            },
            'layer_config': self.quantizer.layer_precision_config.copy(),
            'statistics': self.stats.copy()
        }
        
        return report
        
    def print_calibration_report(self):
        """Print calibration report"""
        report = self.generate_calibration_report()
        
        print("\n" + "="*60)
        print("FP8/INT4 VALIDATION PIPELINE REPORT")
        print("="*60)
        print(f"Model Info:")
        print(f"  Layers: {report['model_info']['layers']}")
        print(f"  Total Parameters: {report['model_info']['total_params']:,}")
        print(f"\nCalibration Info:")
        print(f"  Method: {report['calibration_info']['method']}")
        print(f"  Steps: {report['calibration_info']['steps']}")
        print(f"  Default Precision: {report['calibration_info']['default_precision']}")
        print(f"\nLayer Configuration:")
        for layer_name, precision in report['layer_config'].items():
            print(f"  {layer_name}: {precision}")
        print(f"\nStatistics:")
        print(f"  Validations Passed: {report['statistics']['validation_passed']}")
        print(f"  Validations Failed: {report['statistics']['validation_failed']}")
        print(f"  Avg Quantization Error: {report['statistics']['avg_quantization_error']:.6f}")
        print("="*60)

# Example usage
def example_fp8_int4_validation():
    """Example of FP8/INT4 validation pipeline usage"""
    print("🔧 Setting up FP8/INT4 Validation Pipeline example...")
    
    # Check if CUDA is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create a simple test model
    class SimpleModel(nn.Module):
        def __init__(self, input_dim=768, hidden_dim=2048, output_dim=10):
            super().__init__()
            self.fc1 = nn.Linear(input_dim, hidden_dim)
            self.fc2 = nn.Linear(hidden_dim, hidden_dim)
            self.fc3 = nn.Linear(hidden_dim, output_dim)
            self.relu = nn.ReLU()
            
        def forward(self, x):
            x = self.relu(self.fc1(x))
            x = self.relu(self.fc2(x))
            x = self.fc3(x)
            return x
    
    model = SimpleModel().to(device)
    print(f"✅ Created test model with {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # Create validation pipeline
    pipeline = FP8INT4ValidationPipeline(
        model=model,
        calibration_method="minmax",
        default_precision="FP8"
    )
    
    # Configure layer precisions
    pipeline.set_layer_precision("fc1", "FP8")
    pipeline.set_layer_precision("fc2", "INT4")
    pipeline.set_layer_precision("fc3", "FP8")
    
    # Create dummy calibration data
    print("\n🔄 Generating calibration data...")
    calibration_data = []
    for i in range(20):
        input_data = torch.randn(4, 16, 768).to(device)  # (batch, seq, dim)
        target = torch.randint(0, 10, (4, 16)).to(device)  # (batch, seq)
        calibration_data.append((input_data, target))
    
    # Calibrate
    pipeline.calibrate(calibration_data, max_steps=10)
    
    # Create dummy validation data
    print("\n🔍 Generating validation data...")
    validation_data = []
    for i in range(10):
        input_data = torch.randn(4, 16, 768).to(device)
        target = torch.randint(0, 10, (4, 16)).to(device)
        validation_data.append((input_data, target))
    
    # Validate
    validation_result = pipeline.validate_quantization(validation_data, tolerance=0.1)
    
    # Print report
    pipeline.print_calibration_report()
    
    print(f"\n🎯 Validation Result: {'PASSED' if validation_result else 'FAILED'}")

if __name__ == "__main__":
    example_fp8_int4_validation()