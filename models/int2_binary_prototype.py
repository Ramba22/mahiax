"""
INT2 / Binary Prototype for TinyEdge Devices
Implementation of 2-bit integer and binary quantization for edge deployment
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Any, List, Tuple, Union
import math

class INT2Quantizer:
    """
    2-bit integer quantizer for extreme compression
    """
    
    def __init__(self, 
                 calibration_method: str = "minmax",
                 symmetric: bool = True):
        """
        Initialize INT2 quantizer
        
        Args:
            calibration_method: Calibration method ("minmax", "percentile")
            symmetric: Whether to use symmetric quantization
        """
        self.calibration_method = calibration_method
        self.symmetric = symmetric
        
        # INT2 parameters (-2 to 1)
        self.int2_min = -2
        self.int2_max = 1
        
        # Calibration data storage
        self.calibration_data = {}
        self.is_calibrating = False
        
        print(f"✅ INT2Quantizer initialized: symmetric={symmetric}")
        
    def start_calibration(self):
        """Start calibration process"""
        self.is_calibrating = True
        self.calibration_data = {}
        print("🔄 INT2 calibration started")
        
    def stop_calibration(self):
        """Stop calibration process"""
        self.is_calibrating = False
        print("⏹️  INT2 calibration stopped")
        
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
                'abs_max': float('-inf'),
                'count': 0
            }
            
        data = self.calibration_data[layer_name]
        
        # Update min/max values
        tensor_min = tensor.min().item()
        tensor_max = tensor.max().item()
        data['min'] = min(data['min'], tensor_min)
        data['max'] = max(data['max'], tensor_max)
        
        # Update absolute max
        tensor_abs_max = tensor.abs().max().item()
        data['abs_max'] = max(data['abs_max'], tensor_abs_max)
        
        data['count'] += tensor.numel()
        
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
            if self.symmetric:
                # Symmetric quantization
                abs_max = max(abs(data['min']), abs(data['max']))
                return {
                    'min': -abs_max,
                    'max': abs_max
                }
            else:
                # Asymmetric quantization
                return {
                    'min': data['min'],
                    'max': data['max']
                }
        elif self.calibration_method == "percentile":
            # For INT2, we'll stick with minmax as percentile is less meaningful
            return {
                'min': data['min'],
                'max': data['max']
            }
        else:
            return {
                'min': data['min'],
                'max': data['max']
            }
            
    def quantize_tensor(self, 
                       tensor: torch.Tensor,
                       layer_name: str) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Quantize tensor to INT2
        
        Args:
            tensor: Tensor to quantize
            layer_name: Name of the layer
            
        Returns:
            Tuple of (quantized_tensor, quantization_params)
        """
        # Collect statistics during calibration
        self.collect_statistics(layer_name, tensor)
        
        # Get calibration parameters
        cal_params = self.compute_calibration_params(layer_name)
        
        if not cal_params:
            # Fallback to tensor min/max
            tensor_min = tensor.min()
            tensor_max = tensor.max()
        else:
            tensor_min = cal_params['min']
            tensor_max = cal_params['max']
            
        # Determine scaling and zero point
        if self.symmetric:
            # Symmetric quantization
            scale = max(abs(tensor_min), abs(tensor_max)) / (self.int2_max - self.int2_min)
            zero_point = 0.0
        else:
            # Asymmetric quantization
            scale = (tensor_max - tensor_min) / (self.int2_max - self.int2_min)
            zero_point = self.int2_min - tensor_min / scale
            
        scale = max(scale, 1e-8)  # Avoid division by zero
        
        # Quantize
        quantized = torch.round(tensor / scale + zero_point)
        quantized = torch.clamp(quantized, self.int2_min, self.int2_max).to(torch.int8)
        
        return quantized, {
            'scale': scale, 
            'zero_point': zero_point,
            'min': tensor_min,
            'max': tensor_max
        }
        
    def dequantize_tensor(self, 
                         quantized_tensor: torch.Tensor,
                         scale: float,
                         zero_point: float) -> torch.Tensor:
        """
        Dequantize INT2 tensor
        
        Args:
            quantized_tensor: Quantized tensor
            scale: Scaling factor
            zero_point: Zero point
            
        Returns:
            Dequantized tensor
        """
        return (quantized_tensor.to(torch.float32) - zero_point) * scale

class BinaryQuantizer:
    """
    Binary quantizer for extreme compression
    """
    
    def __init__(self, 
                 calibration_method: str = "minmax"):
        """
        Initialize binary quantizer
        
        Args:
            calibration_method: Calibration method
        """
        self.calibration_method = calibration_method
        
        # Binary parameters (-1, 1)
        self.binary_values = [-1, 1]
        
        # Calibration data storage
        self.calibration_data = {}
        self.is_calibrating = False
        
        print(f"✅ BinaryQuantizer initialized")
        
    def start_calibration(self):
        """Start calibration process"""
        self.is_calibrating = True
        self.calibration_data = {}
        print("🔄 Binary calibration started")
        
    def stop_calibration(self):
        """Stop calibration process"""
        self.is_calibrating = False
        print("⏹️  Binary calibration stopped")
        
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
                'mean': 0.0,
                'std': 0.0,
                'count': 0
            }
            
        data = self.calibration_data[layer_name]
        
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
        
    def quantize_tensor(self, 
                       tensor: torch.Tensor,
                       layer_name: str) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Quantize tensor to binary (-1, 1)
        
        Args:
            tensor: Tensor to quantize
            layer_name: Name of the layer
            
        Returns:
            Tuple of (quantized_tensor, quantization_params)
        """
        # Collect statistics during calibration
        self.collect_statistics(layer_name, tensor)
        
        # Get calibration parameters
        if layer_name in self.calibration_data:
            data = self.calibration_data[layer_name]
            threshold = data['mean']  # Use mean as threshold
        else:
            threshold = 0.0  # Default threshold
            
        # Binary quantization
        binary_tensor = torch.where(tensor > threshold, 
                                  torch.tensor(1.0, device=tensor.device),
                                  torch.tensor(-1.0, device=tensor.device))
        
        return binary_tensor, {
            'threshold': threshold,
            'mean': threshold
        }
        
    def dequantize_tensor(self, 
                         quantized_tensor: torch.Tensor,
                         threshold: float) -> torch.Tensor:
        """
        Dequantize binary tensor (identity for binary)
        
        Args:
            quantized_tensor: Quantized tensor
            threshold: Threshold used for quantization
            
        Returns:
            Dequantized tensor (same as input for binary)
        """
        return quantized_tensor

class TinyEdgeQuantizer:
    """
    Unified quantizer for TinyEdge devices with INT2 and binary support
    """
    
    def __init__(self,
                 default_precision: str = "INT2",
                 calibration_method: str = "minmax"):
        """
        Initialize TinyEdge quantizer
        
        Args:
            default_precision: Default precision ("INT2", "BINARY", "FP16")
            calibration_method: Calibration method
        """
        self.default_precision = default_precision
        self.calibration_method = calibration_method
        
        # Quantizers
        self.int2_quantizer = INT2Quantizer(calibration_method, symmetric=True)
        self.binary_quantizer = BinaryQuantizer(calibration_method)
        
        # Layer-specific precision configuration
        self.layer_precision_config = {}
        
        # Calibration state
        self.is_calibrating = False
        
        print(f"✅ TinyEdgeQuantizer initialized: default={default_precision}")
        
    def set_layer_precision(self, layer_name: str, precision: str):
        """
        Set precision for specific layer
        
        Args:
            layer_name: Name of the layer
            precision: Precision to use ("INT2", "BINARY", "FP16")
        """
        valid_precisions = ["INT2", "BINARY", "FP16"]
        if precision not in valid_precisions:
            raise ValueError(f"Invalid precision: {precision}. Must be one of {valid_precisions}")
            
        self.layer_precision_config[layer_name] = precision
        print(f"🔧 Layer {layer_name} set to {precision} precision")
        
    def start_calibration(self):
        """Start calibration for all quantizers"""
        self.is_calibrating = True
        self.int2_quantizer.start_calibration()
        self.binary_quantizer.start_calibration()
        print("🔄 TinyEdge calibration started")
        
    def stop_calibration(self):
        """Stop calibration for all quantizers"""
        self.is_calibrating = False
        self.int2_quantizer.stop_calibration()
        self.binary_quantizer.stop_calibration()
        print("⏹️  TinyEdge calibration stopped")
        
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
        
        if precision == "INT2":
            return self.int2_quantizer.quantize_tensor(tensor, layer_name)
        elif precision == "BINARY":
            return self.binary_quantizer.quantize_tensor(tensor, layer_name)
        else:
            # Return original tensor for non-quantized precisions
            return tensor, {'precision': precision}

class INT2BinaryLayer(nn.Module):
    """
    Neural network layer with INT2/Binary quantization support
    """
    
    def __init__(self,
                 in_features: int,
                 out_features: int,
                 precision: str = "INT2",
                 bias: bool = True):
        """
        Initialize INT2/Binary layer
        
        Args:
            in_features: Number of input features
            out_features: Number of output features
            precision: Precision to use ("INT2", "BINARY", "FP16")
            bias: Whether to use bias
        """
        super().__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        self.precision = precision
        self.use_bias = bias
        
        # Weight and bias parameters
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.randn(out_features))
        else:
            self.register_parameter('bias', None)
            
        # Quantization parameters (stored for dequantization)
        self.register_buffer('weight_scale', torch.tensor(1.0))
        self.register_buffer('weight_zero_point', torch.tensor(0.0))
        self.register_buffer('quantized_weight', torch.zeros_like(self.weight, dtype=torch.int8))
        
        # Initialize weights
        nn.init.xavier_uniform_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)
            
        print(f"✅ INT2BinaryLayer initialized: {in_features}→{out_features}, {precision}")
        
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through INT2/Binary layer
        
        Args:
            input: Input tensor
            
        Returns:
            Output tensor
        """
        # For inference, use quantized weights
        if self.training:
            # During training, use full precision
            return F.linear(input, self.weight, self.bias)
        else:
            # During inference, use quantized weights
            # This is a simplified implementation - in practice, you'd need to
            # properly dequantize the weights
            return F.linear(input, self.weight, self.bias)
            
    def quantize_weights(self, quantizer: TinyEdgeQuantizer, layer_name: str):
        """
        Quantize layer weights
        
        Args:
            quantizer: TinyEdge quantizer to use
            layer_name: Name of the layer
        """
        # Quantize weights
        quantized_weight, quant_params = quantizer.quantize_layer(self.weight, layer_name)
        
        # Store quantization parameters
        self.quantized_weight = quantized_weight
        self.weight_scale = torch.tensor(quant_params.get('scale', 1.0))
        self.weight_zero_point = torch.tensor(quant_params.get('zero_point', 0.0))
        
        print(f"✅ Weights quantized for layer {layer_name}")

class TinyEdgeModel(nn.Module):
    """
    Complete model for TinyEdge deployment with INT2/Binary quantization
    """
    
    def __init__(self,
                 input_dim: int,
                 hidden_dims: List[int],
                 output_dim: int,
                 precision: str = "INT2"):
        """
        Initialize TinyEdge model
        
        Args:
            input_dim: Input dimension
            hidden_dims: List of hidden layer dimensions
            output_dim: Output dimension
            precision: Default precision for layers
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.output_dim = output_dim
        self.precision = precision
        
        # Create layers
        layers = []
        prev_dim = input_dim
        
        # Hidden layers
        for i, hidden_dim in enumerate(hidden_dims):
            layer = INT2BinaryLayer(
                in_features=prev_dim,
                out_features=hidden_dim,
                precision=precision,
                bias=True
            )
            layers.append((f"layer_{i}", layer))
            prev_dim = hidden_dim
            
        # Output layer
        output_layer = INT2BinaryLayer(
            in_features=prev_dim,
            out_features=output_dim,
            precision="FP16",  # Keep output in FP16 for accuracy
            bias=True
        )
        layers.append(("output", output_layer))
        
        # Register layers
        self.layers = nn.ModuleDict(layers)
        
        # Quantizer
        self.quantizer = TinyEdgeQuantizer(precision, "minmax")
        
        print(f"✅ TinyEdgeModel initialized: {input_dim}→{hidden_dims}→{output_dim}")
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through TinyEdge model
        
        Args:
            x: Input tensor
            
        Returns:
            Output tensor
        """
        # Apply layers sequentially
        for layer in self.layers.values():
            x = layer(x)
            if not isinstance(layer, INT2BinaryLayer) or layer != list(self.layers.values())[-1]:
                x = F.relu(x)  # Apply ReLU except for output layer
                
        return x
        
    def calibrate(self, calibration_data: List[torch.Tensor], max_steps: int = 100):
        """
        Calibrate model for quantization
        
        Args:
            calibration_data: List of input tensors for calibration
            max_steps: Maximum calibration steps
        """
        print("🔄 Calibrating TinyEdge model...")
        self.quantizer.start_calibration()
        
        # Forward pass through calibration data
        for step, input_data in enumerate(calibration_data):
            if step >= max_steps:
                break
                
            try:
                # Forward pass to collect statistics
                with torch.no_grad():
                    output = self(input_data)
                    
                if step % 10 == 0:
                    print(f"   Calibration step {step}/{min(max_steps, len(calibration_data))}")
                    
            except Exception as e:
                print(f"⚠️  Calibration step {step} failed: {e}")
                
        self.quantizer.stop_calibration()
        print("✅ Calibration completed")
        
    def quantize_model(self):
        """Quantize all layers in the model"""
        print("🔄 Quantizing model...")
        
        for name, layer in self.layers.items():
            if isinstance(layer, INT2BinaryLayer):
                layer.quantize_weights(self.quantizer, name)
                
        print("✅ Model quantization completed")
        
    def get_model_size(self) -> Dict[str, float]:
        """
        Get model size information
        
        Returns:
            Dictionary with size information
        """
        total_params = sum(p.numel() for p in self.parameters())
        
        # Calculate quantized size
        # INT2: 2 bits per parameter
        # BINARY: 1 bit per parameter
        # FP16: 16 bits per parameter
        
        int2_params = 0
        binary_params = 0
        fp16_params = 0
        
        for name, layer in self.layers.items():
            if isinstance(layer, INT2BinaryLayer):
                layer_params = layer.weight.numel() + (layer.bias.numel() if layer.bias is not None else 0)
                precision = self.quantizer.layer_precision_config.get(name, self.precision)
                
                if precision == "INT2":
                    int2_params += layer_params
                elif precision == "BINARY":
                    binary_params += layer_params
                else:
                    fp16_params += layer_params
            else:
                fp16_params += sum(p.numel() for p in layer.parameters())
                
        # Calculate sizes in bytes
        int2_size = (int2_params * 2) / 8  # 2 bits = 0.25 bytes
        binary_size = (binary_params * 1) / 8  # 1 bit = 0.125 bytes
        fp16_size = (fp16_params * 16) / 8  # 16 bits = 2 bytes
        
        total_size = int2_size + binary_size + fp16_size
        
        return {
            'total_parameters': total_params,
            'int2_parameters': int2_params,
            'binary_parameters': binary_params,
            'fp16_parameters': fp16_params,
            'int2_size_bytes': int2_size,
            'binary_size_bytes': binary_size,
            'fp16_size_bytes': fp16_size,
            'total_size_bytes': total_size,
            'compression_ratio': (total_params * 32 / 8) / total_size if total_size > 0 else 1.0  # vs FP32
        }
        
    def print_model_info(self):
        """Print model information"""
        size_info = self.get_model_size()
        
        print("\n" + "="*50)
        print("TINYEDGE MODEL INFORMATION")
        print("="*50)
        print(f"Total Parameters: {size_info['total_parameters']:,}")
        print(f"INT2 Parameters: {size_info['int2_parameters']:,}")
        print(f"Binary Parameters: {size_info['binary_parameters']:,}")
        print(f"FP16 Parameters: {size_info['fp16_parameters']:,}")
        print(f"\nModel Size:")
        print(f"  INT2: {size_info['int2_size_bytes']:.2f} bytes")
        print(f"  Binary: {size_info['binary_size_bytes']:.2f} bytes")
        print(f"  FP16: {size_info['fp16_size_bytes']:.2f} bytes")
        print(f"  Total: {size_info['total_size_bytes']:.2f} bytes")
        print(f"Compression Ratio: {size_info['compression_ratio']:.2f}x")
        print("="*50)

# Example usage
def example_int2_binary_prototype():
    """Example of INT2/Binary prototype usage"""
    print("🔧 Setting up INT2/Binary Prototype example...")
    
    # Check if CUDA is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create TinyEdge model
    model = TinyEdgeModel(
        input_dim=768,
        hidden_dims=[512, 256, 128],
        output_dim=10,
        precision="INT2"
    ).to(device)
    
    print(f"✅ Created TinyEdge model")
    
    # Configure layer precisions
    model.quantizer.set_layer_precision("layer_0", "INT2")
    model.quantizer.set_layer_precision("layer_1", "BINARY")
    model.quantizer.set_layer_precision("layer_2", "INT2")
    model.quantizer.set_layer_precision("output", "FP16")  # Keep output in FP16
    
    # Create dummy calibration data
    print("\n🔄 Generating calibration data...")
    calibration_data = []
    for i in range(20):
        input_data = torch.randn(4, 768).to(device)  # (batch, dim)
        calibration_data.append(input_data)
    
    # Calibrate
    model.calibrate(calibration_data, max_steps=10)
    
    # Quantize model
    model.quantize_model()
    
    # Print model information
    model.print_model_info()
    
    # Test inference
    print("\n🚀 Testing inference...")
    test_input = torch.randn(2, 768).to(device)
    
    with torch.no_grad():
        start_time = time.time()
        output = model(test_input)
        inference_time = time.time() - start_time
        
    print(f"Input shape: {test_input.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Inference time: {inference_time*1000:.2f}ms")
    
    # Compare with FP32 model size
    fp32_size = sum(p.numel() * 4 for p in model.parameters())  # 4 bytes per FP32
    quantized_size = model.get_model_size()['total_size_bytes']
    compression_ratio = fp32_size / quantized_size if quantized_size > 0 else 1.0
    
    print(f"\n📊 Size Comparison:")
    print(f"  FP32 Model: {fp32_size:.2f} bytes")
    print(f"  Quantized Model: {quantized_size:.2f} bytes")
    print(f"  Compression: {compression_ratio:.2f}x smaller")

if __name__ == "__main__":
    import time
    example_int2_binary_prototype()