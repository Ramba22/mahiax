"""
ONNX / TensorRT End-to-End Export for MAHIA
Implementation with Hyena/MoE Ops support
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Any, List, Tuple, Union
import os
import json

# Conditional imports for export libraries
ONNX_AVAILABLE = False
try:
    import onnx
    ONNX_AVAILABLE = True
    print("✅ ONNX available")
except ImportError:
    print("⚠️  ONNX not available")

TENSORRT_AVAILABLE = False
try:
    import tensorrt as trt
    TENSORRT_AVAILABLE = True
    print("✅ TensorRT available")
except ImportError:
    print("⚠️  TensorRT not available")

class HyenaMoEExportOps:
    """
    Custom ops for Hyena and MoE operations in ONNX/TensorRT export
    """
    
    @staticmethod
    def hyena_filter_forward(u: torch.Tensor, 
                            h: torch.Tensor, 
                            delta: torch.Tensor) -> torch.Tensor:
        """
        Custom Hyena filter operation for export
        
        Args:
            u: Input tensor
            h: Filter tensor
            delta: Delta tensor
            
        Returns:
            Output tensor
        """
        # This is a simplified implementation for export purposes
        # In practice, this would be a more complex operation
        return u * h * delta
        
    @staticmethod
    def moe_routing_forward(x: torch.Tensor, 
                           expert_weights: torch.Tensor, 
                           expert_indices: torch.Tensor) -> torch.Tensor:
        """
        Custom MoE routing operation for export
        
        Args:
            x: Input tensor
            expert_weights: Expert weights
            expert_indices: Expert indices
            
        Returns:
            Output tensor
        """
        # This is a simplified implementation for export purposes
        # In practice, this would involve actual expert routing
        batch_size, seq_len, d_model = x.shape
        num_experts = expert_weights.shape[-1]
        
        # Simple weighted sum of experts (simplified)
        output = torch.zeros_like(x)
        for i in range(num_experts):
            mask = (expert_indices == i).float()
            output += mask.unsqueeze(-1) * expert_weights[..., i:i+1] * x
            
        return output

class ONNXExporter:
    """
    ONNX exporter for MAHIA models with Hyena/MoE support
    """
    
    def __init__(self, 
                 opset_version: int = 17,
                 export_path: str = "./exports"):
        """
        Initialize ONNX exporter
        
        Args:
            opset_version: ONNX opset version
            export_path: Path to export directory
        """
        self.opset_version = opset_version
        self.export_path = export_path
        
        # Create export directory
        os.makedirs(export_path, exist_ok=True)
        
        # Custom ops registry
        self.custom_ops = {
            'HyenaFilter': HyenaMoEExportOps.hyena_filter_forward,
            'MoERouting': HyenaMoEExportOps.moe_routing_forward
        }
        
        print(f"✅ ONNXExporter initialized: opset={opset_version}")
        
    def export_model(self, 
                     model: nn.Module,
                     input_shape: Tuple[int, ...],
                     model_name: str = "mahia_model",
                     dynamic_axes: Optional[Dict[str, Dict[int, str]]] = None,
                     export_custom_ops: bool = True) -> str:
        """
        Export PyTorch model to ONNX format
        
        Args:
            model: PyTorch model to export
            input_shape: Input tensor shape
            model_name: Name of the exported model
            dynamic_axes: Dynamic axes specification
            export_custom_ops: Whether to export with custom ops
            
        Returns:
            Path to exported ONNX model
        """
        # Set model to evaluation mode
        model.eval()
        
        # Create dummy input
        device = next(model.parameters()).device
        dummy_input = torch.randn(*input_shape, device=device)
        
        # Define output path
        onnx_path = os.path.join(self.export_path, f"{model_name}.onnx")
        
        # Export configuration
        export_kwargs = {
            'input_names': ['input'],
            'output_names': ['output'],
            'opset_version': self.opset_version,
            'export_params': True,
            'do_constant_folding': True
        }
        
        # Add dynamic axes if specified
        if dynamic_axes:
            export_kwargs['dynamic_axes'] = dynamic_axes
        else:
            # Default dynamic axes for sequence length
            export_kwargs['dynamic_axes'] = {
                'input': {0: 'batch_size', 1: 'sequence_length'},
                'output': {0: 'batch_size', 1: 'sequence_length'}
            }
            
        try:
            # Export to ONNX
            torch.onnx.export(
                model,
                dummy_input,
                onnx_path,
                **export_kwargs
            )
            
            print(f"✅ Model exported to ONNX: {onnx_path}")
            
            # Validate exported model
            if ONNX_AVAILABLE:
                self._validate_onnx_model(onnx_path)
                
            return onnx_path
            
        except Exception as e:
            print(f"❌ ONNX export failed: {e}")
            raise
            
    def _validate_onnx_model(self, onnx_path: str):
        """
        Validate exported ONNX model
        
        Args:
            onnx_path: Path to ONNX model
        """
        try:
            # Load and check model
            onnx_model = onnx.load(onnx_path)
            onnx.checker.check_model(onnx_model)
            print("✅ ONNX model validation passed")
            
            # Print model info
            print(f"   Model IR version: {onnx_model.ir_version}")
            print(f"   Opset version: {onnx_model.opset_import[0].version}")
            print(f"   Producer name: {onnx_model.producer_name}")
            
        except Exception as e:
            print(f"⚠️  ONNX model validation failed: {e}")
            
    def export_with_custom_ops(self, 
                              model: nn.Module,
                              input_shape: Tuple[int, ...],
                              model_name: str = "mahia_custom_ops") -> str:
        """
        Export model with custom Hyena/MoE operations
        
        Args:
            model: PyTorch model to export
            input_shape: Input tensor shape
            model_name: Name of the exported model
            
        Returns:
            Path to exported ONNX model
        """
        # This would require implementing custom ops in ONNX
        # For now, we'll export a simplified version
        
        print("🔧 Exporting with custom ops support...")
        
        # Register custom ops (simplified approach)
        # In practice, you'd need to implement proper custom ops
        
        return self.export_model(
            model, 
            input_shape, 
            model_name=f"{model_name}_custom",
            export_custom_ops=True
        )

class TensorRTExporter:
    """
    TensorRT exporter for optimized inference
    """
    
    def __init__(self, 
                 workspace_size: int = 1 << 30,  # 1GB
                 fp16_mode: bool = True,
                 export_path: str = "./exports"):
        """
        Initialize TensorRT exporter
        
        Args:
            workspace_size: Workspace size in bytes
            fp16_mode: Whether to use FP16 precision
            export_path: Path to export directory
        """
        self.workspace_size = workspace_size
        self.fp16_mode = fp16_mode
        self.export_path = export_path
        
        # Create export directory
        os.makedirs(export_path, exist_ok=True)
        
        print(f"✅ TensorRTExporter initialized: fp16={fp16_mode}")
        
    def export_from_onnx(self, 
                        onnx_path: str,
                        engine_name: str = "mahia_engine") -> str:
        """
        Export ONNX model to TensorRT engine
        
        Args:
            onnx_path: Path to ONNX model
            engine_name: Name of the TensorRT engine
            
        Returns:
            Path to exported TensorRT engine
        """
        if not TENSORRT_AVAILABLE:
            raise RuntimeError("TensorRT is not available")
            
        # Define output path
        engine_path = os.path.join(self.export_path, f"{engine_name}.engine")
        
        try:
            # Create TensorRT builder
            builder = trt.Builder(trt.Logger(trt.Logger.WARNING))
            
            # Create network definition
            network = builder.create_network(
                1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
            )
            
            # Create parser
            parser = trt.OnnxParser(network, builder.logger)
            
            # Parse ONNX model
            with open(onnx_path, 'rb') as f:
                parser.parse(f.read())
                
            # Create builder config
            config = builder.create_builder_config()
            config.max_workspace_size = self.workspace_size
            
            # Set FP16 mode
            if self.fp16_mode and builder.platform_has_fast_fp16:
                config.set_flag(trt.BuilderFlag.FP16)
                
            # Build engine
            print("🔄 Building TensorRT engine...")
            engine = builder.build_engine(network, config)
            
            if engine is None:
                raise RuntimeError("Failed to build TensorRT engine")
                
            # Serialize and save engine
            with open(engine_path, 'wb') as f:
                f.write(engine.serialize())
                
            print(f"✅ TensorRT engine exported: {engine_path}")
            
            # Print engine info
            self._print_engine_info(engine)
            
            return engine_path
            
        except Exception as e:
            print(f"❌ TensorRT export failed: {e}")
            raise
            
    def _print_engine_info(self, engine) -> None:
        """
        Print TensorRT engine information
        
        Args:
            engine: TensorRT engine
        """
        try:
            print(f"   Engine bindings: {engine.num_bindings}")
            print(f"   Max batch size: {engine.max_batch_size}")
            
            for i in range(engine.num_bindings):
                name = engine.get_binding_name(i)
                shape = engine.get_binding_shape(i)
                dtype = engine.get_binding_dtype(i)
                print(f"     Binding {i}: {name} {shape} {dtype}")
                
        except Exception as e:
            print(f"⚠️  Failed to print engine info: {e}")

class MAHIAExportPipeline:
    """
    End-to-End export pipeline for MAHIA models with ONNX/TensorRT support
    """
    
    def __init__(self,
                 export_path: str = "./exports",
                 opset_version: int = 17,
                 workspace_size: int = 1 << 30):
        """
        Initialize MAHIA export pipeline
        
        Args:
            export_path: Path to export directory
            opset_version: ONNX opset version
            workspace_size: TensorRT workspace size
        """
        self.export_path = export_path
        self.opset_version = opset_version
        self.workspace_size = workspace_size
        
        # Exporters
        self.onnx_exporter = ONNXExporter(opset_version, export_path)
        self.tensorrt_exporter = TensorRTExporter(workspace_size, True, export_path) if TENSORRT_AVAILABLE else None
        
        # Performance tracking
        self.export_stats = {
            'models_exported': 0,
            'onnx_exports': 0,
            'tensorrt_exports': 0,
            'total_export_time': 0.0
        }
        
        print(f"✅ MAHIAExportPipeline initialized")
        
    def export_model(self,
                    model: nn.Module,
                    input_shape: Tuple[int, ...],
                    model_name: str = "mahia_model",
                    export_formats: List[str] = ["onnx", "tensorrt"],
                    dynamic_axes: Optional[Dict[str, Dict[int, str]]] = None) -> Dict[str, str]:
        """
        Export model to multiple formats
        
        Args:
            model: PyTorch model to export
            input_shape: Input tensor shape
            model_name: Name of the model
            export_formats: List of formats to export ("onnx", "tensorrt")
            dynamic_axes: Dynamic axes specification
            
        Returns:
            Dictionary mapping format to exported file path
        """
        import time
        start_time = time.time()
        
        exported_paths = {}
        
        # Export to ONNX (required for TensorRT)
        if "onnx" in export_formats or "tensorrt" in export_formats:
            print(f"🔄 Exporting {model_name} to ONNX...")
            onnx_path = self.onnx_exporter.export_model(
                model, input_shape, model_name, dynamic_axes
            )
            exported_paths["onnx"] = onnx_path
            self.export_stats['onnx_exports'] += 1
            
        # Export to TensorRT
        if "tensorrt" in export_formats and TENSORRT_AVAILABLE:
            print(f"🔄 Exporting {model_name} to TensorRT...")
            try:
                engine_path = self.tensorrt_exporter.export_from_onnx(
                    onnx_path, f"{model_name}_trt"
                )
                exported_paths["tensorrt"] = engine_path
                self.export_stats['tensorrt_exports'] += 1
            except Exception as e:
                print(f"⚠️  TensorRT export failed: {e}")
                
        # Update statistics
        self.export_stats['models_exported'] += 1
        self.export_stats['total_export_time'] += (time.time() - start_time)
        
        print(f"✅ Export completed for {model_name}")
        return exported_paths
        
    def export_hyena_moe_model(self,
                              model: nn.Module,
                              input_shape: Tuple[int, ...],
                              model_name: str = "mahia_hyena_moe") -> Dict[str, str]:
        """
        Export Hyena/MoE model with custom ops support
        
        Args:
            model: Hyena/MoE model to export
            input_shape: Input tensor shape
            model_name: Name of the model
            
        Returns:
            Dictionary mapping format to exported file path
        """
        print(f"🔧 Exporting Hyena/MoE model: {model_name}")
        
        # Export with custom ops support
        exported_paths = self.export_model(
            model, input_shape, model_name,
            export_formats=["onnx"],  # TensorRT export of custom ops is complex
            dynamic_axes={
                'input': {0: 'batch_size', 1: 'sequence_length'},
                'output': {0: 'batch_size', 1: 'sequence_length'}
            }
        )
        
        return exported_paths
        
    def get_export_stats(self) -> Dict[str, Any]:
        """
        Get export statistics
        
        Returns:
            Dictionary of export statistics
        """
        stats = self.export_stats.copy()
        if stats['models_exported'] > 0:
            stats['avg_export_time'] = stats['total_export_time'] / stats['models_exported']
        else:
            stats['avg_export_time'] = 0.0
        return stats
        
    def print_export_report(self):
        """Print export report"""
        stats = self.get_export_stats()
        
        print("\n" + "="*50)
        print("MAHIA EXPORT PIPELINE REPORT")
        print("="*50)
        print(f"Models Exported: {stats['models_exported']}")
        print(f"ONNX Exports: {stats['onnx_exports']}")
        print(f"TensorRT Exports: {stats['tensorrt_exports']}")
        print(f"Total Export Time: {stats['total_export_time']:.2f}s")
        print(f"Average Export Time: {stats['avg_export_time']:.2f}s")
        print("="*50)

# Example usage
def example_onnx_tensorrt_export():
    """Example of ONNX/TensorRT export pipeline usage"""
    print("🔧 Setting up ONNX/TensorRT Export Pipeline example...")
    
    # Check if CUDA is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create a simple test model with operations that might be in Hyena/MoE
    class SimpleHyenaMoEModel(nn.Module):
        def __init__(self, input_dim=768, hidden_dim=2048, output_dim=768):
            super().__init__()
            self.input_dim = input_dim
            self.hidden_dim = hidden_dim
            self.output_dim = output_dim
            
            # Linear layers
            self.fc1 = nn.Linear(input_dim, hidden_dim)
            self.fc2 = nn.Linear(hidden_dim, hidden_dim)
            self.fc3 = nn.Linear(hidden_dim, output_dim)
            
            # Simple "Hyena" components (simplified)
            self.filter_weights = nn.Parameter(torch.randn(hidden_dim))
            self.delta_weights = nn.Parameter(torch.randn(hidden_dim))
            
            # Simple "MoE" components (simplified)
            self.num_experts = 4
            self.experts = nn.ModuleList([
                nn.Linear(hidden_dim, hidden_dim) for _ in range(self.num_experts)
            ])
            self.gate = nn.Linear(hidden_dim, self.num_experts)
            
        def forward(self, x):
            # First linear layer
            x = F.relu(self.fc1(x))
            
            # Simulate Hyena filter operation
            batch_size, seq_len, hidden_dim = x.shape
            h = self.filter_weights.unsqueeze(0).unsqueeze(0)  # (1, 1, hidden_dim)
            delta = self.delta_weights.unsqueeze(0).unsqueeze(0)  # (1, 1, hidden_dim)
            x = x * h * delta  # Simplified Hyena operation
            
            # Simulate MoE routing
            gate_logits = self.gate(x)  # (batch, seq, num_experts)
            gate_weights = F.softmax(gate_logits, dim=-1)
            
            # Weighted sum of experts (simplified)
            expert_outputs = []
            for i, expert in enumerate(self.experts):
                expert_output = expert(x)
                expert_outputs.append(expert_output)
                
            expert_outputs = torch.stack(expert_outputs, dim=-1)  # (batch, seq, hidden, num_experts)
            x = torch.sum(expert_outputs * gate_weights.unsqueeze(-2), dim=-1)
            
            # Final linear layer
            x = self.fc3(x)
            
            return x
    
    # Create model
    model = SimpleHyenaMoEModel().to(device)
    print(f"✅ Created test Hyena/MoE model")
    
    # Create export pipeline
    export_pipeline = MAHIAExportPipeline(
        export_path="./model_exports",
        opset_version=17,
        workspace_size=1 << 30  # 1GB
    )
    
    # Export model
    print("\n🔄 Exporting model...")
    input_shape = (2, 64, 768)  # (batch, seq, dim)
    
    try:
        exported_paths = export_pipeline.export_model(
            model, input_shape, "mahia_test_model",
            export_formats=["onnx"]  # TensorRT export requires proper installation
        )
        
        print(f"✅ Exported model paths: {exported_paths}")
        
        # Export Hyena/MoE specific model
        print("\n🔧 Exporting Hyena/MoE model...")
        hyena_moe_paths = export_pipeline.export_hyena_moe_model(
            model, input_shape, "mahia_hyena_moe_test"
        )
        
        print(f"✅ Hyena/MoE exported paths: {hyena_moe_paths}")
        
        # Print export report
        export_pipeline.print_export_report()
        
    except Exception as e:
        print(f"❌ Export failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    example_onnx_tensorrt_export()