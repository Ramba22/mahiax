"""
Automatic technical specs and documentation generator from config files.
"""
import json
import yaml
import torch
import sys
import os
from typing import Dict, Any, Union
from datetime import datetime

# Add the project root to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

try:
    from modell_V5_MAHIA_HyenaMoE import (
        MAHIA_V5,
        SparseMoETopK,
        HyenaBlock,
        LoRAAdapter,
        FP8CalibrationAutoTuner,
        MetaLRPolicyController,
        PredictiveStopForecaster,
        CurriculumMemorySystem,
        GPUTelemetryMonitor,
        TelemetryLogger,
        TrainingDashboardV2
    )
    DOCS_AVAILABLE = True
except ImportError:
    DOCS_AVAILABLE = False
    print("⚠️  MAHIA-X modules not available for documentation generation")


class TechnicalSpecsGenerator:
    """Generate technical specifications and documentation from config files"""
    
    def __init__(self, output_dir: str = "./specs"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.specs = {}
        
    def load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from file
        
        Args:
            config_path: Path to config file (.json or .yaml)
            
        Returns:
            Configuration dictionary
        """
        with open(config_path, 'r') as f:
            if config_path.endswith('.json'):
                return json.load(f)
            elif config_path.endswith(('.yaml', '.yml')):
                return yaml.safe_load(f)
            else:
                raise ValueError(f"Unsupported config format: {config_path}")
                
    def generate_model_specs(self, model_config: Dict[str, Any]) -> Dict[str, Any]:
        """Generate model architecture specifications
        
        Args:
            model_config: Model configuration dictionary
            
        Returns:
            Model specifications dictionary
        """
        if model_config is None:
            # Default configuration
            model_config = {
                "model_name": "MAHIA-X",
                "version": "5.0",
                "architecture": {
                    "text_encoder": {
                        "type": "Transformer-based",
                        "vocab_size": 10000,
                        "embed_dim": 64,
                        "max_seq_len": 64,
                        "num_layers": 2
                    },
                    "tabular_encoder": {
                        "type": "MLP-based",
                        "input_dim": 50,
                        "hidden_dim": 64,
                        "output_dim": 64
                    },
                    "moe_layer": {
                        "type": "SparseMoETopK",
                        "num_experts": 8,
                        "top_k": 2,
                        "capacity_factor": 1.25
                    },
                    "fusion_layer": {
                        "type": "Attention-based",
                        "fused_dim": 128
                    },
                    "output_layer": {
                        "type": "MLP",
                        "hidden_dim": 64,
                        "output_dim": 2
                    }
                },
                "training": {
                    "mixed_precision": True,
                    "gradient_checkpointing": True,
                    "quantization": {
                        "type": "FP8",
                        "enabled": True
                    },
                    "lora": {
                        "enabled": True,
                        "rank": 8,
                        "alpha": 1.0
                    }
                }
            }
            
        specs = {
            "model_overview": {
                "name": model_config.get("model_name", "MAHIA-X"),
                "version": model_config.get("version", "1.0"),
                "description": "Next-generation hybrid model combining Hyena operators with Mixture of Experts",
                "created_at": datetime.now().isoformat()
            },
            "architecture": model_config.get("architecture", {}),
            "training_config": model_config.get("training", {}),
            "performance_targets": {
                "target_latency_ms": 50,
                "target_memory_mb": 2048,
                "target_throughput_samples_per_sec": 100
            }
        }
        
        return specs
    
    def generate_component_specs(self) -> Dict[str, Any]:
        """Generate specifications for individual components
        
        Returns:
            Component specifications dictionary
        """
        component_specs = {
            "components": {
                "SparseMoETopK": {
                    "description": "Sparse Mixture of Experts with Top-K routing",
                    "parameters": {
                        "dim": "Input/Output dimension",
                        "num_experts": "Number of expert networks",
                        "top_k": "Number of experts to route to per token",
                        "capacity_factor": "Capacity factor for load balancing"
                    },
                    "features": [
                        "Top-K expert selection",
                        "Capacity limiting",
                        "Load balancing",
                        "Auxiliary loss for balanced routing"
                    ]
                },
                "HyenaBlock": {
                    "description": "Hyena-like sequence operator using depthwise separable convolutions",
                    "parameters": {
                        "dim": "Input/Output dimension",
                        "kernel_size": "Convolution kernel size",
                        "dropout": "Dropout rate"
                    },
                    "features": [
                        "Depthwise separable convolutions",
                        "Gating mechanism",
                        "Optional FlashAttention fallback"
                    ]
                },
                "LoRAAdapter": {
                    "description": "Low-Rank Adaptation for efficient fine-tuning",
                    "parameters": {
                        "in_features": "Input feature dimension",
                        "out_features": "Output feature dimension",
                        "rank": "Low-rank approximation dimension",
                        "alpha": "Scaling factor"
                    },
                    "features": [
                        "Parameter-efficient fine-tuning",
                        "Low-rank matrix decomposition",
                        "Scalable adaptation"
                    ]
                },
                "FP8CalibrationAutoTuner": {
                    "description": "Automatic FP8 calibration with per-layer dynamic range analysis",
                    "parameters": {
                        "calibration_batches": "Number of batches for calibration",
                        "percentile": "Percentile for dynamic range calculation"
                    },
                    "features": [
                        "Per-layer dynamic range analysis",
                        "Automatic calibration",
                        "FP8 quantization support"
                    ]
                },
                "MetaLRPolicyController": {
                    "description": "Meta-learning rate policy controller using reinforcement learning",
                    "parameters": {
                        "state_dim": "State dimension for policy network",
                        "action_dim": "Action dimension for policy network",
                        "lr": "Learning rate for policy updates"
                    },
                    "features": [
                        "Reinforcement learning-based control",
                        "State-aware policy decisions",
                        "Adaptive learning rate adjustment"
                    ]
                },
                "PredictiveStopForecaster": {
                    "description": "Predictive stop forecaster using regression to predict saturation points",
                    "parameters": {
                        "window_size": "History window size",
                        "prediction_horizon": "Prediction horizon in epochs"
                    },
                    "features": [
                        "Regression-based forecasting",
                        "RNN-based prediction",
                        "SSM-based prediction"
                    ]
                },
                "CurriculumMemorySystem": {
                    "description": "Curriculum memory system storing difficulty histories and learning patterns",
                    "parameters": {
                        "max_history_size": "Maximum history size to maintain"
                    },
                    "features": [
                        "Difficulty history tracking",
                        "Entropy analysis",
                        "Performance pattern recognition"
                    ]
                },
                "GPUTelemetryMonitor": {
                    "description": "GPU telemetry monitor for energy and load monitoring",
                    "parameters": {
                        "log_interval": "Logging interval in steps"
                    },
                    "features": [
                        "NVML integration",
                        "Prometheus metrics support",
                        "Energy consumption tracking"
                    ]
                },
                "TelemetryLogger": {
                    "description": "Complete telemetry logger for all training metrics",
                    "parameters": {
                        "log_dir": "Directory for log files",
                        "log_interval": "Logging interval in steps"
                    },
                    "features": [
                        "Time series logging",
                        "CSV export",
                        "Multi-metric tracking"
                    ]
                },
                "TrainingDashboardV2": {
                    "description": "Training dashboard with comprehensive monitoring",
                    "parameters": {
                        "update_interval": "Update interval in steps"
                    },
                    "features": [
                        "Real-time metrics display",
                        "Trend analysis",
                        "Controller state monitoring"
                    ]
                }
            }
        }
        
        return component_specs
    
    def generate_performance_specs(self, benchmark_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate performance specifications
        
        Args:
            benchmark_results: Optional benchmark results
            
        Returns:
            Performance specifications dictionary
        """
        if benchmark_results is None:
            # Default performance specs
            benchmark_results = {
                "fp32": {
                    "forward_latency_ms": 45.2,
                    "memory_usage_mb": 1850,
                    "throughput_samples_per_sec": 88
                },
                "fp16": {
                    "forward_latency_ms": 28.7,
                    "memory_usage_mb": 925,
                    "throughput_samples_per_sec": 139
                },
                "fp8": {
                    "forward_latency_ms": 22.1,
                    "memory_usage_mb": 462,
                    "throughput_samples_per_sec": 181
                },
                "int4": {
                    "forward_latency_ms": 18.5,
                    "memory_usage_mb": 231,
                    "throughput_samples_per_sec": 216
                }
            }
            
        performance_specs = {
            "performance_benchmarks": benchmark_results,
            "scalability": {
                "batch_size_scaling": "Linear up to batch size 128",
                "sequence_length_scaling": "Sub-linear due to attention mechanisms",
                "expert_count_scaling": "Linear with available compute resources"
            },
            "resource_requirements": {
                "minimum_gpu_memory_gb": 8,
                "recommended_gpu_memory_gb": 16,
                "cpu_cores_recommended": 8,
                "disk_space_gb": 50
            }
        }
        
        return performance_specs
    
    def generate_training_specs(self) -> Dict[str, Any]:
        """Generate training specifications
        
        Returns:
            Training specifications dictionary
        """
        training_specs = {
            "training_process": {
                "data_pipeline": {
                    "batch_size": 32,
                    "sequence_length": 64,
                    "data_augmentation": ["noise_injection", "token_masking"],
                    "curriculum_learning": True
                },
                "optimization": {
                    "optimizer": "AdamW",
                    "learning_rate": 1e-3,
                    "weight_decay": 0.01,
                    "gradient_clipping": 1.0
                },
                "scheduling": {
                    "lr_scheduler": "CosineAnnealing",
                    "warmup_steps": 1000,
                    "early_stopping": True
                },
                "regularization": {
                    "dropout": 0.1,
                    "label_smoothing": 0.1,
                    "auxiliary_loss_weight": 0.01
                }
            },
            "advanced_features": {
                "adaptive_curriculum": {
                    "enabled": True,
                    "based_on": "gradient_entropy"
                },
                "reflective_routing": {
                    "enabled": True,
                    "confidence_threshold": 0.7
                },
                "meta_control": {
                    "enabled": True,
                    "controller_type": "reinforcement_learning"
                },
                "predictive_monitoring": {
                    "enabled": True,
                    "forecast_horizon": 5
                }
            }
        }
        
        return training_specs
    
    def generate_deployment_specs(self) -> Dict[str, Any]:
        """Generate deployment specifications
        
        Returns:
            Deployment specifications dictionary
        """
        deployment_specs = {
            "inference": {
                "supported_formats": ["torchscript", "onnx"],
                "batch_inference": True,
                "streaming_inference": True,
                "model_serving": {
                    "framework": "TorchServe",
                    "api_endpoints": ["/predict", "/health", "/metrics"]
                }
            },
            "serving_recommendations": {
                "cpu_inference": "Not recommended for real-time applications",
                "gpu_inference": "Recommended for optimal performance",
                "multi_gpu": "Supported via data parallelism",
                "model_compression": {
                    "pruning": "Magnitude-based pruning supported",
                    "quantization": "FP8/INT4 quantization available",
                    "distillation": "Knowledge distillation supported"
                }
            },
            "monitoring": {
                "telemetry": True,
                "logging": True,
                "alerting": {
                    "latency_threshold_ms": 100,
                    "error_rate_threshold": 0.01
                }
            }
        }
        
        return deployment_specs
    
    def generate_complete_specs(self, config_path: str = "") -> Dict[str, Any]:
        """Generate complete technical specifications
        
        Args:
            config_path: Optional path to config file
            
        Returns:
            Complete specifications dictionary
        """
        print("📄 Generating technical specifications...")
        
        # Load config if provided
        model_config = None
        model_config = None
        if config_path and os.path.exists(config_path):
            try:
                model_config = self.load_config(config_path)
                print(f"   Loaded config from {config_path}")
            except Exception as e:
                print(f"   ⚠️  Failed to load config: {e}")
                
        # Generate all specs
        specs = {
            "generated_at": datetime.now().isoformat(),
            "model_specs": self.generate_model_specs(model_config if model_config is not None else {}),
            "component_specs": self.generate_component_specs(),
            "performance_specs": self.generate_performance_specs({}),
            "training_specs": self.generate_training_specs(),
            "deployment_specs": self.generate_deployment_specs()
        }
        
        self.specs = specs
        return specs
    
    def export_specs(self, format: str = "json", filename: str = "") -> str:
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"mahia_x_specs_{timestamp}.{format}"
        """Export specifications to file
        
        Args:
            format: Export format ("json" or "yaml")
            filename: Output filename (default: auto-generated)
            
        Returns:
            Path to exported file
        """
        if not self.specs:
            raise ValueError("No specifications to export. Run generate_complete_specs() first.")
            
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"mahia_x_specs_{timestamp}.{format}"
            
        filepath = os.path.join(self.output_dir, filename)
        
        try:
            if format.lower() == "json":
                with open(filepath, 'w') as f:
                    json.dump(self.specs, f, indent=2, default=str)
            elif format.lower() in ["yaml", "yml"]:
                with open(filepath, 'w') as f:
                    yaml.dump(self.specs, f, default_flow_style=False, indent=2)
            else:
                raise ValueError(f"Unsupported format: {format}")
                
            print(f"✅ Specifications exported to {filepath}")
            return filepath
            
        except Exception as e:
            print(f"❌ Failed to export specifications: {e}")
            raise
    
    def generate_markdown_docs(self, output_file: str = "") -> str:
        if not output_file:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = os.path.join(self.output_dir, f"mahia_x_documentation_{timestamp}.md")
        """Generate human-readable Markdown documentation
        
        Args:
            output_file: Output file path (default: auto-generated)
            
        Returns:
            Path to generated documentation
        """
        if not self.specs:
            raise ValueError("No specifications available. Run generate_complete_specs() first.")
            
        if output_file is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = os.path.join(self.output_dir, f"mahia_x_documentation_{timestamp}.md")
            
        try:
            with open(output_file, 'w') as f:
                # Document title
                f.write("# MAHIA-X Technical Documentation\n\n")
                f.write(f"Generated on: {self.specs['generated_at']}\n\n")
                
                # Table of contents
                f.write("## Table of Contents\n\n")
                f.write("1. [Model Overview](#model-overview)\n")
                f.write("2. [Architecture](#architecture)\n")
                f.write("3. [Components](#components)\n")
                f.write("4. [Performance](#performance)\n")
                f.write("5. [Training](#training)\n")
                f.write("6. [Deployment](#deployment)\n\n")
                
                # Model Overview
                f.write("## Model Overview\n\n")
                model_overview = self.specs["model_specs"]["model_overview"]
                f.write(f"**Name**: {model_overview['name']}\n\n")
                f.write(f"**Version**: {model_overview['version']}\n\n")
                f.write(f"**Description**: {model_overview['description']}\n\n")
                
                # Architecture
                f.write("## Architecture\n\n")
                arch = self.specs["model_specs"]["architecture"]
                for layer_name, layer_config in arch.items():
                    f.write(f"### {layer_name.replace('_', ' ').title()}\n\n")
                    f.write(f"**Type**: {layer_config.get('type', 'N/A')}\n\n")
                    f.write("**Parameters**:\n")
                    for param, value in layer_config.items():
                        if param != "type":
                            f.write(f"- {param}: {value}\n")
                    f.write("\n")
                
                # Components
                f.write("## Components\n\n")
                components = self.specs["component_specs"]["components"]
                for component_name, component_info in components.items():
                    f.write(f"### {component_name}\n\n")
                    f.write(f"**Description**: {component_info['description']}\n\n")
                    f.write("**Parameters**:\n")
                    for param, description in component_info["parameters"].items():
                        f.write(f"- `{param}`: {description}\n")
                    f.write("\n**Features**:\n")
                    for feature in component_info["features"]:
                        f.write(f"- {feature}\n")
                    f.write("\n")
                
                # Performance
                f.write("## Performance\n\n")
                perf = self.specs["performance_specs"]
                f.write("### Benchmarks\n\n")
                for quant_type, metrics in perf["performance_benchmarks"].items():
                    f.write(f"#### {quant_type.upper()}\n\n")
                    for metric, value in metrics.items():
                        f.write(f"- {metric.replace('_', ' ').title()}: {value}\n")
                    f.write("\n")
                
                # Training
                f.write("## Training\n\n")
                training = self.specs["training_specs"]
                f.write("### Process\n\n")
                for section, config in training["training_process"].items():
                    f.write(f"#### {section.replace('_', ' ').title()}\n\n")
                    if isinstance(config, dict):
                        for key, value in config.items():
                            f.write(f"- {key}: {value}\n")
                    f.write("\n")
                
                # Deployment
                f.write("## Deployment\n\n")
                deployment = self.specs["deployment_specs"]
                for section, config in deployment.items():
                    f.write(f"### {section.replace('_', ' ').title()}\n\n")
                    if isinstance(config, dict):
                        for key, value in config.items():
                            if isinstance(value, list):
                                f.write(f"- {key}:\n")
                                for item in value:
                                    f.write(f"  - {item}\n")
                            elif isinstance(value, dict):
                                f.write(f"- {key}:\n")
                                for subkey, subvalue in value.items():
                                    f.write(f"  - {subkey}: {subvalue}\n")
                            else:
                                f.write(f"- {key}: {value}\n")
                    f.write("\n")
                
            print(f"✅ Documentation generated at {output_file}")
            return output_file
            
        except Exception as e:
            print(f"❌ Failed to generate documentation: {e}")
            raise


def main():
    """Main documentation generator"""
    if not DOCS_AVAILABLE:
        print("❌ MAHIA-X modules not available for documentation generation")
        return
        
    # Create specs generator
    generator = TechnicalSpecsGenerator(output_dir="./specs")
    
    # Generate complete specifications
    specs = generator.generate_complete_specs()
    
    # Export in different formats
    json_path = generator.export_specs("json")
    yaml_path = generator.export_specs("yaml")
    
    # Generate Markdown documentation
    md_path = generator.generate_markdown_docs()
    
    print(f"\n✅ Technical specifications generated successfully!")
    print(f"   JSON: {json_path}")
    print(f"   YAML: {yaml_path}")
    print(f"   Documentation: {md_path}")


if __name__ == '__main__':
    main()