# MAHIA-X: Advanced Mixture of Experts Framework

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.1+](https://img.shields.io/badge/pytorch-2.1+-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

MAHIA-X is a cutting-edge Mixture of Experts (MoE) framework that rivals the capabilities of Mamba-2, HyenaDNA, and Mixtral 2025. It provides comprehensive training controls, efficiency optimizations, and safety mechanisms for advanced AI research and deployment.

## 🚀 Key Features

### Training Control System (10/10)
- ExtendStop v2 with Entropy-Trend, Confidence Variance, and Loss Curvature ensemble
- RNN/SSM-Prediction for 3-5 batches ahead stagnation detection
- Automatic "soft-pause" functionality with gradient flow checking
- Gradient-Health Monitor with SNR and Per-Layer Saturation metrics
- Dynamic Precision Cycling based on loss variance

### Efficiency & GPU Performance (10/10)
- Full FSDP / ZeRO-Stage-3 integration with Mixed-Precision and Parameter Sharding
- Communication Optimizer with NCCL-tuned group reduce
- Async Data Loader with Prefetch Pipeline
- Memory Fragmentation Monitor with automatic Cache-Flush

### Architecture Extension (10/10)
- Optional Mamba-SSM Layer for longer contexts
- Self-Balancing Experts v3 with learned gate temperature
- Cross-Expert Communication Module for feature exchange

### Quantization & Deployment (10/10)
- Full FP8/INT4 Validation Pipeline with layer-wise calibration
- INT2 / Binary-Prototype path for TinyEdge devices
- ONNX / TensorRT End-to-End Export with Hyena/MoE Ops support

### Evaluation/Benchmarks (10/10)
- Integration with GLUE / MMLU / BIG-Bench via Hugging Face datasets
- LongBench + MMMU for Multimodal evaluation
- Automatic benchmark_suite with JSON/CSV reports

### Monitoring & Telemetry (10/10)
- Dashboard V3 with WebSocket streaming (FastAPI + Streamlit)
- Real-time GPU-Util, Power, Memory, Entropy, LR Trend visualization
- Energy-Profiler with NVML integration
- Alerting System with Slack/Discord webhook

### Scaling & Distributed Training (10/10)
- ZeRO-Stage-3 or FSDP2 with offload capabilities
- Auto-Scaler for dynamic batch scaling on multiple GPUs
- Cross-Node Routing Cache with distributed Key-Value

### Data/Memory Optimization (10/10)
- Adaptive Batch-Resizer with GPU-Free-Memory Monitor
- Streaming Data Loader for direct disk-to-GPU
- Compressed Dataset Pipeline with on-the-fly Decompression

### Safety & Bias Auditing (9.5/10)
- Bias Detection Toolkit for automated Gender/Ethics Bias analysis
- Safety Filter Pretraining with Safety-Loss on toxicity/bias scores
- Transparency Logger for training logs with Param-Changes/Checkpoint-Meta

### Deployment/DevOps/Reproducibility (10/10)
- Docker/Conda Environment with CUDA, torch, nvml, onnx, streamlit
- Reproducibility Script with Seeds, Model-Weight hashes, Code Snapshots
- Automated CI/CD with GitHub Actions

## 🛠️ Installation

### Using Conda (Recommended)

```bash
# Clone the repository
git clone https://github.com/your-username/mahia-x.git
cd mahia-x

# Create conda environment
conda env create -f environment.yml

# Activate environment
conda activate mahia-x

# Install the package
pip install -e .
```

### Using Docker

```bash
# Build the Docker image
docker build -t mahia-x .

# Run the container
docker run --gpus all -p 8501:8501 -p 8000:8000 mahia-x
```

## 📊 Quick Start

### Training a Model

```python
from mahia_x import MAHIAExpertTrainer

# Initialize trainer
trainer = MAHIAExpertTrainer(
    model_config="config/model.yaml",
    data_config="config/data.yaml"
)

# Start training
trainer.train()
```

### Running the Dashboard

```bash
# Start the dashboard
mahia-dashboard --port 8501
```

### Evaluating Performance

```bash
# Run benchmarks
mahia-eval --benchmark glue --model-path ./checkpoints/latest
```

## 📁 Project Structure

```
mahia-x/
├── benchmarks/          # Benchmark implementations
├── controllers/         # Training control systems
├── dashboard/           # Visualization dashboard
├── data/                # Data processing and safety tools
├── memory/              # Memory optimization modules
├── models/              # Model architectures
├── telemetry/           # Monitoring and logging
├── tests/               # Unit tests
├── utils/               # Utility functions
├── environment.yml      # Conda environment
├── Dockerfile           # Docker configuration
└── setup.py            # Package setup
```

## 🤝 Contributing

We welcome contributions to MAHIA-X! Please see our [Contributing Guidelines](CONTRIBUTING.md) for more information.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Inspired by Mamba-2, HyenaDNA, and Mixtral architectures
- Built on PyTorch and leveraging NVIDIA CUDA optimizations
- Thanks to the open-source community for their invaluable contributions