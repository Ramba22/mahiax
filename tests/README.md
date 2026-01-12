# MAHIA-X: Meta-Adaptive Hyper-Intelligence Architecture

Enhanced version of the MAHIA-X model with additional features for stability, efficiency, and benchmarking.

## New Features Implemented

### 1. Gradient Norm Clipping for Training Stability
- Added configurable gradient clipping (norm-based and value-based)
- Gradient norm monitoring during training
- Support for mixed precision training with proper gradient scaling

### 2. Component Profiling
- Detailed profiling of model components (MetaAttentionKernel, GraphDiffusionMemory, etc.)
- Time and memory consumption measurements
- Formatted profiling reports

### 3. Quantization Support
- 4-bit and 8-bit quantization using bitsandbytes
- Integration with torchao for additional quantization options
- Model size reduction while maintaining performance

### 4. Enhanced Knowledge Distillation
- Multiple distillation modes (KL divergence, MSE, cosine similarity)
- Feature-level distillation support
- Configurable alpha and temperature parameters

### 5. Benchmarking Capabilities
- GLUE-style task benchmarking
- Tabular dataset benchmarking
- Multimodal sentiment analysis benchmarking
- Comprehensive benchmark reports

## Usage Examples

### Training with Gradient Clipping
```python
from modell_V4_Nvidiaonly import train_example

model = train_example(
    num_epochs=10,
    batch_size=32,
    clip_grad_norm=1.0,
    clip_grad_mode="norm"  # or "value" or "none"
)
```

### Running Profiling
```python
from modell_V4_Nvidiaonly import run_inference_example

# Profiling is automatically enabled in the inference example
run_inference_example()
```

### Quantizing Models
```python
from modell_V4_Nvidiaonly import HybridEfficientModel, quantize_model_8bit, quantize_model_4bit

model = HybridEfficientModel(...)
# 8-bit quantization
quantized_model = quantize_model_8bit(model)

# 4-bit quantization
quantized_model = quantize_model_4bit(model, quant_type="nf4")
```

### Knowledge Distillation
```python
from modell_V4_Nvidiaonly import train_example, HybridEfficientModel

teacher_model = HybridEfficientModel(...)  # Larger model
student_model = HybridEfficientModel(...)  # Smaller model

trained_student = train_example(
    num_epochs=10,
    use_distillation=True,
    teacher_model=teacher_model
)
```

### Running Benchmarks
```python
from modell_V4_Nvidiaonly import run_comprehensive_benchmark, HybridEfficientModel

model = HybridEfficientModel(...)
results = run_comprehensive_benchmark(model)
```

## Demo Script

Run the demo script to see all features in action:
```bash
python mahia_x_demo.py
```

## Requirements
- PyTorch
- bitsandbytes
- torchao
- psutil

Install additional requirements:
```bash
pip install bitsandbytes torchao
```