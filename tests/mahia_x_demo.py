#!/usr/bin/env python3
"""
MAHIA-X Demo Script
Demonstrates all the enhanced features of the MAHIA-X model.
"""

import torch
from modell_V4_Nvidiaonly import (
    HybridEfficientModel,
    train_example,
    run_inference_example,
    KnowledgeDistillationLoss,
    quantize_model_8bit,
    quantize_model_4bit,
    run_comprehensive_benchmark,
    get_device
)

def demo_gradient_clipping():
    """Demonstrate gradient clipping functionality"""
    print("=== Gradient Clipping Demo ===")
    print("Training with gradient norm clipping (clip_grad_norm=1.0)...")  # Updated value
    try:
        model = train_example(
            num_epochs=1,
            batch_size=16,
            print_every=2,
            clip_grad_norm=1.0,  # Updated from 0.5 to 1.0 for better training stability
            clip_grad_mode="norm"
        )
        print("Gradient clipping demo completed successfully!")
        return model
    except Exception as e:
        print(f"Gradient clipping demo failed: {e}")
        return None

def demo_profiling():
    """Demonstrate profiling functionality"""
    print("\n=== Profiling Demo ===")
    print("Running inference with profiling...")
    try:
        run_inference_example()
        print("Profiling demo completed successfully!")
    except Exception as e:
        print(f"Profiling demo failed: {e}")

def demo_quantization():
    """Demonstrate quantization functionality"""
    print("\n=== Quantization Demo ===")
    device = get_device()
    
    # Create a small model for quantization demo
    model = HybridEfficientModel(
        vocab_size=1000,
        text_seq_len=32,
        tab_dim=20,
        output_dim=2,
        embed_dim=32,
        tab_hidden_dim=32,
        fused_dim=64
    )
    model.to(device)
    
    print("Original model size:")
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {total_params:,}")
    
    try:
        # 8-bit quantization
        print("\nApplying 8-bit quantization...")
        quantized_model = quantize_model_8bit(model)
        print("8-bit quantization completed successfully!")
        
        # 4-bit quantization (commented out as it requires more setup)
        # print("\nApplying 4-bit quantization...")
        # quantized_model_nf4 = quantize_model_4bit(model, quant_type="nf4")
        # print("4-bit quantization completed successfully!")
        
    except Exception as e:
        print(f"Quantization demo failed: {e}")

def demo_distillation():
    """Demonstrate knowledge distillation functionality"""
    print("\n=== Knowledge Distillation Demo ===")
    device = get_device()
    
    # Enable anomaly detection for detailed error information
    torch.autograd.set_detect_anomaly(True)
    
    # Create teacher and student models
    teacher_model = HybridEfficientModel(
        vocab_size=10000,  # Match the training function's vocab_size
        text_seq_len=64,
        tab_dim=50,       # Match the training function's tab_dim
        output_dim=2,
        embed_dim=64,
        tab_hidden_dim=64,
        fused_dim=128
    )
    
    student_model = HybridEfficientModel(
        vocab_size=10000,  # Match the training function's vocab_size
        text_seq_len=64,
        tab_dim=50,       # Match the training function's tab_dim
        output_dim=2,
        embed_dim=32,  # Smaller student model
        tab_hidden_dim=32,
        fused_dim=64
    )
    
    teacher_model.to(device)
    student_model.to(device)
    
    print("Training student model with knowledge distillation...")
    try:
        # Train student with distillation from teacher (shorter training for demo)
        trained_student = train_example(
            num_epochs=1,
            batch_size=8,
            print_every=2,
            use_distillation=True,
            teacher_model=teacher_model
        )
        print("Knowledge distillation demo completed successfully!")
    except Exception as e:
        print(f"Knowledge distillation demo failed: {e}")
    finally:
        # Disable anomaly detection
        torch.autograd.set_detect_anomaly(False)

def demo_benchmarking():
    """Demonstrate benchmarking functionality"""
    print("\n=== Benchmarking Demo ===")
    device = get_device()
    
    # Create a model for benchmarking
    model = HybridEfficientModel(
        vocab_size=5000,
        text_seq_len=64,
        tab_dim=30,
        output_dim=2,
        embed_dim=32,
        tab_hidden_dim=32,
        fused_dim=64
    )
    model.to(device)
    
    print("Running comprehensive benchmark...")
    try:
        benchmark_results = run_comprehensive_benchmark(model, device)
        print("Benchmarking demo completed successfully!")
    except Exception as e:
        print(f"Benchmarking demo failed: {e}")

def main():
    """Run all demos"""
    print("MAHIA-X Enhanced Features Demo")
    print("=" * 50)
    
    # Run all demos
    demo_gradient_clipping()
    demo_profiling()
    demo_quantization()
    demo_distillation()
    demo_benchmarking()
    
    print("\n" + "=" * 50)
    print("All demos completed!")

if __name__ == "__main__":
    main()