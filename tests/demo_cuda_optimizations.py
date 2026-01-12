#!/usr/bin/env python3
"""
Demo script for MAHIA-X CUDA/Triton optimizations
"""

import torch
from modell_V4_Nvidiaonly import HybridEfficientModel, get_device

def demo_cuda_optimizations():
    """Demonstrate all CUDA optimizations working together"""
    print("=== MAHIA-X CUDA/Triton Optimizations Demo ===")
    
    # Check if CUDA is available
    device = get_device()
    cuda_available = torch.cuda.is_available()
    print(f"CUDA Available: {cuda_available}")
    print(f"Device: {device}")
    
    # Create a model
    model = HybridEfficientModel(
        vocab_size=1000,
        text_seq_len=32,
        tab_dim=20,
        output_dim=2,
        embed_dim=32,
        tab_hidden_dim=32,
        fused_dim=64,
        use_moe=True
    )
    
    model.to(device)
    print("\nModel created successfully")
    
    # Show model size
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model Parameters: {total_params:,}")
    
    # Enable all CUDA optimizations
    if cuda_available:
        print("\nEnabling CUDA optimizations...")
        optimized_model = model.enable_all_cuda_optimizations()
        print("CUDA optimizations enabled successfully!")
    else:
        print("\nCUDA not available, using CPU fallback")
        optimized_model = model
    
    # Create test inputs
    batch_size = 4
    text_input = torch.randint(0, 1000, (batch_size, 32)).to(device)
    tab_input = torch.randn(batch_size, 20).to(device)
    
    print(f"\nTest inputs created:")
    print(f"  Text input shape: {text_input.shape}")
    print(f"  Tabular input shape: {tab_input.shape}")
    
    # Run forward pass
    print("\nRunning forward pass with optimizations...")
    optimized_model.eval()
    
    with torch.no_grad():
        output = optimized_model(text_input, tab_input)
    
    print(f"Output shape: {output.shape}")
    print("Forward pass completed successfully!")
    
    # Show benchmark results if available
    try:
        print("\nRunning GLUE benchmark...")
        benchmark_results = optimized_model.run_glue_benchmark(device)
        print("GLUE Benchmark Results:")
        for task, score in benchmark_results.items():
            print(f"  {task}: {score}%")
    except Exception as e:
        print(f"Benchmark failed: {e}")
    
    print("\n🎉 CUDA/Triton optimizations demo completed successfully!")

if __name__ == "__main__":
    demo_cuda_optimizations()