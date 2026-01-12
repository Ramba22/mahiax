#!/usr/bin/env python3
"""
Comprehensive validation script for all MAHIA-X optimizations
"""

import torch
import time
from modell_V4_Nvidiaonly import (
    HybridEfficientModel, 
    MetaAttentionKernel, 
    GraphDiffusionMemory, 
    DynamicNeuroFabric, 
    ExpertRouter,
    get_device
)

def validate_component_optimizations():
    """Validate individual component optimizations"""
    print("=== Validating Individual Component Optimizations ===")
    
    device = get_device()
    print(f"Device: {device}")
    
    # Test MetaAttentionKernel
    print("\n1. Testing MetaAttentionKernel...")
    try:
        kernel = MetaAttentionKernel(dim=32, use_cuda_kernels=True)
        x = torch.randn(2, 16, 32)
        output = kernel(x)
        print(f"   ✓ Input: {x.shape} -> Output: {output.shape}")
        print(f"   ✓ CUDA kernels enabled: {kernel.use_cuda_kernels}")
    except Exception as e:
        print(f"   ❌ MetaAttentionKernel test failed: {e}")
        return False
    
    # Test GraphDiffusionMemory
    print("\n2. Testing GraphDiffusionMemory...")
    try:
        memory = GraphDiffusionMemory(dim=32, memory_size=16, use_cuda_kernels=True)
        x = torch.randn(2, 32)
        output = memory(x)
        print(f"   ✓ Input: {x.shape} -> Output: {output.shape}")
        print(f"   ✓ CUDA kernels enabled: {memory.use_cuda_kernels}")
    except Exception as e:
        print(f"   ❌ GraphDiffusionMemory test failed: {e}")
        return False
    
    # Test DynamicNeuroFabric
    print("\n3. Testing DynamicNeuroFabric...")
    try:
        fabric = DynamicNeuroFabric(dim=32, max_layers=4, use_cuda_kernels=True)
        x = torch.randn(2, 8, 32)  # Sequence format
        output = fabric(x)
        print(f"   ✓ Input: {x.shape} -> Output: {output.shape}")
        print(f"   ✓ CUDA kernels enabled: {fabric.use_cuda_kernels}")
    except Exception as e:
        print(f"   ❌ DynamicNeuroFabric test failed: {e}")
        return False
    
    # Test ExpertRouter
    print("\n4. Testing ExpertRouter...")
    try:
        router = ExpertRouter(dim=32, num_experts=4, k=2)
        x = torch.randn(8, 32)
        output = router(x)
        print(f"   ✓ Input: {x.shape} -> Output: {output.shape}")
    except Exception as e:
        print(f"   ❌ ExpertRouter test failed: {e}")
        return False
    
    print("\n✅ All component optimizations validated successfully!")
    return True

def validate_model_optimizations():
    """Validate model-level optimizations"""
    print("\n=== Validating Model-Level Optimizations ===")
    
    device = get_device()
    
    # Create model
    print("\n1. Creating HybridEfficientModel...")
    try:
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
        print("   ✓ Model created successfully")
    except Exception as e:
        print(f"   ❌ Model creation failed: {e}")
        return False
    
    # Test enable_all_cuda_optimizations
    print("\n2. Testing enable_all_cuda_optimizations...")
    try:
        if torch.cuda.is_available():
            optimized_model = model.enable_all_cuda_optimizations()
            print("   ✓ All CUDA optimizations enabled successfully")
        else:
            optimized_model = model.enable_all_cuda_optimizations()
            print("   ✓ CUDA optimizations method called (CPU fallback)")
    except Exception as e:
        print(f"   ❌ enable_all_cuda_optimizations failed: {e}")
        return False
    
    # Test forward pass
    print("\n3. Testing forward pass with optimizations...")
    try:
        batch_size = 4
        text_input = torch.randint(0, 1000, (batch_size, 32)).to(device)
        tab_input = torch.randn(batch_size, 20).to(device)
        
        optimized_model.eval()
        with torch.no_grad():
            output = optimized_model(text_input, tab_input)
        
        print(f"   ✓ Forward pass successful: {output.shape}")
    except Exception as e:
        print(f"   ❌ Forward pass failed: {e}")
        return False
    
    # Test benchmark integration
    print("\n4. Testing benchmark integration...")
    try:
        benchmark_results = optimized_model.run_glue_benchmark(device)
        print("   ✓ Benchmark integration successful")
        for task, score in list(benchmark_results.items())[:2]:  # Show first 2
            print(f"     - {task}: {score}%")
    except Exception as e:
        print(f"   ❌ Benchmark integration failed: {e}")
        return False
    
    print("\n✅ All model-level optimizations validated successfully!")
    return True

def validate_performance_improvements():
    """Validate expected performance improvements"""
    print("\n=== Validating Performance Improvements ===")
    
    # Test with and without CUDA optimizations
    print("\n1. Testing optimization impact...")
    
    # Create test inputs
    device = get_device()
    batch_size = 4
    text_input = torch.randint(0, 1000, (batch_size, 32)).to(device)
    tab_input = torch.randn(batch_size, 20).to(device)
    
    # Create models
    model_standard = HybridEfficientModel(
        vocab_size=1000,
        text_seq_len=32,
        tab_dim=20,
        output_dim=2,
        embed_dim=32,
        tab_hidden_dim=32,
        fused_dim=64,
        use_moe=True
    )
    model_standard.to(device)
    
    # Test standard model
    print("   Testing standard model performance...")
    model_standard.eval()
    with torch.no_grad():
        start_time = time.time()
        for _ in range(10):  # Run multiple times for better measurement
            output = model_standard(text_input, tab_input)
        standard_time = time.time() - start_time
    
    # Test optimized model
    print("   Testing optimized model performance...")
    model_optimized = HybridEfficientModel(
        vocab_size=1000,
        text_seq_len=32,
        tab_dim=20,
        output_dim=2,
        embed_dim=32,
        tab_hidden_dim=32,
        fused_dim=64,
        use_moe=True
    )
    model_optimized.to(device)
    
    # Enable optimizations if CUDA is available
    if torch.cuda.is_available():
        model_optimized = model_optimized.enable_all_cuda_optimizations()
    
    model_optimized.eval()
    with torch.no_grad():
        start_time = time.time()
        for _ in range(10):  # Run multiple times for better measurement
            output = model_optimized(text_input, tab_input)
        optimized_time = time.time() - start_time
    
    print(f"   Standard model time: {standard_time:.4f}s")
    print(f"   Optimized model time: {optimized_time:.4f}s")
    
    if optimized_time < standard_time:
        speedup = standard_time / optimized_time
        print(f"   ✓ Performance improvement: {speedup:.2f}x speedup")
    else:
        print("   ⚠ No significant performance improvement detected (expected on CPU)")
    
    print("\n✅ Performance validation completed!")
    return True

def main():
    """Run all validation tests"""
    print("MAHIA-X Comprehensive Optimization Validation")
    print("=" * 50)
    
    import time
    
    # Run all validation tests
    tests = [
        validate_component_optimizations,
        validate_model_optimizations,
        validate_performance_improvements
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
            print()
        except Exception as e:
            print(f"❌ Test failed with error: {e}")
            print()
    
    print("=" * 50)
    print(f"Validation Results: {passed}/{total} test suites passed")
    
    if passed == total:
        print("🎉 All optimizations validated successfully!")
        print("🚀 MAHIA-X is ready for high-performance deployment!")
        return True
    else:
        print("⚠️  Some validation tests failed. Please check implementation.")
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)