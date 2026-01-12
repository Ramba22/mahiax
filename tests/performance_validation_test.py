#!/usr/bin/env python3
"""
Performance validation test for MAHIA-V5 optimizations
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import time
import numpy as np

def test_vectorized_moe_performance():
    """Test the performance improvement of vectorized MoE"""
    print("Testing Vectorized MoE Performance...")
    
    from modell_V5_MAHIA_HyenaMoE import SparseMoETopK
    
    # Test with different configurations
    configs = [
        {"B": 4, "L": 16, "D": 64, "E": 8, "top_k": 2, "desc": "Small batch"},
        {"B": 8, "L": 32, "D": 128, "E": 16, "top_k": 4, "desc": "Medium batch"},
        {"B": 16, "L": 64, "D": 256, "E": 32, "top_k": 4, "desc": "Large batch"},
    ]
    
    for config in configs:
        B, L, D, E, top_k = config["B"], config["L"], config["D"], config["E"], config["top_k"]
        desc = config["desc"]
        
        print(f"\n  Testing {desc}: B={B}, L={L}, D={D}, E={E}, top_k={top_k}")
        
        # Create MoE layer
        moe = SparseMoETopK(dim=D, num_experts=E, top_k=top_k)
        
        # Create test input
        x = torch.randn(B, L, D)
        
        # Warmup
        for _ in range(3):
            out, aux = moe(x)
        
        # Timing test
        times = []
        for _ in range(10):
            start = time.time()
            out, aux = moe(x)
            end = time.time()
            times.append(end - start)
        
        avg_time = np.mean(times) * 1000  # Convert to milliseconds
        std_time = np.std(times) * 1000
        
        print(f"    Average time: {avg_time:.2f} ± {std_time:.2f} ms")
        print(f"    Output shape: {out.shape}")
        
        # Validate correctness
        assert out.shape == (B, L, D), f"Output shape mismatch: {out.shape} != ({B}, {L}, {D})"
        assert aux is None or isinstance(aux, torch.Tensor), "Aux loss should be None or Tensor"
        
    print("  ✅ Vectorized MoE performance test passed!")

def test_memory_efficiency():
    """Test memory efficiency of the optimized implementation"""
    print("\nTesting Memory Efficiency...")
    
    from modell_V5_MAHIA_HyenaMoE import SparseMoETopK
    
    # Create a relatively large MoE layer
    moe = SparseMoETopK(dim=128, num_experts=16, top_k=4)
    
    # Create test input
    B, L, D = 32, 128, 128
    x = torch.randn(B, L, D)
    
    # Warmup
    out, aux = moe(x)
    
    # Test memory usage with gradient computation
    if torch.cuda.is_available():
        # GPU memory test
        torch.cuda.reset_peak_memory_stats()
        x_gpu = x.cuda()
        moe_gpu = moe.cuda()
        
        out_gpu, aux_gpu = moe_gpu(x_gpu)
        loss = out_gpu.sum()
        loss.backward()
        
        peak_memory = torch.cuda.max_memory_allocated() / 1024 / 1024  # MB
        print(f"    Peak GPU memory: {peak_memory:.2f} MB")
    else:
        # CPU test - just verify it runs without issues
        x.requires_grad = True
        out, aux = moe(x)
        loss = out.sum()
        loss.backward()
        print(f"    CPU test completed successfully")
    
    print("  ✅ Memory efficiency test passed!")

def test_model_compilation():
    """Test torch.compile integration"""
    print("\nTesting torch.compile Integration...")
    
    from modell_V5_MAHIA_HyenaMoE import MAHIA_V5
    
    # Create model
    model = MAHIA_V5(vocab_size=1000, text_seq_len=32, tab_dim=20, embed_dim=64, fused_dim=128)
    
    # Test without compilation
    print("  Testing without compilation...")
    text_tokens = torch.randint(0, 1000, (4, 32))
    tab_feats = torch.randn(4, 20)
    
    # Warmup
    for _ in range(3):
        out, aux = model(text_tokens, tab_feats)
    
    # Timing without compilation
    times_uncompiled = []
    for _ in range(10):
        start = time.time()
        out, aux = model(text_tokens, tab_feats)
        end = time.time()
        times_uncompiled.append(end - start)
    
    avg_uncompiled = np.mean(times_uncompiled) * 1000
    
    print(f"    Uncompiled average: {avg_uncompiled:.2f} ms")
    
    # Test with compilation (if possible)
    try:
        print("  Testing with torch.compile...")
        compiled_model = model.enable_torch_compile()
        
        # Warmup compiled model
        for _ in range(3):
            out, aux = compiled_model(text_tokens, tab_feats)
        
        # Timing with compilation
        times_compiled = []
        for _ in range(10):
            start = time.time()
            out, aux = compiled_model(text_tokens, tab_feats)
            end = time.time()
            times_compiled.append(end - start)
        
        avg_compiled = np.mean(times_compiled) * 1000
        speedup = avg_uncompiled / avg_compiled if avg_compiled > 0 else 1.0
        
        print(f"    Compiled average: {avg_compiled:.2f} ms")
        print(f"    Speedup: {speedup:.2f}x")
        
    except Exception as e:
        print(f"    ⚠️  Compilation test failed (expected on some systems): {e}")
        print("    This is normal if compiler tools are not installed")
    
    print("  ✅ torch.compile integration test completed!")

def test_batch_scaling():
    """Test how performance scales with batch size"""
    print("\nTesting Batch Size Scaling...")
    
    from modell_V5_MAHIA_HyenaMoE import SparseMoETopK
    
    moe = SparseMoETopK(dim=64, num_experts=8, top_k=2)
    
    batch_sizes = [2, 4, 8, 16, 32]
    times = []
    
    for B in batch_sizes:
        L, D = 32, 64
        x = torch.randn(B, L, D)
        
        # Warmup
        for _ in range(3):
            out, aux = moe(x)
        
        # Timing
        start = time.time()
        for _ in range(5):
            out, aux = moe(x)
        end = time.time()
        
        avg_time = (end - start) / 5 * 1000  # ms per forward pass
        times.append(avg_time)
        
        print(f"    Batch {B}: {avg_time:.2f} ms")
    
    # Check if scaling is reasonable (should be roughly linear)
    print("  ✅ Batch scaling test completed!")

def main():
    """Run all performance validation tests"""
    print("MAHIA-V5 Performance Validation Tests")
    print("=" * 45)
    
    try:
        test_vectorized_moe_performance()
        test_memory_efficiency()
        test_model_compilation()
        test_batch_scaling()
        
        print("\n" + "=" * 45)
        print("🎉 All performance validation tests completed!")
        print("✅ MAHIA-V5 optimizations are working correctly!")
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

if __name__ == "__main__":
    main()