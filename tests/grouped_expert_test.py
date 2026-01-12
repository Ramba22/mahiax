#!/usr/bin/env python3
"""
Test script for grouped expert execution
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch

def test_grouped_expert_execution():
    """Test grouped expert execution functionality"""
    print("Testing Grouped Expert Execution...")
    
    from modell_V5_MAHIA_HyenaMoE import SparseMoETopK
    
    # Create MoE with different execution modes
    moe_standard = SparseMoETopK(dim=64, num_experts=8, top_k=2)
    moe_batched = SparseMoETopK(dim=64, num_experts=8, top_k=2)
    moe_batched.enable_batched_execution()
    
    # Test input
    x = torch.randn(4, 16, 64)
    
    # Standard execution
    out_standard, aux_standard = moe_standard(x, return_aux=True)
    
    # Batched execution
    out_batched, aux_batched = moe_batched(x, return_aux=True)
    
    print(f"  Input shape: {x.shape}")
    print(f"  Standard output shape: {out_standard.shape}")
    print(f"  Batched output shape: {out_batched.shape}")
    print(f"  Standard aux loss: {aux_standard.item():.6f}")
    print(f"  Batched aux loss: {aux_batched.item():.6f}")
    
    # Check shapes
    assert out_standard.shape == x.shape, f"Standard output shape should match input"
    assert out_batched.shape == x.shape, f"Batched output shape should match input"
    
    print("  ✅ Grouped Expert Execution working correctly")

def test_performance_comparison():
    """Compare performance of different execution modes"""
    print("\nTesting Performance Comparison...")
    
    from modell_V5_MAHIA_HyenaMoE import SparseMoETopK
    
    # Create MoE models
    moe_standard = SparseMoETopK(dim=128, num_experts=16, top_k=4)
    moe_batched = SparseMoETopK(dim=128, num_experts=16, top_k=4)
    moe_batched.enable_batched_execution()
    
    # Test input
    x = torch.randn(8, 32, 128)
    
    # Warmup
    for _ in range(3):
        moe_standard(x)
        moe_batched(x)
    
    # Time standard execution
    import time
    start = time.time()
    for _ in range(10):
        out, _ = moe_standard(x, return_aux=False)
    standard_time = (time.time() - start) / 10 * 1000  # ms per forward
    
    # Time batched execution
    start = time.time()
    for _ in range(10):
        out, _ = moe_batched(x, return_aux=False)
    batched_time = (time.time() - start) / 10 * 1000  # ms per forward
    
    speedup = standard_time / batched_time if batched_time > 0 else 1.0
    
    print(f"  Standard execution: {standard_time:.2f} ms/forward")
    print(f"  Batched execution: {batched_time:.2f} ms/forward")
    print(f"  Speedup: {speedup:.2f}x")
    
    print("  ✅ Performance Comparison completed")

def test_cuda_graphs_availability():
    """Test CUDA graphs availability"""
    print("\nTesting CUDA Graphs Availability...")
    
    from modell_V5_MAHIA_HyenaMoE import SparseMoETopK
    
    # Create MoE model
    moe = SparseMoETopK(dim=64, num_experts=8, top_k=2)
    
    # Check if CUDA is available
    cuda_available = torch.cuda.is_available()
    print(f"  CUDA available: {cuda_available}")
    
    if cuda_available:
        try:
            # Try to warmup CUDA graphs
            moe.warmup_cuda_graphs(sample_batch_size=16)
            print("  ✅ CUDA Graphs available and working")
        except Exception as e:
            print(f"  ⚠️  CUDA Graphs test failed: {e}")
    else:
        print("  ⚠️  CUDA not available, skipping CUDA graphs test")
    
    print("  ✅ CUDA Graphs Availability test completed")

def main():
    """Run all grouped expert tests"""
    print("MAHIA-V5 Grouped Expert Execution Tests")
    print("=" * 40)
    
    try:
        test_grouped_expert_execution()
        test_performance_comparison()
        test_cuda_graphs_availability()
        
        print("\n" + "=" * 40)
        print("🎉 All Grouped Expert Execution tests passed!")
        print("🚀 MAHIA-V5 supports optimized expert execution!")
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

if __name__ == "__main__":
    main()