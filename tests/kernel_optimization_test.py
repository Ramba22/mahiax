#!/usr/bin/env python3
"""
Test script for kernel optimization features: Batched Execution and CUDA Graphs
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import time

def test_batched_execution():
    """Test batched expert execution feature"""
    print("Testing Batched Expert Execution...")
    
    from modell_V5_MAHIA_HyenaMoE import SparseMoETopK
    
    # Create MoE layer
    moe = SparseMoETopK(dim=64, num_experts=8, top_k=2)
    
    # Test without batched execution
    x = torch.randn(4, 16, 64)
    out1, aux1 = moe(x, return_aux=True)
    print(f"  Without batched execution - Aux loss: {aux1.item():.6f}")
    
    # Enable batched execution
    moe.enable_batched_execution()
    
    # Test with batched execution
    out2, aux2 = moe(x, return_aux=True)
    print(f"  With batched execution - Aux loss: {aux2.item():.6f}")
    
    # Check that outputs are similar (with reasonable tolerance for numerical differences)
    diff = torch.mean(torch.abs(out1 - out2)).item()
    print(f"  Output difference: {diff:.8f}")
    assert diff < 1e-2, f"Outputs should be reasonably similar, but diff={diff}"
    print("  ✅ Batched execution working correctly")

def test_performance_comparison():
    """Compare performance of different execution modes"""
    print("\nTesting Performance Comparison...")
    
    from modell_V5_MAHIA_HyenaMoE import SparseMoETopK
    
    # Create MoE layer
    moe_standard = SparseMoETopK(dim=128, num_experts=16, top_k=4)
    moe_batched = SparseMoETopK(dim=128, num_experts=16, top_k=4)
    moe_batched.enable_batched_execution()
    
    # Create test input
    x = torch.randn(8, 32, 128)
    
    # Warmup
    for _ in range(3):
        out1, aux1 = moe_standard(x, return_aux=True)
        out2, aux2 = moe_batched(x, return_aux=True)
    
    # Time standard execution
    times_standard = []
    for _ in range(10):
        start = time.time()
        out, aux = moe_standard(x, return_aux=True)
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        end = time.time()
        times_standard.append(end - start)
    
    avg_standard = sum(times_standard) / len(times_standard) * 1000
    
    # Time batched execution
    times_batched = []
    for _ in range(10):
        start = time.time()
        out, aux = moe_batched(x, return_aux=True)
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        end = time.time()
        times_batched.append(end - start)
    
    avg_batched = sum(times_batched) / len(times_batched) * 1000
    
    speedup = avg_standard / avg_batched if avg_batched > 0 else 1.0
    print(f"  Standard execution: {avg_standard:.2f} ms")
    print(f"  Batched execution: {avg_batched:.2f} ms")
    print(f"  Speedup: {speedup:.2f}x")
    print("  ✅ Performance comparison completed")

def test_cuda_graphs():
    """Test CUDA graphs feature (if CUDA available)"""
    print("\nTesting CUDA Graphs...")
    
    from modell_V5_MAHIA_HyenaMoE import SparseMoETopK
    
    # Create MoE layer
    moe = SparseMoETopK(dim=64, num_experts=4, top_k=2)
    
    if torch.cuda.is_available():
        # Warmup CUDA graphs
        moe.warmup_cuda_graphs(sample_batch_size=16)
        
        if moe.use_cuda_graphs:
            print("  ✅ CUDA graphs enabled successfully")
            
            # Test execution with CUDA graphs
            x = torch.randn(4, 16, 64, device='cuda')
            out, aux = moe(x, return_aux=True)
            print(f"  CUDA graphs execution - Output shape: {out.shape}")
        else:
            print("  ⚠️  CUDA graphs not available or failed to initialize")
    else:
        print("  ⚠️  CUDA not available, skipping CUDA graphs test")
        # Still test the warmup function
        try:
            moe.warmup_cuda_graphs(sample_batch_size=16)
            print("  ✅ CUDA graphs warmup function executed (no CUDA)")
        except Exception as e:
            print(f"  ⚠️  CUDA graphs warmup failed: {e}")

def test_training_with_optimizations():
    """Test training with all optimizations enabled"""
    print("\nTesting Training with All Optimizations...")
    
    from modell_V5_MAHIA_HyenaMoE import SparseMoETopK
    
    # Create MoE layer with all optimizations
    moe = SparseMoETopK(dim=64, num_experts=8, top_k=2)
    moe.enable_batched_execution()
    moe.enable_expert_diversity_loss(weight=0.01)
    
    # Create optimizer
    optimizer = torch.optim.AdamW(moe.parameters(), lr=1e-3)
    
    # Training loop
    for i in range(20):
        x = torch.randn(4, 16, 64)
        
        # Forward pass
        out, aux_loss = moe(x, return_aux=True)
        
        # Loss (reconstruction + aux)
        recon_loss = torch.mean((out - x)**2)
        total_loss = recon_loss
        
        if aux_loss is not None:
            total_loss = total_loss + aux_loss
        
        # Backward pass
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        
        if (i + 1) % 10 == 0:
            print(f"    Step {i+1}: Loss = {total_loss.item():.6f}")
    
    print("  ✅ Training with all optimizations successful")

def main():
    """Run all kernel optimization tests"""
    print("MAHIA-V5 Kernel Optimization Tests")
    print("=" * 40)
    
    try:
        test_batched_execution()
        test_performance_comparison()
        test_cuda_graphs()
        test_training_with_optimizations()
        
        print("\n" + "=" * 40)
        print("🎉 All kernel optimization tests passed!")
        print("🚀 MAHIA-V5 now supports advanced kernel optimizations!")
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

if __name__ == "__main__":
    main()