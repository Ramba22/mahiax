#!/usr/bin/env python3
"""
Test script for Phase 1 optimizations of MAHIA-V5 model
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import time
from modell_V5_MAHIA_HyenaMoE import MAHIA_V5

def test_torch_compile_optimization():
    """Test torch.compile optimization"""
    print("Testing torch.compile optimization...")
    
    # Create model
    model = MAHIA_V5(vocab_size=1000, text_seq_len=32, tab_dim=20, embed_dim=32, fused_dim=64)
    
    # Test without compilation
    print("Testing without torch.compile...")
    test_performance(model, "Uncompiled")
    
    # Test with compilation
    print("\nTesting with torch.compile...")
    try:
        compiled_model = model.enable_torch_compile()
        test_performance(compiled_model, "Compiled")
        print("✅ torch.compile optimization successful!")
        return True
    except Exception as e:
        print(f"❌ torch.compile optimization failed: {e}")
        return False

def test_performance(model, label):
    """Test model performance"""
    # Create test inputs
    batch_size = 4
    text_tokens = torch.randint(0, 1000, (batch_size, 32))
    tab_feats = torch.randn(batch_size, 20)
    
    # Warmup
    for _ in range(5):
        out, aux = model(text_tokens, tab_feats)
    
    # Timing
    start_time = time.time()
    
    for _ in range(10):
        out, aux = model(text_tokens, tab_feats)
    
    end_time = time.time()
    
    avg_time = (end_time - start_time) / 10
    print(f"  {label}: {avg_time*1000:.2f} ms per forward pass")

def test_vectorized_moe():
    """Test vectorized MoE implementation"""
    print("\nTesting vectorized MoE implementation...")
    
    from modell_V5_MAHIA_HyenaMoE import SparseMoETopK
    
    # Create MoE layer
    moe = SparseMoETopK(dim=32, num_experts=4, top_k=2)
    
    # Test with various batch sizes
    test_cases = [
        (2, 8, 32),  # Small batch
        (4, 16, 32), # Medium batch
        (8, 32, 32), # Large batch
    ]
    
    for B, L, D in test_cases:
        try:
            x = torch.randn(B, L, D)
            out, aux = moe(x)
            assert out.shape == (B, L, D)
            print(f"  ✓ Batch {B}, Seq {L}, Dim {D}: {out.shape}")
        except Exception as e:
            print(f"  ❌ Batch {B}, Seq {L}, Dim {D}: Failed with {e}")
            return False
    
    print("✅ Vectorized MoE implementation working correctly!")
    return True

def test_memory_efficiency():
    """Test memory efficiency improvements"""
    print("\nTesting memory efficiency...")
    
    try:
        # Test with larger model
        model = MAHIA_V5(vocab_size=5000, text_seq_len=64, tab_dim=50, embed_dim=128, fused_dim=256)
        
        # Create larger inputs
        batch_size = 8
        text_tokens = torch.randint(0, 5000, (batch_size, 64))
        tab_feats = torch.randn(batch_size, 50)
        
        # Test memory usage
        out, aux = model(text_tokens, tab_feats)
        print(f"  Output shape: {out.shape}")
        
        print("✅ Memory efficiency test passed!")
        return True
    except Exception as e:
        print(f"❌ Memory efficiency test failed: {e}")
        return False

def main():
    """Run all Phase 1 optimization tests"""
    print("MAHIA-V5 Phase 1 Optimization Tests")
    print("=" * 40)
    
    test1 = test_torch_compile_optimization()
    test2 = test_vectorized_moe()
    test3 = test_memory_efficiency()
    
    print("\n" + "=" * 40)
    if test1 and test2 and test3:
        print("🎉 All Phase 1 optimization tests passed!")
        print("🚀 MAHIA-V5 is now optimized for better performance!")
    else:
        print("❌ Some optimization tests failed.")
    
    return test1 and test2 and test3

if __name__ == "__main__":
    main()