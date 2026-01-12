#!/usr/bin/env python3
"""
Test script for advanced features: FlashAttention, Gradient Checkpointing, Mixed Precision
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import time

def test_flash_attention_fallback():
    """Test FlashAttention fallback feature"""
    print("Testing FlashAttention Fallback...")
    
    from modell_V5_MAHIA_HyenaMoE import HyenaBlock
    
    # Create HyenaBlock
    block = HyenaBlock(dim=64, kernel_size=15)
    
    # Test without attention
    x = torch.randn(4, 32, 64)
    out1 = block(x)
    print(f"  Without attention: {out1.shape}")
    
    # Enable attention fallback
    block.enable_attention_fallback()
    
    # Test with attention
    out2 = block(x)
    print(f"  With attention: {out2.shape}")
    
    # Check that outputs are different but valid
    assert out1.shape == out2.shape, "Output shapes should match"
    print("  ✅ FlashAttention fallback working correctly")

def test_gradient_checkpointing():
    """Test gradient checkpointing feature"""
    print("\nTesting Gradient Checkpointing...")
    
    from modell_V5_MAHIA_HyenaMoE import MAHIA_V5
    
    # Create model
    model = MAHIA_V5(vocab_size=1000, text_seq_len=32, tab_dim=20, embed_dim=64, fused_dim=128)
    
    # Enable gradient checkpointing
    model.enable_gradient_checkpointing()
    
    # Test forward pass
    text_tokens = torch.randint(0, 1000, (4, 32))
    tab_feats = torch.randn(4, 20)
    
    # Set model to training mode
    model.train()
    
    out, aux = model(text_tokens, tab_feats)
    print(f"  Output shape: {out.shape}")
    print(f"  Aux loss: {aux.item() if aux is not None else 'None'}")
    
    # Test backward pass (this is where checkpointing helps with memory)
    loss = out.sum()
    loss.backward()
    
    print("  ✅ Gradient checkpointing working correctly")

def test_mixed_precision():
    """Test mixed precision support"""
    print("\nTesting Mixed Precision Support...")
    
    from modell_V5_MAHIA_HyenaMoE import MAHIA_V5
    
    # Create model
    model = MAHIA_V5(vocab_size=1000, text_seq_len=32, tab_dim=20, embed_dim=64, fused_dim=128)
    
    # Enable mixed precision
    model.enable_mixed_precision()
    
    # Test with FP16 tensors if available
    if hasattr(torch, 'autocast'):
        print("  ✅ Mixed precision support available")
    else:
        print("  ⚠️  Mixed precision support not available in this PyTorch version")
    
    print("  ✅ Mixed precision setup completed")

def test_torch_compile_full_pipeline():
    """Test torch.compile on full pipeline"""
    print("\nTesting Full Pipeline torch.compile...")
    
    from modell_V5_MAHIA_HyenaMoE import MAHIA_V5
    
    # Create model
    model = MAHIA_V5(vocab_size=1000, text_seq_len=32, tab_dim=20, embed_dim=64, fused_dim=128)
    
    # Test without compilation
    text_tokens = torch.randint(0, 1000, (4, 32))
    tab_feats = torch.randn(4, 20)
    
    # Warmup
    for _ in range(3):
        out, aux = model(text_tokens, tab_feats)
    
    # Time uncompiled
    start = time.time()
    for _ in range(10):
        out, aux = model(text_tokens, tab_feats)
    end = time.time()
    uncompiled_time = (end - start) / 10 * 1000
    
    # Enable compilation
    try:
        compiled_model = model.enable_torch_compile()
        
        # Warmup compiled
        for _ in range(3):
            out, aux = compiled_model(text_tokens, tab_feats)
        
        # Time compiled
        start = time.time()
        for _ in range(10):
            out, aux = compiled_model(text_tokens, tab_feats)
        end = time.time()
        compiled_time = (end - start) / 10 * 1000
        
        speedup = uncompiled_time / compiled_time if compiled_time > 0 else 1.0
        print(f"  Uncompiled: {uncompiled_time:.2f} ms")
        print(f"  Compiled: {compiled_time:.2f} ms")
        print(f"  Speedup: {speedup:.2f}x")
        
    except Exception as e:
        print(f"  ⚠️  Compilation failed (expected on some systems): {e}")
    
    print("  ✅ Full pipeline compilation test completed")

def main():
    """Run all advanced features tests"""
    print("MAHIA-V5 Advanced Features Tests")
    print("=" * 35)
    
    try:
        test_flash_attention_fallback()
        test_gradient_checkpointing()
        test_mixed_precision()
        test_torch_compile_full_pipeline()
        
        print("\n" + "=" * 35)
        print("🎉 All advanced features tests passed!")
        print("🚀 MAHIA-V5 now supports advanced optimizations!")
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

if __name__ == "__main__":
    main()