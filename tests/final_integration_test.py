#!/usr/bin/env python3
"""
Final integration test to validate all enhancements work together
"""

import torch
from modell_V5_MAHIA_HyenaMoE import MAHIA_V5, SparseMoETopK

def test_mahia_v5_with_real_token_moe():
    """Test the enhanced MAHIA_V5 with real token-level MoE"""
    print("Testing enhanced MAHIA_V5 with real token-level MoE...")
    
    # Create model with various configurations
    model = MAHIA_V5(
        vocab_size=1000,
        text_seq_len=32,
        tab_dim=20,
        embed_dim=32,
        fused_dim=64,
        moe_experts=4,
        moe_topk=2
    )
    
    # Test with real token sequences
    batch_size = 3
    seq_len = 16
    text_tokens = torch.randint(0, 1000, (batch_size, seq_len))
    tab_feats = torch.randn(batch_size, 20)
    
    print(f"Input shapes - Text: {text_tokens.shape}, Tabular: {tab_feats.shape}")
    
    try:
        # Forward pass
        out, aux = model(text_tokens, tab_feats)
        
        print(f"Output shape: {out.shape}")
        print(f"Auxiliary loss: {aux.item():.6f}")
        
        # Verify correct output shape for binary classification
        assert out.shape == (batch_size, 2), f"Expected {(batch_size, 2)}, got {out.shape}"
        
        # Backward pass
        loss = out.sum()
        loss.backward()
        
        print("✅ MAHIA_V5 forward and backward passes successful!")
        return True
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        return False

def test_sparse_moe_edge_cases():
    """Test SparseMoETopK with various edge cases"""
    print("\nTesting SparseMoETopK edge cases...")
    
    test_cases = [
        # (B, L, D, E, top_k, description)
        (1, 4, 16, 2, 1, "Minimal batch"),
        (4, 8, 32, 4, 4, "top_k equals experts"),
        (2, 16, 64, 8, 4, "Standard case"),
        (8, 32, 128, 16, 4, "Large batch/sequence"),
    ]
    
    for B, L, D, E, top_k, description in test_cases:
        with torch.no_grad():  # Disable gradients for efficiency
            try:
                moe = SparseMoETopK(dim=D, num_experts=E, top_k=top_k)
                x = torch.randn(B, L, D)
                out, aux = moe(x, return_aux=True)
                
                assert out.shape == (B, L, D), f"Shape mismatch in {description}"
                assert aux.dim() == 0, f"Aux loss should be scalar in {description}"
                
                print(f"  ✓ {description}: {out.shape}")
                
            except Exception as e:
                print(f"  ❌ {description} failed: {e}")
                return False
    
    print("✅ All SparseMoETopK edge cases passed!")
    return True

def test_deterministic_behavior():
    """Test that models behave deterministically with fixed seeds"""
    print("\nTesting deterministic behavior...")
    
    def create_and_test_model(seed):
        torch.manual_seed(seed)
        model = SparseMoETopK(dim=32, num_experts=4, top_k=2)
        x = torch.randn(2, 8, 32)
        out, _ = model(x)
        return out.clone()
    
    # Test with same seed
    out1 = create_and_test_model(42)
    out2 = create_and_test_model(42)
    
    # Should be identical
    diff = (out1 - out2).abs().max().item()
    print(f"Maximum difference with same seed: {diff:.2e}")
    
    if diff < 1e-6:
        print("✅ Deterministic behavior confirmed!")
        return True
    else:
        print("❌ Deterministic behavior failed!")
        return False

def main():
    """Run all integration tests"""
    print("MAHIA-X Final Integration Test")
    print("=" * 40)
    
    test1 = test_mahia_v5_with_real_token_moe()
    test2 = test_sparse_moe_edge_cases()
    test3 = test_deterministic_behavior()
    
    print("\n" + "=" * 40)
    if test1 and test2 and test3:
        print("🎉 All integration tests passed!")
        print("✅ MAHIA-X enhancements are working correctly!")
    else:
        print("❌ Some integration tests failed.")
        print("⚠️  Please review the implementation.")
    
    return test1 and test2 and test3

if __name__ == "__main__":
    main()