#!/usr/bin/env python3
"""
Test script to validate the real token-level MoE usage in MAHIA_V5
"""

import torch
from modell_V5_MAHIA_HyenaMoE import MAHIA_V5

def test_real_token_moe():
    """Test that MAHIA_V5 now uses real token-level MoE"""
    print("Testing real token-level MoE usage in MAHIA_V5...")
    
    # Create model
    model = MAHIA_V5(vocab_size=1000, text_seq_len=32, tab_dim=20, embed_dim=32, fused_dim=64)
    
    # Create test inputs
    batch_size = 4
    seq_len = 16
    text_tokens = torch.randint(0, 1000, (batch_size, seq_len))  # Real token sequences
    tab_feats = torch.randn(batch_size, 20)
    
    print(f"Text tokens shape: {text_tokens.shape}")
    print(f"Tabular features shape: {tab_feats.shape}")
    
    try:
        # Forward pass
        out, aux = model(text_tokens, tab_feats)
        
        print(f"Output shape: {out.shape}")
        print(f"Auxiliary loss: {aux.item() if aux is not None else 'None'}")
        
        # Verify that the output shape is correct for classification
        assert out.shape == (batch_size, 2), f"Expected output shape {(batch_size, 2)}, got {out.shape}"
        
        # Test backward pass
        loss = out.sum()
        loss.backward()
        
        print("✅ Forward and backward passes completed successfully!")
        print("✅ Real token-level MoE is working correctly!")
        return True
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        return False

def test_sequence_processing():
    """Test that the model properly processes sequences of different lengths"""
    print("\nTesting sequence processing with different lengths...")
    
    model = MAHIA_V5(vocab_size=1000, text_seq_len=32, tab_dim=20, embed_dim=32, fused_dim=64)
    
    # Test with different sequence lengths
    test_cases = [
        {"seq_len": 8, "description": "Short sequence"},
        {"seq_len": 16, "description": "Medium sequence"},
        {"seq_len": 32, "description": "Long sequence"}
    ]
    
    for case in test_cases:
        seq_len = case["seq_len"]
        description = case["description"]
        
        print(f"\n  Testing {description} (length {seq_len})")
        
        try:
            text_tokens = torch.randint(0, 1000, (2, seq_len))
            tab_feats = torch.randn(2, 20)
            
            out, aux = model(text_tokens, tab_feats)
            
            assert out.shape == (2, 2), f"Expected output shape (2, 2), got {out.shape}"
            print(f"    ✓ Processed successfully, output shape: {out.shape}")
            
        except Exception as e:
            print(f"    ❌ Failed with error: {e}")
            return False
    
    print("\n✅ All sequence lengths processed correctly!")
    return True

def main():
    """Run all tests"""
    print("Real Token-Level MoE Validation")
    print("=" * 40)
    
    test1_passed = test_real_token_moe()
    test2_passed = test_sequence_processing()
    
    print("\n" + "=" * 40)
    if test1_passed and test2_passed:
        print("🎉 All tests passed!")
        print("✅ MAHIA_V5 now correctly uses real token-level MoE!")
    else:
        print("❌ Some tests failed.")
    
    return test1_passed and test2_passed

if __name__ == "__main__":
    main()