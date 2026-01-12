#!/usr/bin/env python3
"""
Specific test to validate the broadcast/shape fix in SparseMoETopK
This test reproduces the exact issue mentioned in the problem description:
- expert_counts had shape (B, 1, E) due to keepdim=True
- expert_inputs had shape (B, E, D)
- Division failed with "The size of tensor a (32) must match the size of tensor b (4) at non-singleton dimension 2"
"""

import torch
from modell_V5_MAHIA_HyenaMoE import SparseMoETopK

def test_broadcast_shape_fix():
    """Test the specific broadcast/shape issue that was fixed"""
    print("Testing broadcast/shape fix in SparseMoETopK...")
    
    # Create a SparseMoETopK with parameters that would trigger the issue
    # B=4, L=8, D=32, E=4, top_k=2
    moe = SparseMoETopK(dim=32, num_experts=4, top_k=2)
    
    # Create test input that matches the scenario described in the issue
    B, L, D, E = 4, 8, 32, 4
    x = torch.randn(B, L, D)
    
    print(f"Input shape: {x.shape}")
    print(f"Number of experts: {E}")
    print(f"Top-K: 2")
    
    try:
        # Forward pass
        out, aux_loss = moe(x, return_aux=True)
        
        # Check shapes
        assert out.shape == x.shape, f"Output shape mismatch: {out.shape} != {x.shape}"
        
        print(f"Output shape: {out.shape}")
        print(f"Auxiliary loss: {aux_loss.item():.6f}")
        
        # Test backward pass (this is where the original error would occur)
        loss = out.sum()
        loss.backward()
        
        print("✅ Forward and backward passes completed successfully!")
        print("✅ Broadcast/shape issue has been fixed!")
        return True
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        return False

def test_original_error_conditions():
    """Test conditions that would cause the original error"""
    print("\nTesting conditions that would cause the original error...")
    
    # The original error was:
    # "The size of tensor a (32) must match the size of tensor b (4) at non-singleton dimension 2"
    # This suggests:
    # - tensor a had size 32 (likely D dimension)
    # - tensor b had size 4 (likely E dimension)
    # - The error occurred at dimension 2
    
    # Let's create a scenario that would match this:
    B, L, D, E = 2, 4, 32, 4  # Smaller batch for clearer error
    
    print(f"Test parameters: B={B}, L={L}, D={D}, E={E}")
    
    try:
        moe = SparseMoETopK(dim=D, num_experts=E, top_k=2)
        x = torch.randn(B, L, D)
        
        # This would fail with the original implementation
        out, aux = moe(x)
        
        print(f"✅ Successfully processed with shapes:")
        print(f"   Input: {x.shape}")
        print(f"   Output: {out.shape}")
        return True
        
    except Exception as e:
        print(f"❌ Error occurred: {e}")
        return False

def main():
    """Run all validation tests"""
    print("Validating Broadcast/Shape Fix in SparseMoETopK")
    print("=" * 50)
    
    test1_passed = test_broadcast_shape_fix()
    test2_passed = test_original_error_conditions()
    
    print("\n" + "=" * 50)
    if test1_passed and test2_passed:
        print("🎉 All validation tests passed!")
        print("✅ The broadcast/shape bug in SparseMoETopK has been successfully fixed!")
    else:
        print("❌ Some validation tests failed.")
        print("⚠️  The fix may not be complete.")
    
    return test1_passed and test2_passed

if __name__ == "__main__":
    main()