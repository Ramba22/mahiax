#!/usr/bin/env python3
"""
Test script to validate the SparseMoETopK fix
"""

import torch
from modell_V5_MAHIA_HyenaMoE import SparseMoETopK

def test_sparse_moe_shapes():
    """Test SparseMoETopK with different batch sizes and shapes"""
    print("Testing SparseMoETopK shape compatibility...")
    
    # Test with different configurations
    test_cases = [
        {"B": 2, "L": 4, "D": 32, "E": 4, "top_k": 2},
        {"B": 4, "L": 8, "D": 64, "E": 8, "top_k": 3},
        {"B": 1, "L": 2, "D": 16, "E": 2, "top_k": 1},
    ]
    
    for i, case in enumerate(test_cases):
        print(f"\nTest case {i+1}: B={case['B']}, L={case['L']}, D={case['D']}, E={case['E']}, top_k={case['top_k']}")
        
        try:
            # Create MoE layer
            moe = SparseMoETopK(dim=case["D"], num_experts=case["E"], top_k=case["top_k"])
            
            # Create test input
            x = torch.randn(case["B"], case["L"], case["D"])
            
            # Forward pass
            out, aux_loss = moe(x, return_aux=True)
            
            # Check shapes
            assert out.shape == x.shape, f"Output shape mismatch: {out.shape} != {x.shape}"
            
            print(f"  ✓ Input shape: {x.shape}")
            print(f"  ✓ Output shape: {out.shape}")
            
            if aux_loss is not None:
                print(f"  ✓ Aux loss: {aux_loss.item():.6f}")
            
            # Test backward pass
            loss = out.sum()
            loss.backward()
            print(f"  ✓ Backward pass successful")
            
        except Exception as e:
            print(f"  ❌ Test failed with error: {e}")
            return False
    
    print("\n✅ All SparseMoETopK shape tests passed!")
    return True

def test_edge_cases():
    """Test edge cases for SparseMoETopK"""
    print("\nTesting SparseMoETopK edge cases...")
    
    # Test with top_k equal to num_experts (maximum valid value)
    print("\nEdge case 1: top_k == num_experts")
    try:
        moe = SparseMoETopK(dim=32, num_experts=4, top_k=4)  # top_k == num_experts
        x = torch.randn(2, 4, 32)
        out, aux = moe(x)
        print(f"  ✓ Handled top_k == num_experts: {out.shape}")
    except Exception as e:
        print(f"  ❌ Failed with error: {e}")
        return False
    
    # Test with capacity limiting
    print("\nEdge case 2: Capacity limiting")
    try:
        moe = SparseMoETopK(dim=32, num_experts=4, top_k=2, capacity_factor=0.1)  # Very low capacity
        x = torch.randn(8, 16, 32)  # Large batch to trigger capacity limiting
        out, aux = moe(x)
        print(f"  ✓ Handled capacity limiting: {out.shape}")
    except Exception as e:
        print(f"  ❌ Failed with error: {e}")
        return False
    
    print("\n✅ All edge case tests passed!")
    return True

def main():
    """Run all tests"""
    print("SparseMoETopK Fix Validation")
    print("=" * 40)
    
    success = test_sparse_moe_shapes() and test_edge_cases()
    
    print("\n" + "=" * 40)
    if success:
        print("🎉 All tests passed! The SparseMoETopK fix is working correctly.")
    else:
        print("❌ Some tests failed. Please check the implementation.")
    
    return success

if __name__ == "__main__":
    main()