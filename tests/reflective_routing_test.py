#!/usr/bin/env python3
"""
Test script for Reflective Router Head implementation
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch

def test_reflective_router_head():
    """Test reflective router head implementation"""
    print("Testing Reflective Router Head...")
    
    from modell_V5_MAHIA_HyenaMoE import ReflectiveRouterHead
    
    # Create reflective router
    router = ReflectiveRouterHead(dim=64, num_experts=8)
    
    # Test forward pass
    x = torch.randn(4, 16, 64)
    confidence, uncertainty = router(x)
    
    print(f"  Input shape: {x.shape}")
    print(f"  Confidence shape: {confidence.shape}")
    print(f"  Uncertainty shape: {uncertainty.shape}")
    print(f"  Confidence range: [{confidence.min().item():.3f}, {confidence.max().item():.3f}]")
    print(f"  Uncertainty range: [{uncertainty.min().item():.3f}, {uncertainty.max().item():.3f}]")
    
    # Check shapes
    assert confidence.shape == (4, 16, 8), f"Confidence shape should be (4, 16, 8), got {confidence.shape}"
    assert uncertainty.shape == (4, 16, 8), f"Uncertainty shape should be (4, 16, 8), got {uncertainty.shape}"
    
    print("  ✅ Reflective Router Head working correctly")

def test_reflective_moe():
    """Test SparseMoETopK with reflective routing"""
    print("\nTesting SparseMoETopK with Reflective Routing...")
    
    from modell_V5_MAHIA_HyenaMoE import SparseMoETopK_Reflective
    
    # Create MoE with reflective routing
    moe = SparseMoETopK_Reflective(dim=64, num_experts=8, top_k=2)
    moe.enable_reflective_routing()
    
    # Test forward pass
    x = torch.randn(4, 16, 64)
    out, aux_loss = moe(x, return_aux=True)
    
    print(f"  Input shape: {x.shape}")
    print(f"  Output shape: {out.shape}")
    print(f"  Aux loss: {aux_loss.item():.6f}")
    
    # Check shapes
    assert out.shape == x.shape, f"Output shape should match input shape, got {out.shape}"
    assert aux_loss is not None, "Auxiliary loss should be computed"
    
    print("  ✅ SparseMoETopK with Reflective Routing working correctly")

def test_training_with_reflective_routing():
    """Test training with reflective routing enabled"""
    print("\nTesting Training with Reflective Routing...")
    
    from modell_V5_MAHIA_HyenaMoE import SparseMoETopK_Reflective
    
    # Create MoE with reflective routing
    moe = SparseMoETopK_Reflective(dim=64, num_experts=8, top_k=2)
    moe.enable_reflective_routing()
    
    # Create optimizer
    optimizer = torch.optim.AdamW(moe.parameters(), lr=1e-3)
    
    # Training loop
    for i in range(10):
        x = torch.randn(4, 16, 64)
        target = torch.randn(4, 16, 64)
        
        # Forward pass
        out, aux_loss = moe(x, return_aux=True)
        
        # Loss
        recon_loss = torch.mean((out - target)**2)
        total_loss = recon_loss
        
        if aux_loss is not None:
            total_loss = total_loss + aux_loss * 0.01
        
        # Backward pass
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        
        if (i + 1) % 5 == 0:
            print(f"    Step {i+1}: Loss = {total_loss.item():.6f}")
    
    print("  ✅ Training with Reflective Routing successful")

def main():
    """Run all reflective routing tests"""
    print("MAHIA-V5 Reflective Routing Tests")
    print("=" * 35)
    
    try:
        test_reflective_router_head()
        test_reflective_moe()
        test_training_with_reflective_routing()
        
        print("\n" + "=" * 35)
        print("🎉 All reflective routing tests passed!")
        print("🚀 MAHIA-V5 now supports confidence-based routing!")
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

if __name__ == "__main__":
    main()