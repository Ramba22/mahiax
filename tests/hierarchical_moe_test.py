#!/usr/bin/env python3
"""
Test script for Hierarchical MoE implementation
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch

def test_hierarchical_moe():
    """Test hierarchical MoE implementation"""
    print("Testing Hierarchical MoE...")
    
    from modell_V5_MAHIA_HyenaMoE import HierarchicalMoE
    
    # Create hierarchical MoE layer
    moe = HierarchicalMoE(dim=64, num_domains=4, experts_per_domain=4, 
                         top_k_coarse=2, top_k_fine=2)
    
    # Test forward pass
    x = torch.randn(4, 16, 64)
    out, aux_loss = moe(x, return_aux=True)
    
    print(f"  Input shape: {x.shape}")
    print(f"  Output shape: {out.shape}")
    print(f"  Aux loss: {aux_loss.item():.6f}")
    
    # Check shapes
    assert out.shape == x.shape, f"Output shape {out.shape} should match input shape {x.shape}"
    assert aux_loss is not None, "Auxiliary loss should be computed"
    
    print("  ✅ Hierarchical MoE working correctly")

def test_hierarchical_moe_training():
    """Test training with hierarchical MoE"""
    print("\nTesting Hierarchical MoE Training...")
    
    from modell_V5_MAHIA_HyenaMoE import HierarchicalMoE
    
    # Create hierarchical MoE layer
    moe = HierarchicalMoE(dim=64, num_domains=3, experts_per_domain=3, 
                         top_k_coarse=2, top_k_fine=2)
    
    # Create optimizer
    optimizer = torch.optim.AdamW(moe.parameters(), lr=1e-3)
    
    # Training loop
    for i in range(10):
        x = torch.randn(4, 16, 64)
        target = torch.randn(4, 16, 64)
        
        # Forward pass
        out, aux_loss = moe(x, return_aux=True)
        
        # Loss (reconstruction + aux)
        recon_loss = torch.mean((out - target)**2)
        total_loss = recon_loss
        
        if aux_loss is not None:
            total_loss = total_loss + aux_loss * 0.01  # Weight aux loss
        
        # Backward pass
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        
        if (i + 1) % 5 == 0:
            print(f"    Step {i+1}: Loss = {total_loss.item():.6f}")
    
    print("  ✅ Hierarchical MoE training successful")

def test_expert_diversity_loss():
    """Test expert diversity loss feature"""
    print("\nTesting Expert Diversity Loss...")
    
    from modell_V5_MAHIA_HyenaMoE import HierarchicalMoE
    
    # Create hierarchical MoE layer with diversity loss
    moe = HierarchicalMoE(dim=64, num_domains=4, experts_per_domain=4)
    moe.enable_expert_diversity_loss(weight=0.02)
    
    # Test forward pass
    x = torch.randn(4, 16, 64)
    out, aux_loss = moe(x, return_aux=True)
    
    print(f"  Aux loss with diversity: {aux_loss.item():.6f}")
    assert aux_loss is not None, "Auxiliary loss should be computed"
    
    print("  ✅ Expert diversity loss working correctly")

def main():
    """Run all hierarchical MoE tests"""
    print("MAHIA-V5 Hierarchical MoE Tests")
    print("=" * 35)
    
    try:
        test_hierarchical_moe()
        test_hierarchical_moe_training()
        test_expert_diversity_loss()
        
        print("\n" + "=" * 35)
        print("🎉 All hierarchical MoE tests passed!")
        print("🚀 MAHIA-V5 now supports hierarchical routing!")
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

if __name__ == "__main__":
    main()