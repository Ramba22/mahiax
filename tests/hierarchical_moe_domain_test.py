#!/usr/bin/env python3
"""
Test script for Hierarchical MoE with Domain Clustering implementation
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch

def test_expert_adapter():
    """Test ExpertAdapter implementation"""
    print("Testing ExpertAdapter...")
    
    from modell_V5_MAHIA_HyenaMoE import ExpertAdapter
    
    # Create expert adapter
    adapter = ExpertAdapter(dim=64)
    
    # Test forward pass
    x = torch.randn(4, 16, 64)
    expert_a_out = torch.randn(4, 16, 64)
    expert_b_out = torch.randn(4, 16, 64)
    
    composed = adapter(x, expert_a_out, expert_b_out)
    
    print(f"  Input shape: {x.shape}")
    print(f"  Expert A output shape: {expert_a_out.shape}")
    print(f"  Expert B output shape: {expert_b_out.shape}")
    print(f"  Composed output shape: {composed.shape}")
    
    # Check shapes
    assert composed.shape == x.shape, f"Composed output shape should match input, got {composed.shape}"
    
    print("  ✅ ExpertAdapter working correctly")

def test_hierarchical_moe_domain():
    """Test HierarchicalMoE_DomainClustering implementation"""
    print("\nTesting HierarchicalMoE_DomainClustering...")
    
    from modell_V5_MAHIA_HyenaMoE import HierarchicalMoE_DomainClustering
    
    # Create hierarchical MoE
    moe = HierarchicalMoE_DomainClustering(
        dim=64, 
        num_domains=4, 
        experts_per_domain=4,
        top_k_coarse=2, 
        top_k_fine=2
    )
    
    # Test forward pass
    x = torch.randn(4, 16, 64)
    out, aux_loss = moe(x, return_aux=True)
    
    print(f"  Input shape: {x.shape}")
    print(f"  Output shape: {out.shape}")
    print(f"  Aux loss: {aux_loss.item():.6f}")
    
    # Check shapes
    assert out.shape == x.shape, f"Output shape should match input, got {out.shape}"
    assert aux_loss is not None, "Auxiliary loss should be computed"
    
    print("  ✅ HierarchicalMoE_DomainClustering working correctly")

def test_hierarchical_moe_with_diversity():
    """Test HierarchicalMoE with expert diversity loss"""
    print("\nTesting HierarchicalMoE with Expert Diversity...")
    
    from modell_V5_MAHIA_HyenaMoE import HierarchicalMoE_DomainClustering
    
    # Create hierarchical MoE with diversity loss
    moe = HierarchicalMoE_DomainClustering(dim=64, num_domains=3, experts_per_domain=3)
    moe.enable_expert_diversity_loss(weight=0.02)
    
    # Test forward pass
    x = torch.randn(4, 16, 64)
    out, aux_loss = moe(x, return_aux=True)
    
    print(f"  Output shape: {out.shape}")
    print(f"  Aux loss with diversity: {aux_loss.item():.6f}")
    
    # Check that aux loss is computed
    assert aux_loss is not None, "Auxiliary loss should be computed"
    
    print("  ✅ HierarchicalMoE with Expert Diversity working correctly")

def test_training_hierarchical_moe():
    """Test training with HierarchicalMoE"""
    print("\nTesting Training with HierarchicalMoE...")
    
    from modell_V5_MAHIA_HyenaMoE import HierarchicalMoE_DomainClustering
    
    # Create hierarchical MoE
    moe = HierarchicalMoE_DomainClustering(dim=64, num_domains=3, experts_per_domain=3)
    
    # Create optimizer
    optimizer = torch.optim.AdamW(moe.parameters(), lr=1e-3)
    
    # Training loop
    for i in range(10):
        x = torch.randn(4, 16, 64)
        target = torch.randn(4, 16, 64)
        
        # Forward pass
        out, aux_loss = moe(x, return_aux=True)
        
        # Loss computation
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
    
    print("  ✅ Training with HierarchicalMoE successful")

def main():
    """Run all hierarchical MoE domain clustering tests"""
    print("MAHIA-V5 Hierarchical MoE Domain Clustering Tests")
    print("=" * 50)
    
    try:
        test_expert_adapter()
        test_hierarchical_moe_domain()
        test_hierarchical_moe_with_diversity()
        test_training_hierarchical_moe()
        
        print("\n" + "=" * 50)
        print("🎉 All hierarchical MoE domain clustering tests passed!")
        print("🚀 MAHIA-V5 now supports hierarchical MoE with domain clustering!")
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

if __name__ == "__main__":
    main()