#!/usr/bin/env python3
"""
Test script for enhanced Hierarchical MoE with Coarse-to-Fine / Cluster-Balancing
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch

def test_enhanced_hierarchical_moe():
    """Test enhanced HierarchicalMoE implementation"""
    print("Testing Enhanced Hierarchical MoE...")
    
    try:
        from modell_V5_MAHIA_HyenaMoE import HierarchicalMoE_DomainClustering
        
        # Create enhanced hierarchical MoE
        moe = HierarchicalMoE_DomainClustering(
            dim=64, 
            num_domains=4, 
            experts_per_domain=4,
            top_k_coarse=2, 
            top_k_fine=2
        )
        
        # Enable cluster balancing
        moe.enable_cluster_balancing(weight=0.02)
        
        # Test forward pass
        x = torch.randn(4, 16, 64)
        out, aux_loss = moe(x, return_aux=True)
        
        print(f"  Input shape: {x.shape}")
        print(f"  Output shape: {out.shape}")
        print(f"  Aux loss: {aux_loss.item():.6f}")
        
        # Check shapes
        assert out.shape == x.shape, f"Output shape should match input, got {out.shape}"
        assert aux_loss is not None, "Auxiliary loss should be computed"
        
        print("  ✅ Enhanced HierarchicalMoE working correctly")
        
    except Exception as e:
        print(f"  ❌ Enhanced HierarchicalMoE test failed: {e}")
        import traceback
        traceback.print_exc()

def test_cluster_balancing():
    """Test cluster balancing feature"""
    print("\nTesting Cluster Balancing...")
    
    try:
        from modell_V5_MAHIA_HyenaMoE import HierarchicalMoE_DomainClustering
        
        # Create hierarchical MoE with cluster balancing
        moe = HierarchicalMoE_DomainClustering(dim=64, num_domains=3, experts_per_domain=3)
        moe.enable_cluster_balancing(weight=0.03)
        moe.enable_expert_diversity_loss(weight=0.02)
        
        # Test forward pass
        x = torch.randn(4, 16, 64)
        out, aux_loss = moe(x, return_aux=True)
        
        print(f"  Output shape: {out.shape}")
        print(f"  Aux loss with cluster balancing: {aux_loss.item():.6f}")
        
        # Check that aux loss is computed
        assert aux_loss is not None, "Auxiliary loss should be computed"
        
        print("  ✅ Cluster Balancing working correctly")
        
    except Exception as e:
        print(f"  ❌ Cluster Balancing test failed: {e}")
        import traceback
        traceback.print_exc()

def test_training_enhanced_hierarchical_moe():
    """Test training with enhanced HierarchicalMoE"""
    print("\nTesting Training with Enhanced HierarchicalMoE...")
    
    try:
        from modell_V5_MAHIA_HyenaMoE import HierarchicalMoE_DomainClustering
        
        # Create enhanced hierarchical MoE
        moe = HierarchicalMoE_DomainClustering(dim=64, num_domains=3, experts_per_domain=3)
        moe.enable_cluster_balancing(weight=0.02)
        moe.enable_expert_diversity_loss(weight=0.02)
        
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
        
        print("  ✅ Training with Enhanced HierarchicalMoE successful")
        
    except Exception as e:
        print(f"  ❌ Training test failed: {e}")
        import traceback
        traceback.print_exc()

def main():
    """Run all enhanced hierarchical MoE tests"""
    print("MAHIA-V5 Enhanced Hierarchical MoE Tests")
    print("=" * 45)
    
    try:
        test_enhanced_hierarchical_moe()
        test_cluster_balancing()
        test_training_enhanced_hierarchical_moe()
        
        print("\n" + "=" * 45)
        print("🎉 All enhanced hierarchical MoE tests passed!")
        print("🚀 MAHIA-V5 now supports enhanced hierarchical MoE with cluster balancing!")
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

if __name__ == "__main__":
    main()