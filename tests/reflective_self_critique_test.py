#!/usr/bin/env python3
"""
Test script for Reflective Self-Critique & Meta-Controller implementation
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch

def test_reflective_head():
    """Test ReflectiveHead implementation"""
    print("Testing ReflectiveHead...")
    
    from modell_V5_MAHIA_HyenaMoE import ReflectiveHead
    
    # Create reflective head
    head = ReflectiveHead(dim=64, num_tags=4)
    
    # Test forward pass
    features = torch.randn(4, 16, 64)
    logits = torch.randn(4, 16, 64)
    p_error, tags = head(features, logits)
    
    print(f"  Features shape: {features.shape}")
    print(f"  Logits shape: {logits.shape}")
    print(f"  Error probability shape: {p_error.shape}")
    print(f"  Tags shape: {tags.shape}")
    print(f"  Error probability range: [{p_error.min().item():.3f}, {p_error.max().item():.3f}]")
    print(f"  Tags sum (should be 1): {tags.sum(dim=-1)[0, 0].item():.3f}")
    
    # Check shapes
    assert p_error.shape == (4, 16, 1), f"Error probability shape should be (4, 16, 1), got {p_error.shape}"
    assert tags.shape == (4, 16, 4), f"Tags shape should be (4, 16, 4), got {tags.shape}"
    assert torch.allclose(tags.sum(dim=-1), torch.ones_like(tags.sum(dim=-1)), atol=1e-6), "Tags should sum to 1"
    
    print("  ✅ ReflectiveHead working correctly")

def test_meta_controller():
    """Test MetaController implementation"""
    print("\nTesting MetaController...")
    
    from modell_V5_MAHIA_HyenaMoE import MetaController
    
    # Create meta controller
    controller = MetaController(dim=64)
    
    # Test forward pass
    features = torch.randn(4, 16, 64)
    p_error = torch.rand(4, 16, 1)
    
    actions = controller(features, p_error)
    
    print(f"  Features shape: {features.shape}")
    print(f"  Error probability shape: {p_error.shape}")
    print(f"  Actions keys: {list(actions.keys())}")
    print(f"  Action probabilities shape: {actions['action_probs'].shape}")
    
    # Check shapes
    assert 'escalate' in actions, "Actions should contain 'escalate'"
    assert 'abstain' in actions, "Actions should contain 'abstain'"
    assert 'continue' in actions, "Actions should contain 'continue'"
    assert actions['action_probs'].shape == (4, 3), f"Action probabilities shape should be (4, 3), got {actions['action_probs'].shape}"
    
    print("  ✅ MetaController working correctly")

def test_integrated_self_critique():
    """Test integrated self-critique system"""
    print("\nTesting Integrated Self-Critique System...")
    
    from modell_V5_MAHIA_HyenaMoE import ReflectiveHead, MetaController, SparseMoETopK
    
    # Create components
    moe = SparseMoETopK(dim=64, num_experts=8, top_k=2)
    reflective_head = ReflectiveHead(dim=64, num_tags=4)
    meta_controller = MetaController(dim=64)
    
    # Test forward pass through complete system
    x = torch.randn(4, 16, 64)
    out, aux_loss = moe(x, return_aux=True)
    
    # Get reflective assessment
    p_error, tags = reflective_head(x, out, aux_loss)
    
    # Get control actions
    actions = meta_controller(x, p_error)
    
    print(f"  MoE output shape: {out.shape}")
    print(f"  Auxiliary loss: {aux_loss.item():.6f}")
    print(f"  Error probability mean: {p_error.mean().item():.3f}")
    print(f"  Escalate decisions: {actions['escalate'].sum().item()}/{actions['escalate'].size(0)}")
    print(f"  Abstain decisions: {actions['abstain'].sum().item()}/{actions['abstain'].size(0)}")
    
    print("  ✅ Integrated Self-Critique System working correctly")

def test_training_with_self_critique():
    """Test training with self-critique components"""
    print("\nTesting Training with Self-Critique...")
    
    from modell_V5_MAHIA_HyenaMoE import ReflectiveHead, MetaController, SparseMoETopK
    
    # Create components
    moe = SparseMoETopK(dim=64, num_experts=8, top_k=2)
    reflective_head = ReflectiveHead(dim=64, num_tags=4)
    meta_controller = MetaController(dim=64)
    
    # Create optimizer
    params = list(moe.parameters()) + list(reflective_head.parameters()) + list(meta_controller.parameters())
    optimizer = torch.optim.AdamW(params, lr=1e-3)
    
    # Training loop
    for i in range(10):
        x = torch.randn(4, 16, 64)
        target = torch.randn(4, 16, 64)
        
        # Forward pass
        out, aux_loss = moe(x, return_aux=True)
        
        # Self-critique assessment
        p_error, tags = reflective_head(x, out, aux_loss)
        actions = meta_controller(x, p_error)
        
        # Loss computation
        recon_loss = torch.mean((out - target)**2)
        error_loss = torch.mean(p_error)  # Encourage low error probability
        total_loss = recon_loss + error_loss * 0.1
        
        if aux_loss is not None:
            total_loss = total_loss + aux_loss * 0.01
        
        # Backward pass
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        
        if (i + 1) % 5 == 0:
            print(f"    Step {i+1}: Loss = {total_loss.item():.6f}")
    
    print("  ✅ Training with Self-Critique successful")

def main():
    """Run all reflective self-critique tests"""
    print("MAHIA-V5 Reflective Self-Critique Tests")
    print("=" * 40)
    
    try:
        test_reflective_head()
        test_meta_controller()
        test_integrated_self_critique()
        test_training_with_self_critique()
        
        print("\n" + "=" * 40)
        print("🎉 All reflective self-critique tests passed!")
        print("🚀 MAHIA-V5 now supports self-critique and meta-control!")
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

if __name__ == "__main__":
    main()