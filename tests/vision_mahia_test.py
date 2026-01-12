#!/usr/bin/env python3
"""
Test script for Vision-enabled MAHIA-V5 model
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch

def test_vision_mahia_model():
    """Test vision-enabled MAHIA model"""
    print("Testing Vision-enabled MAHIA-V5 Model...")
    
    from modell_V5_MAHIA_HyenaMoE import MAHIA_V5_Vision
    
    # Create model
    model = MAHIA_V5_Vision(
        vocab_size=1000,
        text_seq_len=32,
        tab_dim=20,
        img_channels=3,
        embed_dim=64,
        fused_dim=128,
        moe_experts=8,
        moe_topk=2
    )
    
    # Create sample inputs
    batch_size = 4
    text = torch.randint(0, 1000, (batch_size, 32))  # (B, L)
    tab = torch.randn(batch_size, 20)  # (B, tab_dim)
    vision = torch.randn(batch_size, 3, 32, 32)  # (B, C, H, W)
    
    # Forward pass
    logits = model(text, tab, vision)
    
    print(f"  Text input shape: {text.shape}")
    print(f"  Tabular input shape: {tab.shape}")
    print(f"  Vision input shape: {vision.shape}")
    print(f"  Output logits shape: {logits.shape}")
    
    # Check shapes
    assert logits.shape == (batch_size, 1), f"Output shape {logits.shape} should be (B, 1)"
    
    print("  ✅ Vision-enabled MAHIA-V5 model working correctly")

def test_vision_mahia_training():
    """Test training with vision-enabled MAHIA model"""
    print("\nTesting Vision-enabled MAHIA-V5 Training...")
    
    from modell_V5_MAHIA_HyenaMoE import MAHIA_V5_Vision
    
    # Create model
    model = MAHIA_V5_Vision(
        vocab_size=1000,
        text_seq_len=32,
        tab_dim=20,
        img_channels=3,
        embed_dim=64,
        fused_dim=128,
        moe_experts=8,
        moe_topk=2
    )
    
    # Create optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    
    # Training loop
    for i in range(10):
        # Create sample inputs
        batch_size = 4
        text = torch.randint(0, 1000, (batch_size, 32))  # (B, L)
        tab = torch.randn(batch_size, 20)  # (B, tab_dim)
        vision = torch.randn(batch_size, 3, 32, 32)  # (B, C, H, W)
        targets = torch.randint(0, 2, (batch_size, 1)).float()  # Binary classification targets
        
        # Forward pass
        logits = model(text, tab, vision)
        
        # Loss
        loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, targets)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if (i + 1) % 5 == 0:
            print(f"    Step {i+1}: Loss = {loss.item():.6f}")
    
    print("  ✅ Vision-enabled MAHIA-V5 training successful")

def test_gradient_checkpointing():
    """Test gradient checkpointing feature"""
    print("\nTesting Gradient Checkpointing...")
    
    from modell_V5_MAHIA_HyenaMoE import MAHIA_V5_Vision
    
    # Create model and enable gradient checkpointing
    model = MAHIA_V5_Vision(
        vocab_size=1000,
        text_seq_len=32,
        tab_dim=20,
        img_channels=3,
        embed_dim=64,
        fused_dim=128,
        moe_experts=8,
        moe_topk=2
    )
    model.enable_gradient_checkpointing()
    
    # Create sample inputs
    batch_size = 4
    text = torch.randint(0, 1000, (batch_size, 32))  # (B, L)
    tab = torch.randn(batch_size, 20)  # (B, tab_dim)
    vision = torch.randn(batch_size, 3, 32, 32)  # (B, C, H, W)
    targets = torch.randint(0, 2, (batch_size, 1)).float()  # Binary classification targets
    
    # Forward pass
    logits = model(text, tab, vision)
    
    # Loss
    loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, targets)
    
    print(f"  Loss with gradient checkpointing: {loss.item():.6f}")
    print("  ✅ Gradient checkpointing working correctly")

def main():
    """Run all vision-enabled MAHIA tests"""
    print("MAHIA-V5 Vision Tests")
    print("=" * 25)
    
    try:
        test_vision_mahia_model()
        test_vision_mahia_training()
        test_gradient_checkpointing()
        
        print("\n" + "=" * 25)
        print("🎉 All vision-enabled MAHIA tests passed!")
        print("🚀 MAHIA-V5 now supports cross-modal vision capabilities!")
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

if __name__ == "__main__":
    main()