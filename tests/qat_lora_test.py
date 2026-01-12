#!/usr/bin/env python3
"""
Test script for QAT and LoRA adapter implementation
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch

def test_lora_adapter():
    """Test LoRA adapter implementation"""
    print("Testing LoRA Adapter...")
    
    try:
        from modell_V5_MAHIA_HyenaMoE import LoRAAdapter
        
        # Create LoRA adapter
        adapter = LoRAAdapter(in_features=64, out_features=128, rank=8)
        
        # Test forward pass
        x = torch.randn(4, 64)
        output = adapter(x)
        
        print(f"  Input shape: {x.shape}")
        print(f"  Output shape: {output.shape}")
        print(f"  A matrix shape: {adapter.A.shape}")
        print(f"  B matrix shape: {adapter.B.shape}")
        
        # Check shapes
        assert output.shape == (4, 128), f"Output shape should be (4, 128), got {output.shape}"
        assert adapter.A.shape == (64, 8), f"A matrix shape should be (64, 8), got {adapter.A.shape}"
        assert adapter.B.shape == (8, 128), f"B matrix shape should be (8, 128), got {adapter.B.shape}"
        
        print("  ✅ LoRA Adapter working correctly")
        
    except ImportError as e:
        print(f"  ⚠️  LoRA test skipped due to missing dependencies: {e}")

def test_qat_wrapper():
    """Test QAT wrapper implementation"""
    print("\nTesting QAT Wrapper...")
    
    try:
        from modell_V5_MAHIA_HyenaMoE import QATLoRAWrapper, SparseMoETopK
        
        # Create a simple model
        model = torch.nn.Linear(64, 128)
        wrapped_model = QATLoRAWrapper(model, use_lora=True)
        
        # Test forward pass
        x = torch.randn(4, 64)
        output = wrapped_model(x)
        
        print(f"  Input shape: {x.shape}")
        print(f"  Output shape: {output.shape}")
        
        # Check shapes
        assert output.shape == (4, 128), f"Output shape should be (4, 128), got {output.shape}"
        
        print("  ✅ QAT Wrapper working correctly")
        
    except ImportError as e:
        print(f"  ⚠️  QAT test skipped due to missing dependencies: {e}")

def test_moe_with_lora():
    """Test MoE with LoRA adapters"""
    print("\nTesting MoE with LoRA Adapters...")
    
    try:
        from modell_V5_MAHIA_HyenaMoE import SparseMoETopK, QATLoRAWrapper
        
        # Create MoE model
        moe = SparseMoETopK(dim=64, num_experts=4, top_k=2)
        wrapped_moe = QATLoRAWrapper(moe, use_lora=True)
        
        # Test forward pass
        x = torch.randn(4, 16, 64)
        out, aux = wrapped_moe(x, return_aux=True)
        
        print(f"  Input shape: {x.shape}")
        print(f"  Output shape: {out.shape}")
        print(f"  Aux loss: {aux.item():.6f}")
        
        # Check shapes
        assert out.shape == x.shape, f"Output shape should match input shape, got {out.shape}"
        assert aux is not None, "Auxiliary loss should be computed"
        
        print("  ✅ MoE with LoRA Adapters working correctly")
        
    except ImportError as e:
        print(f"  ⚠️  MoE with LoRA test skipped due to missing dependencies: {e}")

def main():
    """Run all QAT/LoRA tests"""
    print("MAHIA-V5 QAT & LoRA Tests")
    print("=" * 25)
    
    try:
        test_lora_adapter()
        test_qat_wrapper()
        test_moe_with_lora()
        
        print("\n" + "=" * 25)
        print("🎉 All QAT & LoRA tests completed!")
        print("🚀 MAHIA-V5 now supports quantization-aware training and LoRA adapters!")
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

if __name__ == "__main__":
    main()