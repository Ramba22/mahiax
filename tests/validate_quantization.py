#!/usr/bin/env python3
"""
Quick validation script for 8-bit quantization numerical accuracy
"""

import torch
from modell_V4_Nvidiaonly import HybridEfficientModel, quantize_model_8bit

def validate_quantization_numerics():
    """Validate numerical differences between FP32 and 8-bit quantized models"""
    print("Validating 8-bit quantization numerical accuracy...")
    
    device = torch.device('cpu')  # Use CPU for testing
    
    # Create a small model for validation
    model_fp32 = HybridEfficientModel(
        vocab_size=1000,
        text_seq_len=32,
        tab_dim=20,
        output_dim=2,
        embed_dim=32,
        tab_hidden_dim=32,
        fused_dim=64
    )
    
    model_fp32.to(device)
    model_fp32.eval()
    
    # Create 8-bit quantized version
    model_8bit = HybridEfficientModel(
        vocab_size=1000,
        text_seq_len=32,
        tab_dim=20,
        output_dim=2,
        embed_dim=32,
        tab_hidden_dim=32,
        fused_dim=64
    )
    
    model_8bit.to(device)
    model_8bit.eval()
    
    # Apply 8-bit quantization
    try:
        model_8bit = quantize_model_8bit(model_8bit)
        print("8-bit quantization applied successfully")
    except Exception as e:
        print(f"8-bit quantization failed: {e}")
        return False
    
    # Test with sample inputs
    with torch.no_grad():
        x_text = torch.randint(0, 1000, (2, 32))
        x_tab = torch.randn(2, 20)
        
        x_text = x_text.to(device)
        x_tab = x_tab.to(device)
        
        # Get outputs from both models
        o_fp32 = model_fp32(x_text, x_tab)
        o_8bit = model_8bit(x_text, x_tab)
        
        # Calculate differences
        diff = (o_fp32 - o_8bit).abs()
        mean_diff = diff.mean()
        max_diff = diff.max()
        
        print(f"Mean absolute difference: {mean_diff.item():.6f}")
        print(f"Max absolute difference: {max_diff.item():.6f}")
        
        # Check if differences are within acceptable range
        if mean_diff.item() < 0.1 and max_diff.item() < 0.5:
            print("✅ Quantization validation PASSED - differences are acceptable")
            return True
        else:
            print("⚠️  Quantization validation FAILED - differences may be too large")
            print("Consider using QAT (Quantization Aware Training) or eval-only quantization")
            return False

if __name__ == "__main__":
    success = validate_quantization_numerics()
    if success:
        print("\nQuantization validation completed successfully!")
    else:
        print("\nQuantization validation completed with warnings!")