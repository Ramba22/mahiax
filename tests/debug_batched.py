#!/usr/bin/env python3
"""
Debug script to understand batched execution differences
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch

def debug_batched_execution():
    """Debug batched expert execution feature"""
    print("Debugging Batched Expert Execution...")
    
    from modell_V5_MAHIA_HyenaMoE import SparseMoETopK
    
    # Create MoE layer
    moe = SparseMoETopK(dim=64, num_experts=4, top_k=2)
    
    # Create test input
    torch.manual_seed(42)
    x = torch.randn(2, 8, 64)
    
    # Test without batched execution
    moe.use_batched_execution = False
    out1, aux1 = moe(x, return_aux=True)
    print(f"Without batched execution - Aux loss: {aux1.item():.6f}")
    
    # Test with batched execution
    moe.use_batched_execution = True
    out2, aux2 = moe(x, return_aux=True)
    print(f"With batched execution - Aux loss: {aux2.item():.6f}")
    
    # Check that outputs are similar
    diff = torch.mean(torch.abs(out1 - out2)).item()
    max_diff = torch.max(torch.abs(out1 - out2)).item()
    print(f"Mean output difference: {diff:.8f}")
    print(f"Max output difference: {max_diff:.8f}")
    
    # Check expert inputs
    print("\nChecking intermediate values...")
    
    # Get the gate logits and probs
    logits = moe.gate(x)
    probs = torch.softmax(logits, dim=-1)
    print(f"Probs shape: {probs.shape}")
    print(f"Probs sum (should be 1 per token): {probs.sum(dim=-1)[0, :4]}")
    
    # Check top-k selection
    topk_vals, topk_idx = torch.topk(probs, k=moe.top_k, dim=-1)
    print(f"Top-k vals shape: {topk_vals.shape}")
    print(f"Top-k indices shape: {topk_idx.shape}")
    print(f"Top-k indices (first token): {topk_idx[0, 0, :]}")
    
    # Check dispatch tensor
    weight_norm = topk_vals.sum(dim=-1, keepdim=True).clamp_min(1e-9)
    normalized_vals = topk_vals / weight_norm
    dispatch = torch.zeros((2, 8, moe.num_experts), device=x.device, dtype=x.dtype)
    dispatch.scatter_(-1, topk_idx, normalized_vals)
    print(f"Dispatch tensor shape: {dispatch.shape}")
    print(f"Dispatch sum per token (should be 1): {dispatch.sum(dim=-1)[0, :4]}")
    
    # Check expert inputs computation
    expert_counts = dispatch.sum(dim=1).clamp(min=1.0)
    expert_inputs = torch.einsum('bld,ble->bed', x, dispatch)
    expert_inputs_normalized = expert_inputs / expert_counts.unsqueeze(-1)
    print(f"Expert inputs shape: {expert_inputs.shape}")
    print(f"Expert counts: {expert_counts[0]}")
    print(f"Expert inputs (first expert): {expert_inputs_normalized[0, 0, :5]}")

if __name__ == "__main__":
    debug_batched_execution()