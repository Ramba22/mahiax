#!/usr/bin/env python3
"""
Shape validation tests for SparseMoETopK with different B, L, E, top_k values
"""

import torch
import unittest
from modell_V5_MAHIA_HyenaMoE import SparseMoETopK

class TestSparseMoETopKShapes(unittest.TestCase):
    
    def test_standard_shapes(self):
        """Test SparseMoETopK with standard parameter combinations"""
        test_cases = [
            # (B, L, D, E, top_k, description)
            (2, 8, 32, 4, 2, "Small batch, medium sequence"),
            (4, 16, 64, 8, 3, "Medium batch, longer sequence"),
            (8, 32, 128, 16, 4, "Large batch, long sequence"),
            (1, 4, 16, 2, 1, "Single batch item"),
        ]
        
        for B, L, D, E, top_k, description in test_cases:
            with self.subTest(description=description):
                moe = SparseMoETopK(dim=D, num_experts=E, top_k=top_k)
                x = torch.randn(B, L, D)
                out, aux = moe(x, return_aux=True)
                
                # Check shapes
                self.assertEqual(out.shape, (B, L, D), 
                               f"Output shape mismatch for {description}")
                self.assertEqual(aux.dim(), 0, 
                               f"Auxiliary loss should be scalar for {description}")
    
    def test_edge_cases(self):
        """Test SparseMoETopK with edge case parameters"""
        # Test with top_k > E (should be clamped)
        moe = SparseMoETopK(dim=32, num_experts=2, top_k=4)
        x = torch.randn(2, 4, 32)
        out, aux = moe(x)
        self.assertEqual(out.shape, (2, 4, 32))
        
        # Test with top_k = 1 (minimum valid value)
        moe = SparseMoETopK(dim=32, num_experts=4, top_k=1)
        x = torch.randn(2, 4, 32)
        out, aux = moe(x)
        self.assertEqual(out.shape, (2, 4, 32))
        
        # Test with top_k = E (maximum valid value)
        moe = SparseMoETopK(dim=32, num_experts=4, top_k=4)
        x = torch.randn(2, 4, 32)
        out, aux = moe(x)
        self.assertEqual(out.shape, (2, 4, 32))
    
    def test_small_capacity(self):
        """Test SparseMoETopK with small capacity factor"""
        moe = SparseMoETopK(dim=32, num_experts=4, top_k=2, capacity_factor=0.1)
        x = torch.randn(8, 16, 32)  # Large batch to trigger capacity limiting
        out, aux = moe(x)
        self.assertEqual(out.shape, (8, 16, 32))
    
    def test_large_expert_count(self):
        """Test SparseMoETopK with large number of experts"""
        moe = SparseMoETopK(dim=32, num_experts=32, top_k=4)
        x = torch.randn(2, 8, 32)
        out, aux = moe(x)
        self.assertEqual(out.shape, (2, 8, 32))
    
    def test_single_expert(self):
        """Test SparseMoETopK with single expert"""
        moe = SparseMoETopK(dim=32, num_experts=1, top_k=1)
        x = torch.randn(2, 4, 32)
        out, aux = moe(x)
        self.assertEqual(out.shape, (2, 4, 32))
    
    def test_backward_compatibility(self):
        """Test that backward pass works for all configurations"""
        test_cases = [
            (2, 4, 32, 4, 2),
            (4, 8, 64, 8, 3),
            (1, 2, 16, 2, 1),
        ]
        
        for B, L, D, E, top_k in test_cases:
            with self.subTest(B=B, L=L, D=D, E=E, top_k=top_k):
                moe = SparseMoETopK(dim=D, num_experts=E, top_k=top_k)
                x = torch.randn(B, L, D, requires_grad=True)
                out, _ = moe(x)
                loss = out.sum()
                loss.backward()
                self.assertIsNotNone(x.grad)

if __name__ == '__main__':
    unittest.main()