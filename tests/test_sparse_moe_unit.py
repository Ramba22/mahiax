#!/usr/bin/env python3
"""
Unit tests for SparseMoETopK component
"""

import torch
import unittest
from modell_V5_MAHIA_HyenaMoE import SparseMoETopK

class TestSparseMoETopK(unittest.TestCase):
    
    def setUp(self):
        """Set up test fixtures"""
        self.device = torch.device('cpu')  # Use CPU for testing
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
    
    def test_forward_shape_consistency(self):
        """Test that output shape matches input shape"""
        # Test with various configurations
        test_cases = [
            {"B": 2, "L": 4, "D": 32, "E": 4, "top_k": 2},
            {"B": 4, "L": 8, "D": 64, "E": 8, "top_k": 3},
            {"B": 1, "L": 2, "D": 16, "E": 2, "top_k": 1},
        ]
        
        for case in test_cases:
            with self.subTest(case=case):
                moe = SparseMoETopK(dim=case["D"], num_experts=case["E"], top_k=case["top_k"])
                x = torch.randn(case["B"], case["L"], case["D"])
                out, _ = moe(x)
                self.assertEqual(out.shape, x.shape)
    
    def test_backward_pass(self):
        """Test that backward pass works without errors"""
        moe = SparseMoETopK(dim=32, num_experts=4, top_k=2)
        x = torch.randn(2, 4, 32, requires_grad=True)
        
        out, _ = moe(x)
        loss = out.sum()
        
        # This should not raise any errors
        loss.backward()
        
        # Check that gradients were computed
        self.assertIsNotNone(x.grad)
    
    def test_aux_loss_computation(self):
        """Test that auxiliary loss is computed correctly"""
        moe = SparseMoETopK(dim=32, num_experts=4, top_k=2)
        x = torch.randn(2, 4, 32)
        
        _, aux_loss = moe(x, return_aux=True)
        
        # Auxiliary loss should be a scalar tensor
        self.assertIsInstance(aux_loss, torch.Tensor)
        self.assertEqual(aux_loss.dim(), 0)
    
    def test_no_aux_loss_when_disabled(self):
        """Test that auxiliary loss is None when disabled"""
        moe = SparseMoETopK(dim=32, num_experts=4, top_k=2)
        x = torch.randn(2, 4, 32)
        
        _, aux_loss = moe(x, return_aux=False)
        
        # Auxiliary loss should be None when disabled
        self.assertIsNone(aux_loss)
    
    def test_topk_clamping(self):
        """Test that top_k is properly clamped to num_experts"""
        # Create MoE with top_k > num_experts
        moe = SparseMoETopK(dim=32, num_experts=2, top_k=4)  # top_k > num_experts
        
        # The effective top_k should be clamped to num_experts
        self.assertEqual(moe.top_k, 2)
        
        # Should work without errors
        x = torch.randn(2, 4, 32)
        out, _ = moe(x)
        self.assertEqual(out.shape, x.shape)
    
    def test_broadcast_shape_fix(self):
        """Test the specific broadcast/shape fix"""
        # This test reproduces the exact issue that was fixed:
        # expert_counts had shape mismatch with expert_inputs during division
        B, L, D, E = 4, 8, 32, 4
        moe = SparseMoETopK(dim=D, num_experts=E, top_k=2)
        x = torch.randn(B, L, D)
        
        # This should not raise a shape mismatch error
        out, aux = moe(x)
        
        # Validate shapes
        self.assertEqual(out.shape, (B, L, D))
        
        # Backward pass should also work
        loss = out.sum()
        loss.backward()

if __name__ == '__main__':
    unittest.main()