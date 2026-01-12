#!/usr/bin/env python3
"""
Comprehensive test suite for SparseMoETopK component
Combines all individual test cases into a single file
"""

import torch
import unittest
from modell_V5_MAHIA_HyenaMoE import SparseMoETopK, MAHIA_V5

class TestSparseMoETopKComprehensive(unittest.TestCase):
    
    def setUp(self):
        """Set up test fixtures"""
        self.device = torch.device('cpu')
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
    
    # ==================== SHAPE VALIDATION TESTS ====================
    
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
    
    def test_capacity_variations(self):
        """Test SparseMoETopK with different capacity factors"""
        test_cases = [
            (0.1, "Small capacity"),
            (1.0, "Normal capacity"),
            (2.0, "Large capacity"),
        ]
        
        for capacity_factor, description in test_cases:
            with self.subTest(description=description):
                moe = SparseMoETopK(dim=32, num_experts=4, top_k=2, capacity_factor=capacity_factor)
                x = torch.randn(4, 8, 32)
                out, aux = moe(x)
                self.assertEqual(out.shape, (4, 8, 32))
    
    def test_expert_count_variations(self):
        """Test SparseMoETopK with different expert counts"""
        test_cases = [
            (1, "Single expert"),
            (2, "Two experts"),
            (4, "Four experts"),
            (8, "Eight experts"),
            (16, "Sixteen experts"),
        ]
        
        for num_experts, description in test_cases:
            with self.subTest(description=description):
                moe = SparseMoETopK(dim=32, num_experts=num_experts, top_k=min(2, num_experts))
                x = torch.randn(2, 4, 32)
                out, aux = moe(x)
                self.assertEqual(out.shape, (2, 4, 32))
    
    # ==================== FUNCTIONALITY TESTS ====================
    
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
    
    # ==================== BROADCAST/SHAPE FIX TESTS ====================
    
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
    
    def test_original_error_conditions(self):
        """Test conditions that would cause the original error"""
        # The original error was:
        # "The size of tensor a (32) must match the size of tensor b (4) at non-singleton dimension 2"
        # This suggests:
        # - tensor a had size 32 (likely D dimension)
        # - tensor b had size 4 (likely E dimension)
        # - The error occurred at dimension 2
        
        # Let's create a scenario that would match this:
        B, L, D, E = 2, 4, 32, 4  # Smaller batch for clearer error
        
        moe = SparseMoETopK(dim=D, num_experts=E, top_k=2)
        x = torch.randn(B, L, D)
        
        # This would fail with the original implementation
        out, aux = moe(x)
        
        self.assertEqual(out.shape, (B, L, D))
    
    # ==================== REAL TOKEN MOE TESTS ====================
    
    def test_real_token_moe(self):
        """Test that MAHIA_V5 now uses real token-level MoE"""
        # Create model
        model = MAHIA_V5(vocab_size=1000, text_seq_len=32, tab_dim=20, embed_dim=32, fused_dim=64)
        
        # Create test inputs
        batch_size = 4
        seq_len = 16
        text_tokens = torch.randint(0, 1000, (batch_size, seq_len))  # Real token sequences
        tab_feats = torch.randn(batch_size, 20)
        
        # Forward pass
        out, aux = model(text_tokens, tab_feats)
        
        # Verify that the output shape is correct for classification
        self.assertEqual(out.shape, (batch_size, 2))
        
        # Test backward pass
        loss = out.sum()
        loss.backward()
    
    def test_sequence_processing(self):
        """Test that the model properly processes sequences of different lengths"""
        model = MAHIA_V5(vocab_size=1000, text_seq_len=32, tab_dim=20, embed_dim=32, fused_dim=64)
        
        # Test with different sequence lengths
        test_cases = [
            {"seq_len": 8, "description": "Short sequence"},
            {"seq_len": 16, "description": "Medium sequence"},
            {"seq_len": 32, "description": "Long sequence"}
        ]
        
        for case in test_cases:
            seq_len = case["seq_len"]
            description = case["description"]
            
            with self.subTest(description=description):
                text_tokens = torch.randint(0, 1000, (2, seq_len))
                tab_feats = torch.randn(2, 20)
                
                out, aux = model(text_tokens, tab_feats)
                
                self.assertEqual(out.shape, (2, 2))
    
    # ==================== DETERMINISTIC BEHAVIOR TESTS ====================
    
    def test_deterministic_behavior(self):
        """Test that models behave deterministically with fixed seeds"""
        def create_and_test_model(seed):
            torch.manual_seed(seed)
            model = SparseMoETopK(dim=32, num_experts=4, top_k=2)
            x = torch.randn(2, 8, 32)
            out, _ = model(x)
            return out.clone()
        
        # Test with same seed
        out1 = create_and_test_model(42)
        out2 = create_and_test_model(42)
        
        # Should be identical
        diff = (out1 - out2).abs().max().item()
        self.assertLess(diff, 1e-6)

if __name__ == '__main__':
    unittest.main()