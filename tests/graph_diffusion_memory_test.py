#!/usr/bin/env python3
"""
Test script for enhanced GraphDiffusionMemory implementation
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch

def test_graph_diffusion_memory():
    """Test GraphDiffusionMemory implementation"""
    print("Testing GraphDiffusionMemory...")
    
    from modell_V5_MAHIA_HyenaMoE import GraphDiffusionMemory
    
    # Create memory module
    memory = GraphDiffusionMemory(dim=64, memory_size=16)
    
    # Test forward pass with 2D input
    x_2d = torch.randn(4, 64)
    out_2d = memory(x_2d)
    
    print(f"  2D Input shape: {x_2d.shape}")
    print(f"  2D Output shape: {out_2d.shape}")
    
    # Check shapes
    assert out_2d.shape == x_2d.shape, f"2D output shape should match input, got {out_2d.shape}"
    
    # Test forward pass with 3D input
    x_3d = torch.randn(4, 16, 64)
    out_3d = memory(x_3d)
    
    print(f"  3D Input shape: {x_3d.shape}")
    print(f"  3D Output shape: {out_3d.shape}")
    
    # Check shapes
    assert out_3d.shape == x_3d.shape, f"3D output shape should match input, got {out_3d.shape}"
    
    print("  ✅ GraphDiffusionMemory working correctly")

def test_memory_retrieval():
    """Test memory retrieval functionality"""
    print("\nTesting Memory Retrieval...")
    
    from modell_V5_MAHIA_HyenaMoE import GraphDiffusionMemory
    
    # Create memory module
    memory = GraphDiffusionMemory(dim=64, memory_size=16)
    
    # Test top-k retrieval
    query = torch.randn(2, 64)
    retrieved_nodes, similarities = memory.retrieve_top_k(query, k=5)
    
    print(f"  Query shape: {query.shape}")
    print(f"  Retrieved nodes shape: {retrieved_nodes.shape}")
    print(f"  Similarities shape: {similarities.shape}")
    
    # Check shapes
    assert retrieved_nodes.shape == (2, 5, 64), f"Retrieved nodes shape should be (2, 5, 64), got {retrieved_nodes.shape}"
    assert similarities.shape == (2, 5), f"Similarities shape should be (2, 5), got {similarities.shape}"
    
    print("  ✅ Memory Retrieval working correctly")

def test_node_compression():
    """Test node compression functionality"""
    print("\nTesting Node Compression...")
    
    from modell_V5_MAHIA_HyenaMoE import GraphDiffusionMemory
    
    # Create memory module
    memory = GraphDiffusionMemory(dim=64, memory_size=16)
    
    # Test node compression
    node = torch.randn(4, 64)
    compressed = memory.compress_node(node)
    
    print(f"  Node shape: {node.shape}")
    print(f"  Compressed shape: {compressed.shape}")
    
    # Check shapes
    assert compressed.shape == (4, 32), f"Compressed shape should be (4, 32), got {compressed.shape}"
    
    print("  ✅ Node Compression working correctly")

def test_training_with_memory():
    """Test training with GraphDiffusionMemory"""
    print("\nTesting Training with GraphDiffusionMemory...")
    
    from modell_V5_MAHIA_HyenaMoE import GraphDiffusionMemory
    
    # Create memory module
    memory = GraphDiffusionMemory(dim=64, memory_size=16)
    
    # Create optimizer
    optimizer = torch.optim.AdamW(memory.parameters(), lr=1e-3)
    
    # Training loop
    for i in range(10):
        x = torch.randn(4, 64)
        target = torch.randn(4, 64)
        
        # Forward pass
        out = memory(x)
        
        # Loss computation
        loss = torch.mean((out - target)**2)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if (i + 1) % 5 == 0:
            print(f"    Step {i+1}: Loss = {loss.item():.6f}")
    
    print("  ✅ Training with GraphDiffusionMemory successful")

def main():
    """Run all GraphDiffusionMemory tests"""
    print("MAHIA-V5 GraphDiffusionMemory Tests")
    print("=" * 35)
    
    try:
        test_graph_diffusion_memory()
        test_memory_retrieval()
        test_node_compression()
        test_training_with_memory()
        
        print("\n" + "=" * 35)
        print("🎉 All GraphDiffusionMemory tests passed!")
        print("🚀 MAHIA-V5 now supports enhanced graph diffusion memory!")
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

if __name__ == "__main__":
    main()