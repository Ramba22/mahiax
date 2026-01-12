"""
Cross-Expert Communication Layer for MAHIA
Enables interaction between experts via MoE graphs or cross-attention between expert heads
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Any
import numpy as np

class ExpertGraph:
    """Represents connections between experts in a MoE graph"""
    
    def __init__(self, num_experts: int):
        """
        Initialize expert graph
        
        Args:
            num_experts: Number of experts in the system
        """
        self.num_experts = num_experts
        # Adjacency matrix representing expert connections
        self.connections = torch.zeros(num_experts, num_experts)
        # Communication weights between experts
        self.weights = torch.ones(num_experts, num_experts) * 0.1
        
        print(f"✅ Expert Graph initialized with {num_experts} experts")
    
    def add_connection(self, expert_i: int, expert_j: int, weight: float = 1.0):
        """
        Add connection between two experts
        
        Args:
            expert_i: Index of first expert
            expert_j: Index of second expert
            weight: Connection weight
        """
        if 0 <= expert_i < self.num_experts and 0 <= expert_j < self.num_experts:
            self.connections[expert_i, expert_j] = 1
            self.connections[expert_j, expert_i] = 1  # Symmetric connection
            self.weights[expert_i, expert_j] = weight
            self.weights[expert_j, expert_i] = weight
        else:
            print(f"⚠️  Invalid expert indices: {expert_i}, {expert_j}")
    
    def remove_connection(self, expert_i: int, expert_j: int):
        """
        Remove connection between two experts
        
        Args:
            expert_i: Index of first expert
            expert_j: Index of second expert
        """
        if 0 <= expert_i < self.num_experts and 0 <= expert_j < self.num_experts:
            self.connections[expert_i, expert_j] = 0
            self.connections[expert_j, expert_i] = 0
            self.weights[expert_i, expert_j] = 0.1
            self.weights[expert_j, expert_i] = 0.1
    
    def get_neighbors(self, expert_idx: int) -> List[int]:
        """
        Get neighboring experts for a given expert
        
        Args:
            expert_idx: Index of expert
            
        Returns:
            List[int]: Indices of neighboring experts
        """
        if 0 <= expert_idx < self.num_experts:
            neighbors = torch.where(self.connections[expert_idx] > 0)[0].tolist()
            return neighbors
        return []
    
    def compute_communication_matrix(self) -> torch.Tensor:
        """
        Compute communication matrix with weights
        
        Returns:
            torch.Tensor: Weighted communication matrix
        """
        return self.connections * self.weights

class CrossAttentionExpertCommunication(nn.Module):
    """Cross-attention mechanism for expert communication"""
    
    def __init__(self, hidden_size: int, num_heads: int = 8):
        """
        Initialize cross-attention communication layer
        
        Args:
            hidden_size: Hidden size of expert representations
            num_heads: Number of attention heads
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        
        # Cross-attention layers
        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.k_proj = nn.Linear(hidden_size, hidden_size)
        self.v_proj = nn.Linear(hidden_size, hidden_size)
        self.out_proj = nn.Linear(hidden_size, hidden_size)
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(hidden_size)
        
        print(f"✅ Cross-Attention Expert Communication initialized (hidden_size={hidden_size}, heads={num_heads})")
    
    def forward(self, expert_outputs: torch.Tensor, 
                attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Apply cross-attention between expert outputs
        
        Args:
            expert_outputs: Tensor of shape (batch_size, num_experts, hidden_size)
            attention_mask: Optional attention mask
            
        Returns:
            torch.Tensor: Communicated expert outputs
        """
        batch_size, num_experts, hidden_size = expert_outputs.shape
        
        # Apply layer normalization
        residual = expert_outputs
        expert_outputs = self.layer_norm(expert_outputs)
        
        # Project to query, key, value
        q = self.q_proj(expert_outputs)  # (batch, experts, hidden)
        k = self.k_proj(expert_outputs)  # (batch, experts, hidden)
        v = self.v_proj(expert_outputs)  # (batch, experts, hidden)
        
        # Reshape for multi-head attention
        q = q.view(batch_size, num_experts, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, num_experts, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, num_experts, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Compute attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        
        # Apply attention mask if provided
        if attention_mask is not None:
            scores = scores.masked_fill(attention_mask.unsqueeze(1).unsqueeze(2) == 0, float('-inf'))
        
        # Apply softmax
        attention_weights = F.softmax(scores, dim=-1)
        
        # Apply attention to values
        attended = torch.matmul(attention_weights, v)
        
        # Reshape back
        attended = attended.transpose(1, 2).contiguous().view(batch_size, num_experts, hidden_size)
        
        # Apply output projection
        output = self.out_proj(attended)
        
        # Add residual connection
        output = output + residual
        
        return output

class ExpertCommunicationLayer(nn.Module):
    """Main expert communication layer combining graph and attention mechanisms"""
    
    def __init__(self, num_experts: int, hidden_size: int, 
                 use_graph: bool = True, use_attention: bool = True):
        """
        Initialize expert communication layer
        
        Args:
            num_experts: Number of experts
            hidden_size: Hidden size of expert representations
            use_graph: Whether to use graph-based communication
            use_attention: Whether to use attention-based communication
        """
        super().__init__()
        self.num_experts = num_experts
        self.hidden_size = hidden_size
        self.use_graph = use_graph
        self.use_attention = use_attention
        
        # Initialize components
        if use_graph:
            self.expert_graph = ExpertGraph(num_experts)
            # Add some default connections
            for i in range(num_experts - 1):
                self.expert_graph.add_connection(i, i + 1, 0.5)
        
        if use_attention:
            self.cross_attention = CrossAttentionExpertCommunication(hidden_size)
        
        # Communication gating
        self.communication_gate = nn.Linear(hidden_size * 2, 1)
        self.gate_activation = nn.Sigmoid()
        
        print(f"✅ Expert Communication Layer initialized ({num_experts} experts, {hidden_size} hidden)")
    
    def forward(self, expert_outputs: torch.Tensor, 
                expert_indices: Optional[List[int]] = None) -> torch.Tensor:
        """
        Apply expert communication
        
        Args:
            expert_outputs: Tensor of shape (batch_size, num_experts, hidden_size)
            expert_indices: Optional list of active expert indices
            
        Returns:
            torch.Tensor: Communicated expert outputs
        """
        batch_size, num_experts, hidden_size = expert_outputs.shape
        
        # Ensure we have the right number of experts
        if num_experts != self.num_experts:
            print(f"⚠️  Expert count mismatch: expected {self.num_experts}, got {num_experts}")
            return expert_outputs
        
        communicated_outputs = expert_outputs.clone()
        
        # Graph-based communication
        if self.use_graph and hasattr(self, 'expert_graph'):
            communicated_outputs = self._apply_graph_communication(
                communicated_outputs, expert_indices
            )
        
        # Attention-based communication
        if self.use_attention and hasattr(self, 'cross_attention'):
            communicated_outputs = self.cross_attention(communicated_outputs)
        
        # Gate the communication to preserve original information
        gated_outputs = self._gate_communication(expert_outputs, communicated_outputs)
        
        return gated_outputs
    
    def _apply_graph_communication(self, expert_outputs: torch.Tensor,
                                 expert_indices: Optional[List[int]] = None) -> torch.Tensor:
        """
        Apply graph-based communication between experts
        
        Args:
            expert_outputs: Expert outputs tensor
            expert_indices: Active expert indices
            
        Returns:
            torch.Tensor: Graph-communicated outputs
        """
        batch_size, num_experts, hidden_size = expert_outputs.shape
        
        # Get communication matrix
        comm_matrix = self.expert_graph.compute_communication_matrix()
        
        # Apply to expert outputs
        # This is a simplified implementation - in practice, you might want more sophisticated aggregation
        communicated = torch.matmul(comm_matrix, expert_outputs)
        
        return communicated
    
    def _gate_communication(self, original_outputs: torch.Tensor,
                          communicated_outputs: torch.Tensor) -> torch.Tensor:
        """
        Gate communication to balance original and communicated information
        
        Args:
            original_outputs: Original expert outputs
            communicated_outputs: Communicated expert outputs
            
        Returns:
            torch.Tensor: Gated outputs
        """
        # Concatenate original and communicated for gating
        gate_input = torch.cat([original_outputs, communicated_outputs], dim=-1)
        
        # Compute gate values
        gate_values = self.gate_activation(self.communication_gate(gate_input))
        
        # Apply gating
        gated_outputs = gate_values * communicated_outputs + (1 - gate_values) * original_outputs
        
        return gated_outputs
    
    def add_expert_connection(self, expert_i: int, expert_j: int, weight: float = 1.0):
        """
        Add connection between two experts in the graph
        
        Args:
            expert_i: Index of first expert
            expert_j: Index of second expert
            weight: Connection weight
        """
        if self.use_graph and hasattr(self, 'expert_graph'):
            self.expert_graph.add_connection(expert_i, expert_j, weight)
    
    def get_communication_stats(self) -> Dict[str, Any]:
        """
        Get communication statistics
        
        Returns:
            Dict[str, Any]: Communication statistics
        """
        stats = {
            "use_graph": self.use_graph,
            "use_attention": self.use_attention,
            "num_experts": self.num_experts
        }
        
        if self.use_graph and hasattr(self, 'expert_graph'):
            connections = torch.sum(self.expert_graph.connections).item() / 2  # Divide by 2 for symmetric
            stats["graph_connections"] = int(connections)
        
        return stats

class CrossExpertCommunicationBenchmark:
    """Benchmark for cross-expert communication performance"""
    
    def __init__(self, model: nn.Module):
        self.model = model
        self.device = next(model.parameters()).device if model else "cpu"
    
    def benchmark_communication_performance(self, 
                                          batch_sizes: List[int] = [8, 16, 32],
                                          expert_counts: List[int] = [4, 8, 16]) -> Dict[str, Any]:
        """
        Benchmark communication performance with different configurations
        
        Args:
            batch_sizes: List of batch sizes to test
            expert_counts: List of expert counts to test
            
        Returns:
            Dict[str, Any]: Benchmark results
        """
        print("🚀 Benchmarking Cross-Expert Communication Performance")
        print("=" * 60)
        
        results = {}
        
        for batch_size in batch_sizes:
            for num_experts in expert_counts:
                key = f"bs{batch_size}_experts{num_experts}"
                print(f"\n📊 Testing BS={batch_size}, Experts={num_experts}...")
                
                # Create communication layer
                comm_layer = ExpertCommunicationLayer(
                    num_experts=num_experts,
                    hidden_size=128,
                    use_graph=True,
                    use_attention=True
                ).to(self.device)
                
                # Create test data
                expert_outputs = torch.randn(batch_size, num_experts, 128, device=self.device)
                
                # Warmup
                for _ in range(3):
                    _ = comm_layer(expert_outputs)
                
                # Benchmark
                times = []
                for _ in range(10):
                    start_time = time.time()
                    output = comm_layer(expert_outputs)
                    end_time = time.time()
                    times.append(end_time - start_time)
                
                avg_time = np.mean(times)
                std_time = np.std(times)
                
                results[key] = {
                    "batch_size": batch_size,
                    "num_experts": num_experts,
                    "avg_time": float(avg_time),
                    "std_time": float(std_time),
                    "throughput": batch_size / avg_time if avg_time > 0 else 0
                }
                
                print(f"   Avg Time: {avg_time*1000:.2f}ms ± {std_time*1000:.2f}ms")
                print(f"   Throughput: {results[key]['throughput']:.1f} samples/s")
        
        return results

# Example usage
def example_cross_expert_communication():
    """Example of cross-expert communication"""
    print("🔧 Setting up cross-expert communication example...")
    
    # Create communication layer
    comm_layer = ExpertCommunicationLayer(
        num_experts=8,
        hidden_size=128,
        use_graph=True,
        use_attention=True
    )
    
    # Add some expert connections
    comm_layer.add_expert_connection(0, 1, 0.8)
    comm_layer.add_expert_connection(1, 2, 0.7)
    comm_layer.add_expert_connection(2, 3, 0.6)
    comm_layer.add_expert_connection(0, 3, 0.5)
    
    print("✅ Created expert communication layer with custom connections")
    
    # Create test data
    batch_size = 4
    num_experts = 8
    hidden_size = 128
    
    expert_outputs = torch.randn(batch_size, num_experts, hidden_size)
    print(f"✅ Created test data: {batch_size} batches, {num_experts} experts, {hidden_size} hidden dims")
    
    # Apply communication
    print("\n🔄 Applying cross-expert communication...")
    communicated_outputs = comm_layer(expert_outputs)
    
    print(f"✅ Communication applied successfully")
    print(f"   Input shape: {expert_outputs.shape}")
    print(f"   Output shape: {communicated_outputs.shape}")
    
    # Show communication stats
    stats = comm_layer.get_communication_stats()
    print(f"\n📊 Communication Statistics:")
    print(f"   Graph Connections: {stats.get('graph_connections', 0)}")
    print(f"   Using Graph: {stats['use_graph']}")
    print(f"   Using Attention: {stats['use_attention']}")
    
    # Benchmark performance
    print("\n" + "="*60)
    benchmark = CrossExpertCommunicationBenchmark(None)
    results = benchmark.benchmark_communication_performance(
        batch_sizes=[4, 8],
        expert_counts=[4, 8]
    )
    
    print(f"\n📈 Performance Summary:")
    for key, result in results.items():
        print(f"   {key}: {result['throughput']:.1f} samples/s")
    
    return comm_layer

if __name__ == "__main__":
    import time
    example_cross_expert_communication()