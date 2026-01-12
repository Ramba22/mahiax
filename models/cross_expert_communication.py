"""
Cross-Expert Communication Module for MAHIA
Feature exchange via low-rank adapters between experts
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Any, List, Tuple
import math

class LowRankAdapter(nn.Module):
    """
    Low-rank adapter for efficient parameter-efficient fine-tuning
    """
    
    def __init__(self, 
                 d_model: int,
                 bottleneck_dim: int = 32,
                 dropout: float = 0.1,
                 init_scale: float = 1e-3):
        """
        Initialize low-rank adapter
        
        Args:
            d_model: Model dimension
            bottleneck_dim: Bottleneck dimension for low-rank projection
            dropout: Dropout rate
            init_scale: Initialization scale for adapter weights
        """
        super().__init__()
        
        self.d_model = d_model
        self.bottleneck_dim = bottleneck_dim
        
        # Low-rank projection matrices
        self.down_proj = nn.Linear(d_model, bottleneck_dim)
        self.up_proj = nn.Linear(bottleneck_dim, d_model)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Initialize weights
        with torch.no_grad():
            nn.init.normal_(self.down_proj.weight, std=init_scale)
            nn.init.normal_(self.up_proj.weight, std=init_scale)
            nn.init.zeros_(self.down_proj.bias)
            nn.init.zeros_(self.up_proj.bias)
            
        print(f"✅ LowRankAdapter initialized: d_model={d_model}, bottleneck={bottleneck_dim}")
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through low-rank adapter
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)
            
        Returns:
            Adapted tensor of same shape
        """
        # Down projection
        hidden = self.down_proj(x)
        hidden = F.relu(hidden)
        
        # Dropout
        hidden = self.dropout(hidden)
        
        # Up projection
        output = self.up_proj(hidden)
        
        return output

class CrossExpertAttention(nn.Module):
    """
    Cross-attention mechanism for communication between experts
    """
    
    def __init__(self,
                 d_model: int,
                 num_heads: int = 8,
                 dropout: float = 0.1):
        """
        Initialize cross-expert attention
        
        Args:
            d_model: Model dimension
            num_heads: Number of attention heads
            dropout: Dropout rate
        """
        super().__init__()
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.dropout = dropout
        
        # Ensure d_model is divisible by num_heads
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        # Attention projections
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        
        # Dropout
        self.attn_dropout = nn.Dropout(dropout)
        self.out_dropout = nn.Dropout(dropout)
        
        print(f"✅ CrossExpertAttention initialized: d_model={d_model}, heads={num_heads}")
        
    def forward(self, 
                query: torch.Tensor,
                key: torch.Tensor,
                value: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass through cross-expert attention
        
        Args:
            query: Query tensor of shape (batch_size, seq_len_q, d_model)
            key: Key tensor of shape (batch_size, seq_len_k, d_model)
            value: Value tensor of shape (batch_size, seq_len_k, d_model)
            attention_mask: Optional attention mask
            
        Returns:
            Output tensor of shape (batch_size, seq_len_q, d_model)
        """
        batch_size, seq_len_q, _ = query.shape
        seq_len_k = key.shape[1]
        
        # Project to query, key, value
        q = self.q_proj(query).view(batch_size, seq_len_q, self.num_heads, self.head_dim)
        k = self.k_proj(key).view(batch_size, seq_len_k, self.num_heads, self.head_dim)
        v = self.v_proj(value).view(batch_size, seq_len_k, self.num_heads, self.head_dim)
        
        # Transpose for attention computation
        q = q.transpose(1, 2)  # (batch_size, num_heads, seq_len_q, head_dim)
        k = k.transpose(1, 2)  # (batch_size, num_heads, seq_len_k, head_dim)
        v = v.transpose(1, 2)  # (batch_size, num_heads, seq_len_k, head_dim)
        
        # Compute attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        # Apply attention mask if provided
        if attention_mask is not None:
            # Expand mask for multi-head attention
            mask = attention_mask.unsqueeze(1).unsqueeze(2)  # (batch_size, 1, 1, seq_len_k)
            scores = scores.masked_fill(mask == 0, float('-inf'))
            
        # Apply softmax
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)
        
        # Apply attention to values
        attn_output = torch.matmul(attn_weights, v)  # (batch_size, num_heads, seq_len_q, head_dim)
        
        # Transpose back and reshape
        attn_output = attn_output.transpose(1, 2).contiguous()  # (batch_size, seq_len_q, num_heads, head_dim)
        attn_output = attn_output.view(batch_size, seq_len_q, self.d_model)
        
        # Final projection
        output = self.out_proj(attn_output)
        output = self.out_dropout(output)
        
        return output

class ExpertCommunicationLayer(nn.Module):
    """
    Expert communication layer for feature exchange between experts
    """
    
    def __init__(self,
                 d_model: int,
                 num_experts: int,
                 bottleneck_dim: int = 32,
                 num_heads: int = 8,
                 dropout: float = 0.1):
        """
        Initialize expert communication layer
        
        Args:
            d_model: Model dimension
            num_experts: Number of experts
            bottleneck_dim: Bottleneck dimension for low-rank adapters
            num_heads: Number of attention heads
            dropout: Dropout rate
        """
        super().__init__()
        
        self.d_model = d_model
        self.num_experts = num_experts
        
        # Low-rank adapters for each expert
        self.adapters = nn.ModuleList([
            LowRankAdapter(d_model, bottleneck_dim, dropout)
            for _ in range(num_experts)
        ])
        
        # Cross-attention for inter-expert communication
        self.cross_attention = CrossExpertAttention(d_model, num_heads, dropout)
        
        # Gating mechanism to control communication
        self.gate = nn.Sequential(
            nn.Linear(d_model, d_model // 4),
            nn.ReLU(),
            nn.Linear(d_model // 4, 1),
            nn.Sigmoid()
        )
        
        # Layer normalization
        self.norm = nn.LayerNorm(d_model)
        
        print(f"✅ ExpertCommunicationLayer initialized: experts={num_experts}")
        
    def forward(self, expert_outputs: List[torch.Tensor]) -> List[torch.Tensor]:
        """
        Forward pass through expert communication layer
        
        Args:
            expert_outputs: List of expert outputs, each of shape (batch_size, seq_len, d_model)
            
        Returns:
            List of communicated expert outputs
        """
        batch_size, seq_len, d_model = expert_outputs[0].shape
        num_experts = len(expert_outputs)
        
        # Apply low-rank adapters to each expert output
        adapted_outputs = []
        for i, output in enumerate(expert_outputs):
            adapted = self.adapters[i](output)
            adapted_outputs.append(adapted)
            
        # Stack expert outputs for cross-attention
        # Shape: (num_experts, batch_size, seq_len, d_model)
        stacked_outputs = torch.stack(adapted_outputs, dim=0)
        
        # Reshape for cross-attention
        # Each expert attends to all other experts
        communicated_outputs = []
        
        for i in range(num_experts):
            # Current expert output
            query = expert_outputs[i]  # (batch_size, seq_len, d_model)
            
            # All expert outputs as key/value
            # Reshape to (batch_size, num_experts * seq_len, d_model)
            key_value = stacked_outputs.transpose(0, 1).reshape(batch_size, -1, d_model)
            
            # Apply cross-attention
            attn_output = self.cross_attention(query, key_value, key_value)
            
            # Apply gating to control communication strength
            gate_value = self.gate(query.mean(dim=1, keepdim=True))  # (batch_size, 1, 1)
            gated_output = gate_value * attn_output + (1 - gate_value) * query
            
            # Apply layer normalization and residual connection
            communicated = self.norm(gated_output + query)
            
            communicated_outputs.append(communicated)
            
        return communicated_outputs

class CrossExpertCommunicationModule(nn.Module):
    """
    Cross-Expert Communication Module for feature exchange via low-rank adapters
    """
    
    def __init__(self,
                 d_model: int,
                 num_experts: int,
                 num_layers: int = 2,
                 bottleneck_dim: int = 32,
                 num_heads: int = 8,
                 dropout: float = 0.1):
        """
        Initialize cross-expert communication module
        
        Args:
            d_model: Model dimension
            num_experts: Number of experts
            num_layers: Number of communication layers
            bottleneck_dim: Bottleneck dimension for low-rank adapters
            num_heads: Number of attention heads
            dropout: Dropout rate
        """
        super().__init__()
        
        self.d_model = d_model
        self.num_experts = num_experts
        self.num_layers = num_layers
        
        # Communication layers
        self.layers = nn.ModuleList([
            ExpertCommunicationLayer(
                d_model=d_model,
                num_experts=num_experts,
                bottleneck_dim=bottleneck_dim,
                num_heads=num_heads,
                dropout=dropout
            )
            for _ in range(num_layers)
        ])
        
        # Performance tracking
        self.stats = {
            'forward_passes': 0,
            'total_communications': 0,
            'avg_communication_time': 0.0
        }
        
        print(f"✅ CrossExpertCommunicationModule initialized: layers={num_layers}")
        
    def forward(self, expert_outputs: List[torch.Tensor]) -> List[torch.Tensor]:
        """
        Forward pass through cross-expert communication module
        
        Args:
            expert_outputs: List of expert outputs, each of shape (batch_size, seq_len, d_model)
            
        Returns:
            List of communicated expert outputs
        """
        import time
        start_time = time.time()
        
        # Apply communication layers
        communicated_outputs = expert_outputs
        for layer in self.layers:
            communicated_outputs = layer(communicated_outputs)
            
        # Update statistics
        self.stats['forward_passes'] += 1
        self.stats['total_communications'] += len(expert_outputs) * len(expert_outputs)
        self.stats['avg_communication_time'] = (
            (self.stats['avg_communication_time'] * (self.stats['forward_passes'] - 1) +
             (time.time() - start_time)) / self.stats['forward_passes']
        )
        
        return communicated_outputs
        
    def get_stats(self) -> Dict[str, Any]:
        """
        Get communication module statistics
        
        Returns:
            Dictionary of statistics
        """
        return self.stats.copy()
        
    def print_stats(self):
        """
        Print communication module statistics
        """
        stats = self.get_stats()
        print("\n" + "="*50)
        print("CROSS-EXPERT COMMUNICATION MODULE STATISTICS")
        print("="*50)
        print(f"Forward Passes: {stats['forward_passes']}")
        print(f"Total Communications: {stats['total_communications']}")
        print(f"Average Communication Time: {stats['avg_communication_time']*1000:.2f}ms")
        print("="*50)

# Example usage
def example_cross_expert_communication():
    """
    Example of Cross-Expert Communication Module usage
    """
    print("🔧 Setting up Cross-Expert Communication Module example...")
    
    # Check if CUDA is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create cross-expert communication module
    comm_module = CrossExpertCommunicationModule(
        d_model=768,
        num_experts=4,
        num_layers=2,
        bottleneck_dim=32,
        num_heads=8,
        dropout=0.1
    ).to(device)
    
    print("\n🚀 Testing Cross-Expert Communication Module...")
    
    # Create test expert outputs
    batch_size = 2
    seq_len = 64
    d_model = 768
    num_experts = 4
    
    expert_outputs = [
        torch.randn(batch_size, seq_len, d_model).to(device)
        for _ in range(num_experts)
    ]
    
    print(f"Expert outputs: {len(expert_outputs)} tensors of shape {expert_outputs[0].shape}")
    
    # Forward pass
    import time
    start_time = time.time()
    communicated_outputs = comm_module(expert_outputs)
    elapsed_time = time.time() - start_time
    
    print(f"Communicated outputs: {len(communicated_outputs)} tensors of shape {communicated_outputs[0].shape}")
    print(f"Forward pass completed in {elapsed_time*1000:.2f}ms")
    
    # Print statistics
    comm_module.print_stats()
    
    # Test multiple forward passes
    print("\n🚀 Testing multiple forward passes...")
    for i in range(3):
        test_outputs = [
            torch.randn(batch_size, seq_len, d_model).to(device)
            for _ in range(num_experts)
        ]
        result = comm_module(test_outputs)
        print(f"  Pass {i+1}: Completed")

if __name__ == "__main__":
    example_cross_expert_communication()