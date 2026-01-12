"""
Self-Balancing Experts v3 for MAHIA
Implementation with learned gate temperature and EM-based load equalization
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Any, List, Tuple
import math

class ExpertLoadBalancer:
    """
    Expert load balancer with EM-based load equalization
    """
    
    def __init__(self, 
                 num_experts: int,
                 em_iterations: int = 5,
                 load_balance_weight: float = 0.1):
        """
        Initialize expert load balancer
        
        Args:
            num_experts: Number of experts
            em_iterations: Number of EM iterations for load balancing
            load_balance_weight: Weight for load balancing loss
        """
        self.num_experts = num_experts
        self.em_iterations = em_iterations
        self.load_balance_weight = load_balance_weight
        
        # Track expert usage statistics
        self.expert_counts = torch.zeros(num_experts, dtype=torch.float32)
        self.total_tokens = 0
        
        # Learned gate temperature
        self.gate_temperature = nn.Parameter(torch.ones(1) * 1.0)
        
        print(f"✅ ExpertLoadBalancer initialized with {num_experts} experts")
        
    def compute_load_balance_loss(self, expert_indices: torch.Tensor) -> torch.Tensor:
        """
        Compute load balancing loss using EM-based approach
        
        Args:
            expert_indices: Indices of selected experts for each token
            
        Returns:
            Load balancing loss
        """
        # Count expert usage
        expert_usage = torch.bincount(expert_indices, minlength=self.num_experts)
        expert_usage = expert_usage.float()
        
        # Update running statistics
        self.expert_counts += expert_usage
        self.total_tokens += expert_indices.size(0)
        
        # Compute expected uniform distribution
        uniform_dist = torch.ones(self.num_experts) / self.num_experts
        
        # Compute actual usage distribution
        if self.total_tokens > 0:
            actual_dist = self.expert_counts / self.total_tokens
        else:
            actual_dist = uniform_dist
            
        # Compute KL divergence as load balancing loss
        # Add small epsilon to avoid log(0)
        epsilon = 1e-8
        actual_dist = actual_dist + epsilon
        actual_dist = actual_dist / actual_dist.sum()
        
        kl_div = F.kl_div(
            torch.log(actual_dist), 
            uniform_dist, 
            reduction='sum'
        )
        
        return self.load_balance_weight * kl_div
        
    def adjust_gate_temperature(self, loss: torch.Tensor):
        """
        Adjust gate temperature based on load balancing feedback
        
        Args:
            loss: Current loss value
        """
        # Simple adaptive temperature adjustment
        # Reduce temperature when load is balanced, increase when imbalanced
        with torch.no_grad():
            if loss < 0.01:  # Well balanced
                self.gate_temperature.data *= 0.99
            elif loss > 0.1:  # Poorly balanced
                self.gate_temperature.data *= 1.01
                
            # Clamp temperature to reasonable range
            self.gate_temperature.data = torch.clamp(
                self.gate_temperature.data, 
                min=0.1, 
                max=10.0
            )
            
    def get_expert_utilization(self) -> Dict[int, float]:
        """
        Get expert utilization statistics
        
        Returns:
            Dictionary mapping expert index to utilization percentage
        """
        if self.total_tokens == 0:
            return {i: 0.0 for i in range(self.num_experts)}
            
        utilizations = {}
        for i in range(self.num_experts):
            utilizations[i] = (self.expert_counts[i] / self.total_tokens).item()
            
        return utilizations

class SelfBalancingExpertRouter(nn.Module):
    """
    Self-balancing expert router with learned gate temperature
    """
    
    def __init__(self,
                 d_model: int,
                 num_experts: int,
                 top_k: int = 2,
                 gate_temperature: float = 1.0):
        """
        Initialize self-balancing expert router
        
        Args:
            d_model: Model dimension
            num_experts: Number of experts
            top_k: Number of top experts to select
            gate_temperature: Initial gate temperature
        """
        super().__init__()
        
        self.d_model = d_model
        self.num_experts = num_experts
        self.top_k = top_k
        
        # Gate network for expert selection
        self.gate = nn.Linear(d_model, num_experts)
        
        # Learned gate temperature
        self.gate_temperature = nn.Parameter(torch.ones(1) * gate_temperature)
        
        # Load balancer
        self.load_balancer = ExpertLoadBalancer(
            num_experts=num_experts,
            em_iterations=5,
            load_balance_weight=0.1
        )
        
        print(f"✅ SelfBalancingExpertRouter initialized with {num_experts} experts")
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass through expert router
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)
            
        Returns:
            Tuple of (gate_logits, expert_indices, load_balance_loss)
        """
        batch_size, seq_len, d_model = x.shape
        
        # Flatten for expert routing
        x_flat = x.view(-1, d_model)  # (batch_size * seq_len, d_model)
        
        # Compute gate logits
        gate_logits = self.gate(x_flat)  # (batch_size * seq_len, num_experts)
        
        # Apply temperature scaling
        tempered_logits = gate_logits / self.gate_temperature
        
        # Compute expert weights using softmax
        expert_weights = F.softmax(tempered_logits, dim=-1)
        
        # Select top-k experts
        top_k_weights, top_k_indices = torch.topk(
            expert_weights, 
            k=min(self.top_k, self.num_experts), 
            dim=-1
        )
        
        # Normalize top-k weights
        top_k_weights = top_k_weights / (top_k_weights.sum(dim=-1, keepdim=True) + 1e-8)
        
        # Compute load balancing loss
        # For simplicity, we'll use the first expert for load balancing computation
        first_expert_indices = top_k_indices[:, 0]
        load_balance_loss = self.load_balancer.compute_load_balance_loss(first_expert_indices)
        
        # Adjust gate temperature based on load
        self.load_balancer.adjust_gate_temperature(load_balance_loss)
        
        return tempered_logits, top_k_indices, load_balance_loss
        
    def get_expert_utilization(self) -> Dict[int, float]:
        """
        Get expert utilization statistics
        
        Returns:
            Dictionary mapping expert index to utilization percentage
        """
        return self.load_balancer.get_expert_utilization()

class EMExpertBalancer:
    """
    Expectation-Maximization based expert balancer
    """
    
    def __init__(self, 
                 num_experts: int,
                 em_iterations: int = 5):
        """
        Initialize EM expert balancer
        
        Args:
            num_experts: Number of experts
            em_iterations: Number of EM iterations
        """
        self.num_experts = num_experts
        self.em_iterations = em_iterations
        
        # Expert assignment probabilities
        self.expert_probs = torch.ones(num_experts) / num_experts
        
    def balance_expert_load(self, 
                           token_expert_scores: torch.Tensor) -> torch.Tensor:
        """
        Balance expert load using EM algorithm
        
        Args:
            token_expert_scores: Scores for each token-expert pair
                            Shape: (num_tokens, num_experts)
                            
        Returns:
            Balanced expert assignments
        """
        num_tokens, num_experts = token_expert_scores.shape
        
        # Normalize scores
        scores = F.softmax(token_expert_scores, dim=-1)
        
        # Initialize expert probabilities
        expert_probs = self.expert_probs.clone()
        
        # EM iterations
        for iteration in range(self.em_iterations):
            # E-step: Compute expected expert assignments
            expected_assignments = scores * expert_probs.unsqueeze(0)
            expected_assignments = expected_assignments / (
                expected_assignments.sum(dim=-1, keepdim=True) + 1e-8
            )
            
            # M-step: Update expert probabilities
            expert_counts = expected_assignments.sum(dim=0)
            expert_probs = expert_counts / (expert_counts.sum() + 1e-8)
            
        # Final assignment
        balanced_scores = scores * expert_probs.unsqueeze(0)
        
        return balanced_scores

class SelfBalancingExpertsV3(nn.Module):
    """
    Self-Balancing Experts v3 with learned gate temperature and EM-based load equalization
    """
    
    def __init__(self,
                 d_model: int,
                 num_experts: int,
                 expert_dim: int,
                 top_k: int = 2,
                 gate_temperature: float = 1.0):
        """
        Initialize Self-Balancing Experts v3
        
        Args:
            d_model: Model dimension
            num_experts: Number of experts
            expert_dim: Expert dimension
            top_k: Number of top experts to select
            gate_temperature: Initial gate temperature
        """
        super().__init__()
        
        self.d_model = d_model
        self.num_experts = num_experts
        self.expert_dim = expert_dim
        self.top_k = top_k
        
        # Expert router
        self.router = SelfBalancingExpertRouter(
            d_model=d_model,
            num_experts=num_experts,
            top_k=top_k,
            gate_temperature=gate_temperature
        )
        
        # EM-based balancer
        self.em_balancer = EMExpertBalancer(
            num_experts=num_experts,
            em_iterations=5
        )
        
        # Expert networks
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, expert_dim),
                nn.ReLU(),
                nn.Linear(expert_dim, d_model)
            )
            for _ in range(num_experts)
        ])
        
        # Shared expert for fallback
        self.shared_expert = nn.Sequential(
            nn.Linear(d_model, expert_dim),
            nn.ReLU(),
            nn.Linear(expert_dim, d_model)
        )
        
        # Performance tracking
        self.stats = {
            'forward_passes': 0,
            'total_tokens': 0,
            'expert_utilization': {i: 0.0 for i in range(num_experts)},
            'load_balance_losses': []
        }
        
        print(f"✅ SelfBalancingExpertsV3 initialized with {num_experts} experts")
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through self-balancing experts
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)
            
        Returns:
            Tuple of (output, load_balance_loss)
        """
        batch_size, seq_len, d_model = x.shape
        
        # Route tokens to experts
        gate_logits, expert_indices, load_balance_loss = self.router(x)
        
        # Flatten for expert processing
        x_flat = x.view(-1, d_model)  # (batch_size * seq_len, d_model)
        flat_indices = expert_indices.view(-1, self.top_k)  # (batch_size * seq_len, top_k)
        
        # Apply EM-based balancing
        balanced_scores = self.em_balancer.balance_expert_load(gate_logits)
        
        # Process tokens with selected experts
        output_tokens = torch.zeros_like(x_flat)
        
        # Process each expert
        for expert_idx in range(self.num_experts):
            # Find tokens assigned to this expert (using first expert in top-k)
            mask = (flat_indices[:, 0] == expert_idx)
            if mask.any():
                expert_input = x_flat[mask]
                expert_output = self.experts[expert_idx](expert_input)
                output_tokens[mask] = expert_output
                
        # Apply weighted combination for top-k experts
        output_combined = torch.zeros_like(x_flat)
        for k in range(self.top_k):
            weights = balanced_scores[torch.arange(balanced_scores.size(0)), flat_indices[:, k]]
            expert_outputs = torch.zeros_like(x_flat)
            
            for expert_idx in range(self.num_experts):
                mask = (flat_indices[:, k] == expert_idx)
                if mask.any():
                    expert_input = x_flat[mask]
                    expert_output = self.experts[expert_idx](expert_input)
                    expert_outputs[mask] = expert_output
                    
            output_combined += weights.unsqueeze(-1) * expert_outputs
            
        # Add shared expert output for better generalization
        shared_output = self.shared_expert(x_flat)
        output_combined = output_combined + 0.1 * shared_output
        
        # Reshape output
        output = output_combined.view(batch_size, seq_len, d_model)
        
        # Update statistics
        self.stats['forward_passes'] += 1
        self.stats['total_tokens'] += batch_size * seq_len
        self.stats['expert_utilization'] = self.router.get_expert_utilization()
        self.stats['load_balance_losses'].append(load_balance_loss.item())
        
        return output, load_balance_loss
        
    def get_expert_utilization(self) -> Dict[int, float]:
        """
        Get expert utilization statistics
        
        Returns:
            Dictionary mapping expert index to utilization percentage
        """
        return self.router.get_expert_utilization()
        
    def get_stats(self) -> Dict[str, Any]:
        """
        Get self-balancing experts statistics
        
        Returns:
            Dictionary of statistics
        """
        return {
            'forward_passes': self.stats['forward_passes'],
            'total_tokens': self.stats['total_tokens'],
            'expert_utilization': self.stats['expert_utilization'].copy(),
            'avg_load_balance_loss': (
                sum(self.stats['load_balance_losses']) / len(self.stats['load_balance_losses'])
                if self.stats['load_balance_losses'] else 0.0
            ),
            'current_gate_temperature': self.router.gate_temperature.item()
        }
        
    def print_stats(self):
        """
        Print self-balancing experts statistics
        """
        stats = self.get_stats()
        print("\n" + "="*50)
        print("SELF-BALANCING EXPERTS V3 STATISTICS")
        print("="*50)
        print(f"Forward Passes: {stats['forward_passes']}")
        print(f"Total Tokens: {stats['total_tokens']}")
        print(f"Current Gate Temperature: {stats['current_gate_temperature']:.4f}")
        print(f"Avg Load Balance Loss: {stats['avg_load_balance_loss']:.6f}")
        print("\nExpert Utilization:")
        for expert_idx, utilization in stats['expert_utilization'].items():
            print(f"  Expert {expert_idx}: {utilization*100:.2f}%")
        print("="*50)

# Example usage
def example_self_balancing_experts():
    """
    Example of Self-Balancing Experts v3 usage
    """
    print("🔧 Setting up Self-Balancing Experts v3 example...")
    
    # Check if CUDA is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create self-balancing experts
    experts = SelfBalancingExpertsV3(
        d_model=768,
        num_experts=8,
        expert_dim=2048,
        top_k=2,
        gate_temperature=1.0
    ).to(device)
    
    print("\n🚀 Testing Self-Balancing Experts v3...")
    
    # Create test input
    batch_size = 4
    seq_len = 128
    d_model = 768
    
    input_tensor = torch.randn(batch_size, seq_len, d_model).to(device)
    
    print(f"Input shape: {input_tensor.shape}")
    
    # Forward pass
    import time
    start_time = time.time()
    output, load_balance_loss = experts(input_tensor)
    elapsed_time = time.time() - start_time
    
    print(f"Output shape: {output.shape}")
    print(f"Load balance loss: {load_balance_loss.item():.6f}")
    print(f"Forward pass completed in {elapsed_time*1000:.2f}ms")
    
    # Print statistics
    experts.print_stats()
    
    # Test multiple forward passes to see balancing in action
    print("\n🚀 Testing multiple forward passes for load balancing...")
    for i in range(5):
        test_input = torch.randn(batch_size, seq_len, d_model).to(device)
        output, loss = experts(test_input)
        print(f"  Pass {i+1}: Loss = {loss.item():.6f}, Temp = {experts.router.gate_temperature.item():.4f}")
    
    # Print final statistics
    experts.print_stats()

if __name__ == "__main__":
    example_self_balancing_experts()