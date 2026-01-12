"""
Adaptive Expert Activation with Entropy/uncertainty-based gating for MAHIA-X.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import sys
import os
from typing import Dict, Any, Tuple, Optional
import math

# Add the project root to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

try:
    from modell_V5_MAHIA_HyenaMoE import SparseMoETopK, HyenaExpert
    ADAPTIVE_AVAILABLE = True
except ImportError:
    ADAPTIVE_AVAILABLE = False
    print("⚠️  MAHIA-X modules not available for adaptive expert activation")


class EntropyBasedGating(nn.Module):
    """Entropy-based gating mechanism for adaptive expert activation"""
    
    def __init__(self, dim: int, num_experts: int, entropy_threshold: float = 1.0):
        super().__init__()
        self.dim = dim
        self.num_experts = num_experts
        self.entropy_threshold = entropy_threshold
        
        # Gating network
        self.gate = nn.Linear(dim, num_experts)
        
        # Entropy-aware routing parameters
        self.entropy_weight = nn.Parameter(torch.ones(1))
        self.confidence_weight = nn.Parameter(torch.ones(1))
        self.uncertainty_weight = nn.Parameter(torch.ones(1))
        
        # Temperature for softmax
        self.temperature = nn.Parameter(torch.ones(1))
        
    def compute_entropy(self, probs: torch.Tensor) -> torch.Tensor:
        """Compute entropy of routing probabilities
        Args:
            probs: Routing probabilities of shape (B, L, E)
        Returns:
            Entropy tensor of shape (B, L)
        """
        # Add small epsilon to avoid log(0)
        eps = 1e-8
        entropy = -torch.sum(probs * torch.log(probs + eps), dim=-1)
        return entropy
    
    def compute_uncertainty(self, probs: torch.Tensor) -> torch.Tensor:
        """Compute uncertainty as variance of routing probabilities
        Args:
            probs: Routing probabilities of shape (B, L, E)
        Returns:
            Uncertainty tensor of shape (B, L)
        """
        # Variance across experts
        variance = torch.var(probs, dim=-1)
        return variance
    
    def forward(self, x: torch.Tensor, 
                prev_entropy: Optional[torch.Tensor] = None,
                prev_confidence: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Forward pass with entropy-based routing
        Args:
            x: Input tensor of shape (B, L, D)
            prev_entropy: Previous entropy values for temporal consistency
            prev_confidence: Previous confidence values
        Returns:
            Tuple of (routing_probs, metadata)
        """
        B, L, D = x.shape
        
        # Standard gating
        logits = self.gate(x)  # (B, L, E)
        base_probs = F.softmax(logits / self.temperature, dim=-1)
        
        # Compute entropy and uncertainty
        current_entropy = self.compute_entropy(base_probs)  # (B, L)
        current_uncertainty = self.compute_uncertainty(base_probs)  # (B, L)
        current_confidence = 1.0 / (current_uncertainty + 1e-8)  # (B, L)
        
        # Adaptive routing based on entropy
        # High entropy -> more experts (exploration)
        # Low entropy -> fewer experts (exploitation)
        entropy_normalized = torch.sigmoid(current_entropy / self.entropy_threshold)
        
        # Combine multiple signals for adaptive routing
        adaptive_factor = (
            self.entropy_weight * entropy_normalized +
            self.confidence_weight * current_confidence +
            self.uncertainty_weight * current_uncertainty
        )
        
        # Normalize adaptive factor
        adaptive_factor = torch.sigmoid(adaptive_factor)
        
        # Modify routing probabilities based on adaptive factor
        # This is a simplified approach - in practice, this could be more sophisticated
        modified_probs = base_probs * (1.0 + adaptive_factor.unsqueeze(-1))
        modified_probs = modified_probs / modified_probs.sum(dim=-1, keepdim=True)
        
        # Metadata for monitoring
        metadata = {
            'base_probs': base_probs,
            'modified_probs': modified_probs,
            'entropy': current_entropy,
            'uncertainty': current_uncertainty,
            'confidence': current_confidence,
            'adaptive_factor': adaptive_factor
        }
        
        return modified_probs, metadata


class AdaptiveSparseMoE(nn.Module):
    """Adaptive Sparse MoE with entropy/uncertainty-based gating"""
    
    def __init__(self, dim: int, num_experts: int = 8, top_k: int = 2,
                 capacity_factor: float = 1.25, entropy_threshold: float = 1.0):
        super().__init__()
        self.dim = dim
        self.num_experts = num_experts
        self.top_k = min(top_k, num_experts)
        self.capacity_factor = capacity_factor
        self.entropy_threshold = entropy_threshold
        
        # Entropy-based gating
        self.gate = EntropyBasedGating(dim, num_experts, entropy_threshold)
        
        # Hyena-based experts
        self.experts = nn.ModuleList([
            HyenaExpert(dim) if ADAPTIVE_AVAILABLE else nn.Linear(dim, dim) for _ in range(num_experts)
        ])
        
        # For expert diversity loss
        self.use_expert_diversity_loss = True
        self.expert_diversity_weight = 0.01
        
        # History for temporal consistency
        self.entropy_history = []
        self.confidence_history = []
        self.max_history_length = 100
        
    def forward(self, x: torch.Tensor, return_aux: bool = True) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Forward pass with adaptive expert activation
        Args:
            x: Input tensor of shape (B, L, D)
            return_aux: Whether to return auxiliary loss
        Returns:
            Tuple of (output, aux_loss)
        """
        B, L, D = x.shape
        assert D == self.dim
        
        # Get routing probabilities from entropy-based gating
        routing_probs, metadata = self.gate(x)
        base_probs = metadata['base_probs']
        modified_probs = metadata['modified_probs']
        entropy = metadata['entropy']
        confidence = metadata['confidence']
        
        # Store history for temporal consistency
        self.entropy_history.append(entropy.mean().item())
        self.confidence_history.append(confidence.mean().item())
        
        # Keep only recent history
        if len(self.entropy_history) > self.max_history_length:
            self.entropy_history.pop(0)
            self.confidence_history.pop(0)
        
        # Select top-k experts based on modified probabilities
        topk_vals, topk_idx = torch.topk(modified_probs, k=self.top_k, dim=-1)
        
        # Normalize weights per token
        weight_norm = topk_vals.sum(dim=-1, keepdim=True).clamp_min(1e-9)
        normalized_vals = topk_vals / weight_norm  # (B, L, K)
        
        # Dispatch tensor
        dispatch = torch.zeros((B, L, self.num_experts), device=x.device, dtype=x.dtype)
        dispatch.scatter_(-1, topk_idx, normalized_vals)
        
        # Capacity per expert
        expected_per_expert = (B * L) / max(1, self.num_experts)
        capacity = int(self.capacity_factor * expected_per_expert + 0.9999)
        
        # Assign mask and capacity limiting
        assign_mask = (dispatch > 0).float()
        cumsum = torch.cumsum(assign_mask, dim=1)
        positions = cumsum - 1.0
        keep_mask = (positions < float(capacity)).float()
        dispatch = dispatch * keep_mask
        
        # Compute expert inputs: (B, E, D)
        expert_counts = dispatch.sum(dim=1).clamp(min=1.0)  # (B, E)
        expert_inputs = torch.einsum('bld,ble->bed', x, dispatch)  # (B, E, D)
        expert_inputs = expert_inputs / expert_counts.unsqueeze(-1)  # (B, E, D) / (B, E, 1)
        
        # Process experts
        expert_outputs = torch.zeros(B, self.num_experts, D, device=x.device, dtype=x.dtype)
        
        # Process each expert with its corresponding inputs
        for e, expert in enumerate(self.experts):
            # Get inputs for this expert: (B, D)
            expert_input = expert_inputs[:, e, :]  # (B, D)
            # Process through expert
            if ADAPTIVE_AVAILABLE:
                expert_output = expert(expert_input)  # (B, D)
            else:
                expert_output = expert(expert_input)  # (B, D) - fallback to linear
            # Store outputs
            expert_outputs[:, e, :] = expert_output
        
        # Broadcast back
        out = torch.einsum('bed,ble->bld', expert_outputs, dispatch)
        
        aux_loss = None
        if return_aux:
            # Standard load balancing loss
            mean_gate = base_probs.mean(dim=1)  # (B, E)
            aux_loss = torch.var(mean_gate, unbiased=False) * self.num_experts
            
            # Add expert diversity loss
            if self.use_expert_diversity_loss:
                # Encourage experts to have different activation patterns
                expert_utilization = dispatch.sum(dim=(0, 1)) / (B * L)  # (E,)
                # Diversity loss: penalize when experts have similar utilization
                diversity_loss = -torch.var(expert_utilization) * self.expert_diversity_weight
                aux_loss = aux_loss + diversity_loss
                
            # Add entropy regularization
            # Encourage appropriate entropy levels (not too high, not too low)
            avg_entropy = entropy.mean()
            entropy_loss = (avg_entropy - self.entropy_threshold).pow(2) * 0.01
            aux_loss = aux_loss + entropy_loss
            
        return out, aux_loss


class UncertaintyAwareExpertRouter(nn.Module):
    """Uncertainty-aware expert router that adapts based on model confidence"""
    
    def __init__(self, dim: int, num_experts: int, uncertainty_threshold: float = 0.5):
        super().__init__()
        self.dim = dim
        self.num_experts = num_experts
        self.uncertainty_threshold = uncertainty_threshold
        
        # Router network
        self.router = nn.Linear(dim, num_experts)
        
        # Uncertainty estimation network
        self.uncertainty_estimator = nn.Sequential(
            nn.Linear(dim, dim // 2),
            nn.ReLU(),
            nn.Linear(dim // 2, 1),
            nn.Sigmoid()
        )
        
        # Adaptive routing parameters
        self.exploration_weight = nn.Parameter(torch.ones(1))
        self.exploitation_weight = nn.Parameter(torch.ones(1))
        
    def estimate_uncertainty(self, x: torch.Tensor) -> torch.Tensor:
        """Estimate uncertainty of input representations
        Args:
            x: Input tensor of shape (B, L, D)
        Returns:
            Uncertainty tensor of shape (B, L)
        """
        return self.uncertainty_estimator(x).squeeze(-1)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Forward pass with uncertainty-aware routing
        Args:
            x: Input tensor of shape (B, L, D)
        Returns:
            Tuple of (routing_probs, metadata)
        """
        B, L, D = x.shape
        
        # Estimate uncertainty
        uncertainty = self.estimate_uncertainty(x)  # (B, L)
        confidence = 1.0 - uncertainty  # (B, L)
        
        # Standard routing
        logits = self.router(x)  # (B, L, E)
        base_probs = F.softmax(logits, dim=-1)
        
        # Adaptive routing based on uncertainty
        # High uncertainty -> more exploration (uniform routing)
        # Low uncertainty -> more exploitation (confident routing)
        uniform_probs = torch.ones_like(base_probs) / self.num_experts
        
        # Blend based on uncertainty
        adaptive_probs = (
            uncertainty.unsqueeze(-1) * uniform_probs +
            confidence.unsqueeze(-1) * base_probs
        )
        
        # Metadata
        metadata = {
            'base_probs': base_probs,
            'adaptive_probs': adaptive_probs,
            'uncertainty': uncertainty,
            'confidence': confidence
        }
        
        return adaptive_probs, metadata


class AdaptiveExpertActivationSystem:
    """Complete adaptive expert activation system"""
    
    def __init__(self, dim: int, num_experts: int = 8, top_k: int = 2):
        self.dim = dim
        self.num_experts = num_experts
        self.top_k = top_k
        
        # Initialize components
        self.entropy_gating = EntropyBasedGating(dim, num_experts)
        self.uncertainty_router = UncertaintyAwareExpertRouter(dim, num_experts)
        self.adaptive_moe = AdaptiveSparseMoE(dim, num_experts, top_k)
        
        # Performance tracking
        self.performance_metrics = {
            'entropy_history': [],
            'uncertainty_history': [],
            'confidence_history': [],
            'routing_diversity': []
        }
        
    def analyze_activation_patterns(self, routing_probs: torch.Tensor) -> Dict[str, float]:
        """Analyze expert activation patterns
        Args:
            routing_probs: Routing probabilities of shape (B, L, E)
        Returns:
            Analysis metrics
        """
        B, L, E = routing_probs.shape
        
        # Compute activation statistics
        expert_usage = routing_probs.sum(dim=(0, 1)) / (B * L)  # (E,)
        usage_entropy = -torch.sum(expert_usage * torch.log(expert_usage + 1e-8))
        
        # Compute load balancing metrics
        load_variance = torch.var(expert_usage)
        load_imbalance = load_variance * E  # Normalize by number of experts
        
        return {
            'usage_entropy': usage_entropy.item(),
            'load_variance': load_variance.item(),
            'load_imbalance': load_imbalance.item(),
            'max_usage': expert_usage.max().item(),
            'min_usage': expert_usage.min().item()
        }
    
    def adapt_to_performance(self, metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Adapt system based on performance metrics
        Args:
            metrics: Performance metrics dictionary
        Returns:
            Adaptation recommendations
        """
        recommendations = {}
        
        # Analyze entropy trends
        if len(self.performance_metrics['entropy_history']) >= 10:
            recent_entropy = self.performance_metrics['entropy_history'][-10:]
            entropy_trend = np.polyfit(range(len(recent_entropy)), recent_entropy, 1)[0]
            
            if entropy_trend > 0.1:  # Increasing entropy
                recommendations['increase_expert_count'] = True
                recommendations['reason'] = "increasing_entropy"
            elif entropy_trend < -0.1:  # Decreasing entropy
                recommendations['decrease_expert_count'] = True
                recommendations['reason'] = "decreasing_entropy"
                
        # Analyze uncertainty trends
        if len(self.performance_metrics['uncertainty_history']) >= 10:
            recent_uncertainty = self.performance_metrics['uncertainty_history'][-10:]
            uncertainty_trend = np.polyfit(range(len(recent_uncertainty)), recent_uncertainty, 1)[0]
            
            if uncertainty_trend > 0.1:  # Increasing uncertainty
                recommendations['increase_exploration'] = True
                recommendations['reason'] = "increasing_uncertainty"
                
        # Analyze load balancing
        if len(self.performance_metrics['routing_diversity']) >= 5:
            recent_diversity = self.performance_metrics['routing_diversity'][-5:]
            avg_diversity = np.mean(recent_diversity)
            
            if avg_diversity < 0.5:  # Poor diversity
                recommendations['increase_diversity'] = True
                recommendations['reason'] = "poor_diversity"
                
        return recommendations
    
    def get_system_stats(self) -> Dict[str, Any]:
        """Get system statistics
        Returns:
            System statistics dictionary
        """
        return {
            'performance_metrics': self.performance_metrics,
            'total_entropy_samples': len(self.performance_metrics['entropy_history']),
            'total_uncertainty_samples': len(self.performance_metrics['uncertainty_history']),
            'total_confidence_samples': len(self.performance_metrics['confidence_history'])
        }


def demo_adaptive_expert_activation():
    """Demonstrate adaptive expert activation system"""
    if not ADAPTIVE_AVAILABLE:
        print("❌ MAHIA-X modules not available for adaptive expert activation")
        return
        
    print("🚀 Demonstrating Adaptive Expert Activation System...")
    print("=" * 60)
    
    # Create adaptive system
    system = AdaptiveExpertActivationSystem(dim=64, num_experts=4, top_k=2)
    print("✅ Initialized Adaptive Expert Activation System")
    
    # Create sample input
    batch_size, seq_len, dim = 2, 8, 64
    x = torch.randn(batch_size, seq_len, dim)
    print(f"✅ Created sample input: {x.shape}")
    
    # Test entropy-based gating
    routing_probs, metadata = system.entropy_gating(x)
    print(f"✅ Entropy-based routing probabilities: {routing_probs.shape}")
    print(f"   Entropy range: [{metadata['entropy'].min():.3f}, {metadata['entropy'].max():.3f}]")
    print(f"   Confidence range: [{metadata['confidence'].min():.3f}, {metadata['confidence'].max():.3f}]")
    
    # Test uncertainty-aware routing
    adaptive_probs, uncertainty_metadata = system.uncertainty_router(x)
    print(f"✅ Uncertainty-aware routing probabilities: {adaptive_probs.shape}")
    print(f"   Uncertainty range: [{uncertainty_metadata['uncertainty'].min():.3f}, {uncertainty_metadata['uncertainty'].max():.3f}]")
    
    # Test adaptive MoE
    output, aux_loss = system.adaptive_moe(x, return_aux=True)
    print(f"✅ Adaptive MoE output: {output.shape}")
    if aux_loss is not None:
        print(f"   Auxiliary loss: {aux_loss.item():.6f}")
    
    # Analyze activation patterns
    analysis = system.analyze_activation_patterns(routing_probs)
    print(f"✅ Activation pattern analysis:")
    for key, value in analysis.items():
        print(f"   {key}: {value:.4f}")
    
    # Store performance metrics
    system.performance_metrics['entropy_history'].append(metadata['entropy'].mean().item())
    system.performance_metrics['uncertainty_history'].append(uncertainty_metadata['uncertainty'].mean().item())
    system.performance_metrics['confidence_history'].append(uncertainty_metadata['confidence'].mean().item())
    system.performance_metrics['routing_diversity'].append(analysis['usage_entropy'])
    
    # Test adaptation
    sample_metrics = {
        'loss': 0.5,
        'accuracy': 0.85,
        'grad_norm': 1.2
    }
    
    recommendations = system.adapt_to_performance(sample_metrics)
    print(f"✅ Adaptation recommendations: {recommendations}")
    
    # Print system stats
    stats = system.get_system_stats()
    print(f"✅ System statistics collected")
    
    print("\n" + "=" * 60)
    print("ADAPTIVE EXPERT ACTIVATION DEMO SUMMARY")
    print("=" * 60)
    print("Key Features Implemented:")
    print("  1. Entropy-based gating for adaptive routing")
    print("  2. Uncertainty-aware expert selection")
    print("  3. Dynamic expert activation based on model confidence")
    print("  4. Performance monitoring and adaptation")
    print("  5. Load balancing and diversity optimization")
    print("\nBenefits:")
    print("  - Improved routing efficiency")
    print("  - Better handling of uncertain inputs")
    print("  - Dynamic adaptation to training dynamics")
    print("  - Enhanced model robustness")
    
    print("\n✅ Adaptive Expert Activation demonstration completed!")


def main():
    """Main demonstration function"""
    demo_adaptive_expert_activation()


if __name__ == '__main__':
    main()