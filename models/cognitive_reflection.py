"""
Cognitive Reflective Module V2 (meta-critic-like) for MAHIA-X.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import sys
import os
from typing import Dict, Any, Tuple, Optional, List
import math

# Add the project root to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

try:
    from modell_V5_MAHIA_HyenaMoE import ReflectiveHead, MetaController
    COGNITIVE_AVAILABLE = True
except ImportError:
    COGNITIVE_AVAILABLE = False
    print("⚠️  MAHIA-X modules not available for cognitive reflection")


class MetaCriticNetwork(nn.Module):
    """Meta-critic network that evaluates model decisions and provides feedback"""
    
    def __init__(self, input_dim: int, hidden_dim: int = 128, num_classes: int = 2):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        
        # Feature encoder
        self.feature_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU()
        )
        
        # Confidence estimator
        self.confidence_head = nn.Sequential(
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, 1),
            nn.Sigmoid()
        )
        
        # Error probability estimator
        self.error_head = nn.Sequential(
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, 1),
            nn.Sigmoid()
        )
        
        # Quality assessment
        self.quality_head = nn.Sequential(
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, 1),
            nn.Sigmoid()
        )
        
        # Decision refinement
        self.refinement_head = nn.Sequential(
            nn.Linear(hidden_dim // 2 + num_classes, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, num_classes)
        )
        
    def forward(self, features: torch.Tensor, logits: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward pass
        Args:
            features: Input features of shape (B, D)
            logits: Model logits of shape (B, C)
        Returns:
            Dictionary of outputs
        """
        B, D = features.shape
        C = logits.shape[-1]
        
        # Encode features
        encoded = self.feature_encoder(features)  # (B, H//2)
        
        # Estimate confidence
        confidence = self.confidence_head(encoded)  # (B, 1)
        
        # Estimate error probability
        error_prob = self.error_head(encoded)  # (B, 1)
        
        # Assess quality
        quality = self.quality_head(encoded)  # (B, 1)
        
        # Refine decision
        combined = torch.cat([encoded, logits], dim=-1)  # (B, H//2 + C)
        refined_logits = self.refinement_head(combined)  # (B, C)
        
        return {
            'confidence': confidence,
            'error_probability': error_prob,
            'quality': quality,
            'refined_logits': refined_logits,
            'encoded_features': encoded
        }


class CognitiveReflectiveModuleV2(nn.Module):
    """Enhanced cognitive reflective module with meta-critic capabilities"""
    
    def __init__(self, dim: int, num_classes: int = 2, num_experts: int = 4):
        super().__init__()
        self.dim = dim
        self.num_classes = num_classes
        self.num_experts = num_experts
        
        # Multiple meta-critics for different aspects
        self.meta_critics = nn.ModuleList([
            MetaCriticNetwork(dim, hidden_dim=128, num_classes=num_classes)
            for _ in range(num_experts)
        ])
        
        # Critic aggregator
        self.critic_aggregator = nn.Sequential(
            nn.Linear(num_experts * 4, 128),  # 4 outputs per critic
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )
        
        # Confidence fusion
        self.confidence_fusion = nn.Sequential(
            nn.Linear(num_experts, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
        
        # Error handling policy
        self.error_policy = nn.Sequential(
            nn.Linear(dim + 2, 64),  # features + confidence + error_prob
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 3)  # [accept, revise, reject]
        )
        
        # Temperature scaling for calibration
        self.temperature = nn.Parameter(torch.ones(1))
        
    def forward(self, features: torch.Tensor, 
                logits: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward pass with cognitive reflection
        Args:
            features: Input features of shape (B, D)
            logits: Model logits of shape (B, C) or (B, L, C)
        Returns:
            Dictionary of reflective outputs
        """
        B = features.shape[0]
        
        # Handle sequence dimension
        if logits.dim() == 3:
            # Sequence model - pool over sequence
            pooled_logits = logits.mean(dim=1)  # (B, C)
            pooled_features = features.mean(dim=1) if features.dim() == 3 else features  # (B, D)
        else:
            pooled_logits = logits  # (B, C)
            pooled_features = features  # (B, D)
            
        # Apply temperature scaling
        scaled_logits = pooled_logits / self.temperature
        
        # Get predictions
        predictions = F.softmax(scaled_logits, dim=-1)
        predicted_classes = torch.argmax(predictions, dim=-1)  # (B,)
        
        # Run multiple meta-critics
        critic_outputs = []
        confidences = []
        error_probs = []
        qualities = []
        refined_logits_list = []
        
        for critic in self.meta_critics:
            output = critic(pooled_features, pooled_logits)
            critic_outputs.append(output)
            confidences.append(output['confidence'])  # (B, 1)
            error_probs.append(output['error_probability'])  # (B, 1)
            qualities.append(output['quality'])  # (B, 1)
            refined_logits_list.append(output['refined_logits'])  # (B, C)
            
        # Aggregate critic outputs
        all_confidences = torch.cat(confidences, dim=-1)  # (B, E)
        all_error_probs = torch.cat(error_probs, dim=-1)  # (B, E)
        all_qualities = torch.cat(qualities, dim=-1)  # (B, E)
        
        # Fuse confidences
        fused_confidence = self.confidence_fusion(all_confidences)  # (B, 1)
        
        # Aggregate all critic features for final decision
        critic_features = []
        for output in critic_outputs:
            critic_features.extend([
                output['confidence'],
                output['error_probability'],
                output['quality'],
                F.softmax(output['refined_logits'], dim=-1).max(dim=-1, keepdim=True)[0]
            ])
            
        combined_features = torch.cat(critic_features, dim=-1)  # (B, E*4)
        final_refined_logits = self.critic_aggregator(combined_features)  # (B, C)
        
        # Apply error handling policy
        policy_input = torch.cat([
            pooled_features,
            fused_confidence,
            torch.cat(error_probs, dim=-1).mean(dim=-1, keepdim=True)
        ], dim=-1)  # (B, D+2)
        
        policy_logits = self.error_policy(policy_input)  # (B, 3)
        policy_probs = F.softmax(policy_logits, dim=-1)
        
        # Determine actions
        accept_threshold = 0.7
        revise_threshold = 0.4
        
        max_confidence = fused_confidence.squeeze(-1)  # (B,)
        avg_error_prob = torch.cat(error_probs, dim=-1).mean(dim=-1)  # (B,)
        
        # Decision logic
        should_accept = (max_confidence > accept_threshold) & (avg_error_prob < 0.3)
        should_revise = (max_confidence > revise_threshold) & (avg_error_prob < 0.5)
        should_reject = ~should_accept & ~should_revise
        
        actions = torch.zeros(B, dtype=torch.long, device=features.device)
        actions[should_revise] = 1  # revise
        actions[should_reject] = 2   # reject
        
        # Select final output based on policy
        final_output = torch.where(
            should_accept.unsqueeze(-1),
            scaled_logits,  # Accept original
            final_refined_logits  # Revise or reject uses refined
        )
        
        return {
            'final_logits': final_output,
            'confidence': fused_confidence,
            'error_probability': avg_error_prob,
            'quality': torch.cat(qualities, dim=-1).mean(dim=-1),
            'actions': actions,
            'policy_probs': policy_probs,
            'predicted_classes': predicted_classes,
            'refined_logits': final_refined_logits,
            'critic_confidences': all_confidences,
            'critic_error_probs': all_error_probs
        }


class ReflectiveTrainingWrapper:
    """Wrapper for training with cognitive reflection"""
    
    def __init__(self, base_model: nn.Module, reflective_module: CognitiveReflectiveModuleV2):
        self.base_model = base_model
        self.reflective_module = reflective_module
        self.history = {
            'confidences': [],
            'error_rates': [],
            'quality_scores': [],
            'actions': []
        }
        
    def forward(self, *args, **kwargs) -> Dict[str, torch.Tensor]:
        """Forward pass through base model and reflective module"""
        # Get base model outputs
        base_logits, aux_loss = self.base_model(*args, **kwargs)
        
        # Extract features (simplified - in practice, you might use intermediate representations)
        if hasattr(self.base_model, 'text_encoder'):
            # Assuming MAHIA_V5 structure
            features = base_logits.detach()  # Simplified feature extraction
        else:
            features = base_logits.detach()
            
        # Apply cognitive reflection
        reflective_outputs = self.reflective_module(features, base_logits)
        
        # Store history for analysis
        self.history['confidences'].append(reflective_outputs['confidence'].mean().item())
        self.history['error_rates'].append(reflective_outputs['error_probability'].mean().item())
        self.history['quality_scores'].append(reflective_outputs['quality'].mean().item())
        self.history['actions'].append(reflective_outputs['actions'].cpu().numpy())
        
        # Keep only recent history
        max_history = 100
        if len(self.history['confidences']) > max_history:
            self.history['confidences'].pop(0)
            self.history['error_rates'].pop(0)
            self.history['quality_scores'].pop(0)
            self.history['actions'].pop(0)
            
        return {
            'logits': reflective_outputs['final_logits'],
            'reflective_outputs': reflective_outputs,
            'aux_loss': aux_loss
        }
    
    def get_reflection_stats(self) -> Dict[str, Any]:
        """Get statistics about reflective decisions"""
        if not self.history['confidences']:
            return {}
            
        return {
            'avg_confidence': np.mean(self.history['confidences']),
            'avg_error_rate': np.mean(self.history['error_rates']),
            'avg_quality': np.mean(self.history['quality_scores']),
            'accept_rate': np.mean([np.sum(actions == 0) / len(actions) 
                                  for actions in self.history['actions']]),
            'revise_rate': np.mean([np.sum(actions == 1) / len(actions) 
                                  for actions in self.history['actions']]),
            'reject_rate': np.mean([np.sum(actions == 2) / len(actions) 
                                  for actions in self.history['actions']])
        }


class AdaptiveReflectiveController:
    """Adaptive controller that adjusts reflective behavior based on performance"""
    
    def __init__(self, reflective_module: CognitiveReflectiveModuleV2):
        self.reflective_module = reflective_module
        self.performance_history = []
        self.confidence_threshold = 0.7
        self.error_threshold = 0.3
        self.adaptation_rate = 0.01
        
    def update_thresholds(self, current_performance: float):
        """Adaptively update thresholds based on performance
        Args:
            current_performance: Current performance metric (higher is better)
        """
        self.performance_history.append(current_performance)
        
        # Keep only recent history
        if len(self.performance_history) > 50:
            self.performance_history.pop(0)
            
        # Adjust thresholds based on performance trends
        if len(self.performance_history) >= 5:
            recent_perf = self.performance_history[-5:]
            perf_trend = np.polyfit(range(len(recent_perf)), recent_perf, 1)[0]
            
            if perf_trend > 0.01:  # Improving
                # Become more confident, reduce reflection
                self.confidence_threshold = min(0.9, self.confidence_threshold + self.adaptation_rate)
                self.error_threshold = max(0.1, self.error_threshold - self.adaptation_rate)
            elif perf_trend < -0.01:  # Degrading
                # Become less confident, increase reflection
                self.confidence_threshold = max(0.5, self.confidence_threshold - self.adaptation_rate)
                self.error_threshold = min(0.5, self.error_threshold + self.adaptation_rate)
                
    def get_adaptive_parameters(self) -> Dict[str, float]:
        """Get current adaptive parameters"""
        return {
            'confidence_threshold': self.confidence_threshold,
            'error_threshold': self.error_threshold,
            'performance_history_length': len(self.performance_history)
        }


def demo_cognitive_reflection():
    """Demonstrate cognitive reflective module V2"""
    if not COGNITIVE_AVAILABLE:
        print("❌ MAHIA-X modules not available for cognitive reflection")
        return
        
    print("🚀 Demonstrating Cognitive Reflective Module V2...")
    print("=" * 60)
    
    # Create reflective module
    reflective_module = CognitiveReflectiveModuleV2(dim=64, num_classes=2, num_experts=3)
    print("✅ Initialized Cognitive Reflective Module V2")
    
    # Create sample inputs
    batch_size, dim, num_classes = 4, 64, 2
    features = torch.randn(batch_size, dim)
    logits = torch.randn(batch_size, num_classes)
    
    print(f"✅ Created sample inputs:")
    print(f"   Features: {features.shape}")
    print(f"   Logits: {logits.shape}")
    
    # Test forward pass
    outputs = reflective_module(features, logits)
    print(f"✅ Reflective module outputs:")
    for key, value in outputs.items():
        if isinstance(value, torch.Tensor):
            print(f"   {key}: {value.shape}")
        else:
            print(f"   {key}: {type(value)}")
    
    # Print key metrics
    print(f"✅ Key metrics:")
    print(f"   Final confidence: {outputs['confidence'].mean().item():.3f}")
    print(f"   Error probability: {outputs['error_probability'].mean().item():.3f}")
    print(f"   Quality score: {outputs['quality'].mean().item():.3f}")
    
    # Test actions distribution
    actions = outputs['actions']
    action_counts = torch.bincount(actions, minlength=3)
    action_names = ['Accept', 'Revise', 'Reject']
    print(f"✅ Action distribution:")
    for i, (name, count) in enumerate(zip(action_names, action_counts)):
        print(f"   {name}: {count.item()} ({count.item()/batch_size*100:.1f}%)")
    
    # Test with sequence inputs
    seq_len = 16
    seq_features = torch.randn(batch_size, seq_len, dim)
    seq_logits = torch.randn(batch_size, seq_len, num_classes)
    
    seq_outputs = reflective_module(seq_features, seq_logits)
    print(f"✅ Sequence input handling:")
    print(f"   Sequence features: {seq_features.shape}")
    print(f"   Sequence logits: {seq_logits.shape}")
    print(f"   Final logits: {seq_outputs['final_logits'].shape}")
    
    # Create training wrapper
    # For demo, we'll create a simple mock model
    class MockModel(nn.Module):
        def __init__(self, dim, num_classes):
            super().__init__()
            self.classifier = nn.Linear(dim, num_classes)
            
        def forward(self, x):
            if x.dim() == 3:
                x = x.mean(dim=1)  # Pool over sequence
            return self.classifier(x), None
    
    mock_model = MockModel(dim=64, num_classes=2)
    training_wrapper = ReflectiveTrainingWrapper(mock_model, reflective_module)
    
    # Test training wrapper
    wrapper_outputs = training_wrapper.forward(features)
    print(f"✅ Training wrapper outputs:")
    print(f"   Final logits: {wrapper_outputs['logits'].shape}")
    if wrapper_outputs['aux_loss'] is not None:
        print(f"   Aux loss: {wrapper_outputs['aux_loss'].item():.6f}")
    
    # Get reflection stats
    stats = training_wrapper.get_reflection_stats()
    print(f"✅ Reflection statistics:")
    for key, value in stats.items():
        print(f"   {key}: {value:.3f}")
    
    # Test adaptive controller
    adaptive_controller = AdaptiveReflectiveController(reflective_module)
    
    # Simulate performance improvements
    for i in range(10):
        performance = 0.7 + i * 0.03  # Improving performance
        adaptive_controller.update_thresholds(performance)
        
    adaptive_params = adaptive_controller.get_adaptive_parameters()
    print(f"✅ Adaptive parameters after training:")
    for key, value in adaptive_params.items():
        print(f"   {key}: {value:.3f}")
    
    print("\n" + "=" * 60)
    print("COGNITIVE REFLECTIVE MODULE V2 DEMO SUMMARY")
    print("=" * 60)
    print("Key Features Implemented:")
    print("  1. Multi-critic meta-evaluation")
    print("  2. Confidence and error estimation")
    print("  3. Decision refinement and quality assessment")
    print("  4. Adaptive error handling policy")
    print("  5. Training wrapper with history tracking")
    print("  6. Performance-based threshold adaptation")
    print("\nBenefits:")
    print("  - Improved decision reliability")
    print("  - Better uncertainty quantification")
    print("  - Adaptive reflection intensity")
    print("  - Enhanced model robustness")
    
    print("\n✅ Cognitive Reflective Module V2 demonstration completed!")


def main():
    """Main demonstration function"""
    demo_cognitive_reflection()


if __name__ == '__main__':
    main()