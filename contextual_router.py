"""
Contextual Router for MAHIA Expert Engine
Implements routing strategies for selecting experts based on context.
"""

import time
import threading
from typing import Dict, Any, Optional, List, Tuple, Union
from dataclasses import dataclass
from enum import Enum
import math

# Import expert registry
from expert_registry import get_expert_registry, ExpertMetadata

class RoutingMode(Enum):
    """Routing modes"""
    TOP_K = "top_k"
    REFLECTIVE = "reflective"
    ADAPTIVE = "adaptive"
    ENSEMBLE = "ensemble"

@dataclass
class RoutingDecision:
    """Represents a routing decision with explainability"""
    expert_ids: List[str]
    scores: List[float]
    mode: RoutingMode
    confidence: float
    diversity_penalty: float
    freshness_factor: float
    explanation: Dict[str, Any]

class ContextualRouter:
    """Contextual router for expert selection"""
    
    def __init__(self, diversity_threshold: float = 0.3, freshness_window: int = 100):
        self.expert_registry = get_expert_registry()
        self.diversity_threshold = diversity_threshold
        self.freshness_window = freshness_window
        self.routing_history = []  # Store recent routing decisions
        self.lock = threading.RLock()
        
        # Scoring weights
        self.confidence_weight = 0.5
        self.diversity_weight = 0.3
        self.freshness_weight = 0.2
        
        print("🧭 ContextualRouter initialized")
    
    def route(self, inputs: Any, k: int = 3, mode: Union[RoutingMode, str] = RoutingMode.TOP_K,
              required_capabilities: Optional[List[str]] = None) -> RoutingDecision:
        """
        Route inputs to the most suitable experts.
        
        Args:
            inputs: Input data or context vector
            k: Number of experts to select
            mode: Routing mode
            required_capabilities: Required expert capabilities
            
        Returns:
            RoutingDecision: Routing decision with explainability
        """
        if isinstance(mode, str):
            mode = RoutingMode(mode)
            
        with self.lock:
            if mode == RoutingMode.TOP_K:
                return self._top_k_routing(inputs, k, required_capabilities)
            elif mode == RoutingMode.REFLECTIVE:
                return self._reflective_routing(inputs, k, required_capabilities)
            elif mode == RoutingMode.ADAPTIVE:
                return self._adaptive_gating(inputs, k, required_capabilities)
            elif mode == RoutingMode.ENSEMBLE:
                return self._ensemble_routing(inputs, k, required_capabilities)
            else:
                # Default to TOP_K
                return self._top_k_routing(inputs, k, required_capabilities)
    
    def _top_k_routing(self, inputs: Any, k: int, 
                      required_capabilities: Optional[List[str]]) -> RoutingDecision:
        """Top-K Softmax routing strategy"""
        # Convert inputs to context vector if needed
        context_vector = self._extract_context_vector(inputs)
        
        # Get candidate experts
        candidates = self.expert_registry.get_active_experts()
        
        # Filter by capabilities if specified
        if required_capabilities:
            candidates = [
                expert for expert in candidates
                if all(cap in expert.capabilities for cap in required_capabilities)
            ]
        
        if not candidates:
            return RoutingDecision(
                expert_ids=[],
                scores=[],
                mode=RoutingMode.TOP_K,
                confidence=0.0,
                diversity_penalty=0.0,
                freshness_factor=0.0,
                explanation={"error": "No suitable experts found"}
            )
        
        # Score experts based on similarity
        expert_scores = []
        for expert in candidates:
            score = self._calculate_similarity(context_vector, expert.embedding_signature)
            expert_scores.append((expert.expert_id, score, expert))
        
        # Sort by score and take top-k
        expert_scores.sort(key=lambda x: x[1], reverse=True)
        top_k = expert_scores[:k]
        
        # Extract results
        expert_ids = [item[0] for item in top_k]
        scores = [item[1] for item in top_k]
        
        # Calculate metrics
        confidence = self._calculate_confidence(scores)
        diversity_penalty = self._calculate_diversity_penalty([item[2] for item in top_k])
        freshness_factor = self._calculate_freshness_factor([item[0] for item in top_k])
        
        # Create explanation
        explanation = {
            "method": "Top-K Softmax",
            "total_candidates": len(candidates),
            "selected_experts": len(top_k),
            "scoring_details": [
                {
                    "expert_id": item[0],
                    "similarity_score": item[1],
                    "capabilities": item[2].capabilities
                }
                for item in top_k
            ]
        }
        
        decision = RoutingDecision(
            expert_ids=expert_ids,
            scores=scores,
            mode=RoutingMode.TOP_K,
            confidence=confidence,
            diversity_penalty=diversity_penalty,
            freshness_factor=freshness_factor,
            explanation=explanation
        )
        
        # Store in history
        self._store_routing_decision(decision)
        
        return decision
    
    def _reflective_routing(self, inputs: Any, k: int,
                           required_capabilities: Optional[List[str]]) -> RoutingDecision:
        """Reflective routing based on semantic distance"""
        # Convert inputs to context vector
        context_vector = self._extract_context_vector(inputs)
        
        # Get candidate experts
        candidates = self.expert_registry.get_active_experts()
        
        # Filter by capabilities if specified
        if required_capabilities:
            candidates = [
                expert for expert in candidates
                if all(cap in expert.capabilities for cap in required_capabilities)
            ]
        
        if not candidates:
            return RoutingDecision(
                expert_ids=[],
                scores=[],
                mode=RoutingMode.REFLECTIVE,
                confidence=0.0,
                diversity_penalty=0.0,
                freshness_factor=0.0,
                explanation={"error": "No suitable experts found"}
            )
        
        # Calculate reflective scores
        expert_scores = []
        for expert in candidates:
            # Reflective score considers both similarity and expert specialization
            similarity = self._calculate_similarity(context_vector, expert.embedding_signature)
            specialization = len(expert.capabilities)  # Simple specialization measure
            reflective_score = similarity * (1 + 0.1 * specialization)
            expert_scores.append((expert.expert_id, reflective_score, expert))
        
        # Sort by reflective score and take top-k
        expert_scores.sort(key=lambda x: x[1], reverse=True)
        top_k = expert_scores[:k]
        
        # Extract results
        expert_ids = [item[0] for item in top_k]
        scores = [item[1] for item in top_k]
        
        # Calculate metrics
        confidence = self._calculate_confidence(scores)
        diversity_penalty = self._calculate_diversity_penalty([item[2] for item in top_k])
        freshness_factor = self._calculate_freshness_factor([item[0] for item in top_k])
        
        # Create explanation
        explanation = {
            "method": "Reflective Routing",
            "total_candidates": len(candidates),
            "selected_experts": len(top_k),
            "scoring_details": [
                {
                    "expert_id": item[0],
                    "reflective_score": item[1],
                    "similarity": self._calculate_similarity(context_vector, item[2].embedding_signature),
                    "specialization": len(item[2].capabilities)
                }
                for item in top_k
            ]
        }
        
        decision = RoutingDecision(
            expert_ids=expert_ids,
            scores=scores,
            mode=RoutingMode.REFLECTIVE,
            confidence=confidence,
            diversity_penalty=diversity_penalty,
            freshness_factor=freshness_factor,
            explanation=explanation
        )
        
        # Store in history
        self._store_routing_decision(decision)
        
        return decision
    
    def _adaptive_gating(self, inputs: Any, k: int,
                        required_capabilities: Optional[List[str]]) -> RoutingDecision:
        """Adaptive gating routing strategy"""
        # This would typically involve a trainable gating network
        # For now, we'll implement a simplified version based on historical performance
        
        # Convert inputs to context vector
        context_vector = self._extract_context_vector(inputs)
        
        # Get candidate experts
        candidates = self.expert_registry.get_active_experts()
        
        # Filter by capabilities if specified
        if required_capabilities:
            candidates = [
                expert for expert in candidates
                if all(cap in expert.capabilities for cap in required_capabilities)
            ]
        
        if not candidates:
            return RoutingDecision(
                expert_ids=[],
                scores=[],
                mode=RoutingMode.ADAPTIVE,
                confidence=0.0,
                diversity_penalty=0.0,
                freshness_factor=0.0,
                explanation={"error": "No suitable experts found"}
            )
        
        # Calculate adaptive scores based on context and historical performance
        expert_scores = []
        for expert in candidates:
            # Base similarity score
            similarity = self._calculate_similarity(context_vector, expert.embedding_signature)
            
            # Historical performance factor (simplified)
            historical_factor = self._get_historical_performance(expert.expert_id)
            
            # Adaptive score combines similarity and performance
            adaptive_score = similarity * historical_factor
            expert_scores.append((expert.expert_id, adaptive_score, expert))
        
        # Sort by adaptive score and take top-k
        expert_scores.sort(key=lambda x: x[1], reverse=True)
        top_k = expert_scores[:k]
        
        # Extract results
        expert_ids = [item[0] for item in top_k]
        scores = [item[1] for item in top_k]
        
        # Calculate metrics
        confidence = self._calculate_confidence(scores)
        diversity_penalty = self._calculate_diversity_penalty([item[2] for item in top_k])
        freshness_factor = self._calculate_freshness_factor([item[0] for item in top_k])
        
        # Create explanation
        explanation = {
            "method": "Adaptive Gating",
            "total_candidates": len(candidates),
            "selected_experts": len(top_k),
            "scoring_details": [
                {
                    "expert_id": item[0],
                    "adaptive_score": item[1],
                    "similarity": self._calculate_similarity(context_vector, item[2].embedding_signature),
                    "historical_factor": self._get_historical_performance(item[0])
                }
                for item in top_k
            ]
        }
        
        decision = RoutingDecision(
            expert_ids=expert_ids,
            scores=scores,
            mode=RoutingMode.ADAPTIVE,
            confidence=confidence,
            diversity_penalty=diversity_penalty,
            freshness_factor=freshness_factor,
            explanation=explanation
        )
        
        # Store in history
        self._store_routing_decision(decision)
        
        return decision
    
    def _ensemble_routing(self, inputs: Any, k: int,
                         required_capabilities: Optional[List[str]]) -> RoutingDecision:
        """Ensemble routing combining multiple strategies"""
        # Get decisions from different routing strategies
        top_k_decision = self._top_k_routing(inputs, k, required_capabilities)
        reflective_decision = self._reflective_routing(inputs, k, required_capabilities)
        adaptive_decision = self._adaptive_gating(inputs, k, required_capabilities)
        
        # Combine scores (simple averaging)
        combined_scores = {}
        
        # Add scores from each strategy
        for i, expert_id in enumerate(top_k_decision.expert_ids):
            combined_scores[expert_id] = combined_scores.get(expert_id, 0) + top_k_decision.scores[i]
            
        for i, expert_id in enumerate(reflective_decision.expert_ids):
            combined_scores[expert_id] = combined_scores.get(expert_id, 0) + reflective_decision.scores[i]
            
        for i, expert_id in enumerate(adaptive_decision.expert_ids):
            combined_scores[expert_id] = combined_scores.get(expert_id, 0) + adaptive_decision.scores[i]
        
        # Average the scores
        for expert_id in combined_scores:
            combined_scores[expert_id] /= 3.0
        
        # Sort by combined score and take top-k
        sorted_experts = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)
        top_k_experts = sorted_experts[:k]
        
        # Extract results
        expert_ids = [item[0] for item in top_k_experts]
        scores = [item[1] for item in top_k_experts]
        
        # Calculate metrics (use average of individual metrics)
        confidence = (top_k_decision.confidence + reflective_decision.confidence + adaptive_decision.confidence) / 3
        diversity_penalty = (top_k_decision.diversity_penalty + reflective_decision.diversity_penalty + adaptive_decision.diversity_penalty) / 3
        freshness_factor = (top_k_decision.freshness_factor + reflective_decision.freshness_factor + adaptive_decision.freshness_factor) / 3
        
        # Create explanation
        explanation = {
            "method": "Ensemble Routing",
            "strategies_combined": ["Top-K", "Reflective", "Adaptive"],
            "individual_decisions": {
                "top_k": {
                    "expert_ids": top_k_decision.expert_ids,
                    "scores": top_k_decision.scores
                },
                "reflective": {
                    "expert_ids": reflective_decision.expert_ids,
                    "scores": reflective_decision.scores
                },
                "adaptive": {
                    "expert_ids": adaptive_decision.expert_ids,
                    "scores": adaptive_decision.scores
                }
            },
            "combined_scores": dict(top_k_experts)
        }
        
        decision = RoutingDecision(
            expert_ids=expert_ids,
            scores=scores,
            mode=RoutingMode.ENSEMBLE,
            confidence=confidence,
            diversity_penalty=diversity_penalty,
            freshness_factor=freshness_factor,
            explanation=explanation
        )
        
        # Store in history
        self._store_routing_decision(decision)
        
        return decision
    
    def _extract_context_vector(self, inputs: Any) -> List[float]:
        """Extract or convert inputs to context vector"""
        # This is a simplified implementation
        # In practice, this would involve embedding models or feature extraction
        if isinstance(inputs, list):
            return inputs
        elif isinstance(inputs, str):
            # Simple hash-based vector for string inputs
            hash_values = [hash(inputs[i % len(inputs)]) % 1000 / 1000.0 for i in range(10)]
            return hash_values
        else:
            # Default vector
            return [0.1, 0.2, 0.3, 0.4, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0]
    
    def _calculate_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Calculate cosine similarity between two vectors"""
        if len(vec1) != len(vec2):
            return 0.0
            
        # Dot product
        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        
        # Magnitudes
        mag1 = math.sqrt(sum(a * a for a in vec1))
        mag2 = math.sqrt(sum(b * b for b in vec2))
        
        # Cosine similarity
        if mag1 == 0 or mag2 == 0:
            return 0.0
            
        return dot_product / (mag1 * mag2)
    
    def _calculate_confidence(self, scores: List[float]) -> float:
        """Calculate confidence based on score distribution"""
        if not scores:
            return 0.0
            
        # Simple confidence measure: max score normalized
        max_score = max(scores)
        return max(0.0, min(1.0, max_score))
    
    def _calculate_diversity_penalty(self, experts: List[ExpertMetadata]) -> float:
        """Calculate diversity penalty based on expert similarity"""
        if len(experts) < 2:
            return 0.0
            
        # Calculate pairwise similarities
        similarities = []
        for i in range(len(experts)):
            for j in range(i + 1, len(experts)):
                sim = self._calculate_similarity(
                    experts[i].embedding_signature,
                    experts[j].embedding_signature
                )
                similarities.append(sim)
        
        # Average similarity as penalty
        avg_similarity = sum(similarities) / len(similarities) if similarities else 0.0
        return avg_similarity
    
    def _calculate_freshness_factor(self, expert_ids: List[str]) -> float:
        """Calculate freshness factor based on recent usage"""
        if not self.routing_history:
            return 1.0
            
        # Count recent usage of these experts
        recent_history = self.routing_history[-self.freshness_window:] if len(self.routing_history) > self.freshness_window else self.routing_history
        
        usage_count = 0
        for decision in recent_history:
            for expert_id in expert_ids:
                if expert_id in decision.expert_ids:
                    usage_count += 1
        
        # Freshness factor decreases with usage (0.5 to 1.0)
        max_usage = len(recent_history) * len(expert_ids)
        if max_usage == 0:
            return 1.0
            
        usage_ratio = usage_count / max_usage
        return 1.0 - (usage_ratio * 0.5)
    
    def _get_historical_performance(self, expert_id: str) -> float:
        """Get historical performance factor for an expert"""
        # This is a simplified implementation
        # In practice, this would track actual performance metrics
        return 1.0
    
    def _store_routing_decision(self, decision: RoutingDecision):
        """Store routing decision in history"""
        self.routing_history.append(decision)
        # Keep only recent history
        if len(self.routing_history) > 1000:
            self.routing_history = self.routing_history[-1000:]
    
    def get_routing_explanation(self, decision: RoutingDecision) -> str:
        """Generate human-readable explanation for routing decision"""
        explanation = decision.explanation
        method = explanation.get("method", "Unknown")
        
        result = f"Routing Decision (Method: {method})\n"
        result += f"Selected Experts: {', '.join(decision.expert_ids)}\n"
        result += f"Confidence: {decision.confidence:.3f}\n"
        result += f"Diversity Penalty: {decision.diversity_penalty:.3f}\n"
        result += f"Freshness Factor: {decision.freshness_factor:.3f}\n"
        
        if "scoring_details" in explanation:
            result += "Scoring Details:\n"
            for detail in explanation["scoring_details"]:
                result += f"  - {detail['expert_id']}: {detail.get('similarity_score', detail.get('reflective_score', 0.0)):.3f}\n"
        
        return result

# Global instance
_contextual_router = None

def get_contextual_router() -> ContextualRouter:
    """Get the global contextual router instance"""
    global _contextual_router
    if _contextual_router is None:
        _contextual_router = ContextualRouter()
    return _contextual_router

if __name__ == "__main__":
    # Example usage
    router = get_contextual_router()
    
    # Create some test inputs
    test_inputs = [0.1, 0.2, 0.3, 0.4, 0.5]
    
    # Test different routing modes
    top_k_result = router.route(test_inputs, k=2, mode=RoutingMode.TOP_K)
    print("Top-K Routing:")
    print(router.get_routing_explanation(top_k_result))
    
    reflective_result = router.route(test_inputs, k=2, mode=RoutingMode.REFLECTIVE)
    print("\nReflective Routing:")
    print(router.get_routing_explanation(reflective_result))
    
    adaptive_result = router.route(test_inputs, k=2, mode=RoutingMode.ADAPTIVE)
    print("\nAdaptive Gating:")
    print(router.get_routing_explanation(adaptive_result))