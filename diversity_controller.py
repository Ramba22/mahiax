"""
Diversity Controller for MAHIA Expert Engine
Manages expert diversity to prevent redundancy and improve coverage.
"""

import time
import threading
from typing import Dict, Any, Optional, List, Tuple
import math
from dataclasses import dataclass
from collections import deque

# Import expert registry
from expert_registry import get_expert_registry, ExpertMetadata

@dataclass
class DiversityMetrics:
    """Metrics for measuring expert diversity"""
    cosine_similarity_matrix: List[List[float]]
    output_overlap: float
    feature_entropy: float
    timestamp: float

class DiversityController:
    """Controller for managing expert diversity"""
    
    def __init__(self, similarity_threshold: float = 0.8, audit_interval: int = 60):
        self.expert_registry = get_expert_registry()
        self.similarity_threshold = similarity_threshold
        self.audit_interval = audit_interval
        self.metrics_history = deque(maxlen=100)  # Store recent metrics
        self.last_audit_time = 0
        self.lock = threading.RLock()
        
        print("🔄 DiversityController initialized")
    
    def compute_diversity_loss(self, expert_outputs: List[Any]) -> float:
        """
        Compute diversity loss based on expert outputs.
        
        Args:
            expert_outputs: List of outputs from different experts
            
        Returns:
            float: Diversity loss (lower is better)
        """
        if len(expert_outputs) < 2:
            return 0.0
            
        with self.lock:
            # Calculate pairwise similarities
            similarities = []
            for i in range(len(expert_outputs)):
                for j in range(i + 1, len(expert_outputs)):
                    similarity = self._calculate_output_similarity(
                        expert_outputs[i], expert_outputs[j]
                    )
                    similarities.append(similarity)
            
            # Average similarity as diversity loss
            if similarities:
                avg_similarity = sum(similarities) / len(similarities)
                return avg_similarity
            else:
                return 0.0
    
    def _calculate_output_similarity(self, output1: Any, output2: Any) -> float:
        """Calculate similarity between two expert outputs"""
        # This is a simplified implementation
        # In practice, this would depend on the output type
        
        # Handle different output types
        if isinstance(output1, list) and isinstance(output2, list):
            return self._cosine_similarity(output1, output2)
        elif isinstance(output1, str) and isinstance(output2, str):
            # Simple string similarity (Jaccard similarity of character sets)
            set1 = set(output1.lower())
            set2 = set(output2.lower())
            intersection = len(set1.intersection(set2))
            union = len(set1.union(set2))
            return intersection / union if union > 0 else 0.0
        else:
            # Default similarity
            return 0.5
    
    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
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
    
    def calculate_diversity_metrics(self) -> DiversityMetrics:
        """
        Calculate current diversity metrics across all active experts.
        
        Returns:
            DiversityMetrics: Current diversity metrics
        """
        with self.lock:
            experts = self.expert_registry.get_active_experts()
            
            if len(experts) < 2:
                return DiversityMetrics(
                    cosine_similarity_matrix=[[1.0]],
                    output_overlap=0.0,
                    feature_entropy=0.0,
                    timestamp=time.time()
                )
            
            # Calculate similarity matrix
            similarity_matrix = []
            for i, expert1 in enumerate(experts):
                row = []
                for j, expert2 in enumerate(experts):
                    if i == j:
                        row.append(1.0)
                    else:
                        similarity = self._calculate_embedding_similarity(
                            expert1.embedding_signature,
                            expert2.embedding_signature
                        )
                        row.append(similarity)
                similarity_matrix.append(row)
            
            # Calculate output overlap (simplified)
            avg_similarity = 0.0
            count = 0
            for i in range(len(similarity_matrix)):
                for j in range(i + 1, len(similarity_matrix)):
                    avg_similarity += similarity_matrix[i][j]
                    count += 1
            
            output_overlap = avg_similarity / count if count > 0 else 0.0
            
            # Calculate feature entropy (simplified)
            feature_entropy = self._calculate_feature_entropy(experts)
            
            metrics = DiversityMetrics(
                cosine_similarity_matrix=similarity_matrix,
                output_overlap=output_overlap,
                feature_entropy=feature_entropy,
                timestamp=time.time()
            )
            
            # Store in history
            self.metrics_history.append(metrics)
            
            return metrics
    
    def _calculate_embedding_similarity(self, emb1: List[float], emb2: List[float]) -> float:
        """Calculate similarity between expert embeddings"""
        return self._cosine_similarity(emb1, emb2)
    
    def _calculate_feature_entropy(self, experts: List[ExpertMetadata]) -> float:
        """Calculate feature entropy across experts"""
        if not experts:
            return 0.0
            
        # Count capability distribution
        capability_counts = {}
        total_capabilities = 0
        
        for expert in experts:
            for capability in expert.capabilities:
                capability_counts[capability] = capability_counts.get(capability, 0) + 1
                total_capabilities += 1
        
        if total_capabilities == 0:
            return 0.0
            
        # Calculate entropy
        entropy = 0.0
        for count in capability_counts.values():
            probability = count / total_capabilities
            if probability > 0:
                entropy -= probability * math.log2(probability)
        
        return entropy
    
    def periodic_audit(self) -> Dict[str, Any]:
        """
        Perform periodic audit to detect high redundancy.
        
        Returns:
            Dict[str, Any]: Audit results
        """
        current_time = time.time()
        
        # Check if it's time for audit
        if current_time - self.last_audit_time < self.audit_interval:
            return {"status": "skipped", "reason": "audit_interval_not_reached"}
        
        with self.lock:
            self.last_audit_time = current_time
            
            # Calculate current metrics
            metrics = self.calculate_diversity_metrics()
            
            # Detect high redundancy
            high_similarity_pairs = []
            experts = self.expert_registry.get_active_experts()
            
            for i in range(len(metrics.cosine_similarity_matrix)):
                for j in range(i + 1, len(metrics.cosine_similarity_matrix)):
                    similarity = metrics.cosine_similarity_matrix[i][j]
                    if similarity > self.similarity_threshold:
                        high_similarity_pairs.append({
                            "expert1": experts[i].expert_id,
                            "expert2": experts[j].expert_id,
                            "similarity": similarity
                        })
            
            audit_result = {
                "status": "completed",
                "timestamp": current_time,
                "metrics": {
                    "output_overlap": metrics.output_overlap,
                    "feature_entropy": metrics.feature_entropy,
                    "high_similarity_pairs": high_similarity_pairs,
                    "total_experts": len(experts)
                },
                "recommendations": []
            }
            
            # Generate recommendations
            if high_similarity_pairs:
                audit_result["recommendations"].append(
                    f"Found {len(high_similarity_pairs)} highly similar expert pairs"
                )
                audit_result["recommendations"].append(
                    "Consider merging or deactivating redundant experts"
                )
            
            if metrics.feature_entropy < 1.0:  # Low entropy threshold
                audit_result["recommendations"].append(
                    "Low feature entropy detected - consider adding experts with diverse capabilities"
                )
            
            print(f"🔍 Diversity audit completed: {len(high_similarity_pairs)} high similarity pairs found")
            
            return audit_result
    
    def get_diversity_report(self) -> Dict[str, Any]:
        """
        Generate a comprehensive diversity report.
        
        Returns:
            Dict[str, Any]: Diversity report
        """
        with self.lock:
            if not self.metrics_history:
                return {"status": "no_data", "message": "No diversity metrics available"}
            
            # Get latest metrics
            latest_metrics = self.metrics_history[-1]
            
            # Calculate trends
            if len(self.metrics_history) > 1:
                prev_metrics = self.metrics_history[-2]
                overlap_change = latest_metrics.output_overlap - prev_metrics.output_overlap
                entropy_change = latest_metrics.feature_entropy - prev_metrics.feature_entropy
            else:
                overlap_change = 0.0
                entropy_change = 0.0
            
            report = {
                "status": "success",
                "timestamp": latest_metrics.timestamp,
                "current_metrics": {
                    "output_overlap": latest_metrics.output_overlap,
                    "feature_entropy": latest_metrics.feature_entropy,
                    "overlap_trend": "increasing" if overlap_change > 0 else "decreasing" if overlap_change < 0 else "stable",
                    "entropy_trend": "increasing" if entropy_change > 0 else "decreasing" if entropy_change < 0 else "stable"
                },
                "historical_data": {
                    "total_measurements": len(self.metrics_history),
                    "avg_output_overlap": sum(m.output_overlap for m in self.metrics_history) / len(self.metrics_history),
                    "avg_feature_entropy": sum(m.feature_entropy for m in self.metrics_history) / len(self.metrics_history)
                }
            }
            
            return report
    
    def suggest_diversity_improvements(self) -> List[str]:
        """
        Suggest improvements to increase diversity.
        
        Returns:
            List[str]: List of improvement suggestions
        """
        suggestions = []
        
        with self.lock:
            if not self.metrics_history:
                return ["No data available for suggestions"]
            
            latest_metrics = self.metrics_history[-1]
            
            # Suggest based on current metrics
            if latest_metrics.output_overlap > 0.7:
                suggestions.append("High output overlap detected - consider adding experts with different specializations")
            
            if latest_metrics.feature_entropy < 1.5:
                suggestions.append("Low feature entropy - consider expanding capability coverage")
            
            # Check for recent audit results
            audit_result = self.periodic_audit()
            if "recommendations" in audit_result:
                suggestions.extend(audit_result["recommendations"])
            
            if not suggestions:
                suggestions.append("Diversity metrics are within acceptable ranges")
        
        return suggestions

# Global instance
_diversity_controller = None

def get_diversity_controller() -> DiversityController:
    """Get the global diversity controller instance"""
    global _diversity_controller
    if _diversity_controller is None:
        _diversity_controller = DiversityController()
    return _diversity_controller

if __name__ == "__main__":
    # Example usage
    controller = get_diversity_controller()
    
    # Simulate some expert outputs
    outputs = [
        [0.1, 0.2, 0.3, 0.4],
        [0.2, 0.3, 0.4, 0.5],
        [0.8, 0.7, 0.6, 0.5]
    ]
    
    # Compute diversity loss
    loss = controller.compute_diversity_loss(outputs)
    print(f"Diversity loss: {loss:.3f}")
    
    # Calculate diversity metrics
    metrics = controller.calculate_diversity_metrics()
    print(f"Output overlap: {metrics.output_overlap:.3f}")
    print(f"Feature entropy: {metrics.feature_entropy:.3f}")
    
    # Run periodic audit
    audit = controller.periodic_audit()
    print(f"Audit result: {audit}")
    
    # Get diversity report
    report = controller.get_diversity_report()
    print(f"Diversity report: {report}")
    
    # Get suggestions
    suggestions = controller.suggest_diversity_improvements()
    print(f"Improvement suggestions: {suggestions}")