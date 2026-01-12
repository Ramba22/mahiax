"""
Evolution Module for MAHIA Expert Engine
Handles expert evolution through split/merge operations based on performance metrics.
"""

import time
import threading
from typing import Dict, Any, Optional, List, Tuple, Callable
from dataclasses import dataclass
from enum import Enum
import math
import copy

# Import expert registry and related components
from expert_registry import get_expert_registry, ExpertMetadata, ExpertStatus

class EvolutionType(Enum):
    """Types of evolution operations"""
    SPLIT = "split"
    MERGE = "merge"
    MUTATE = "mutate"

@dataclass
class EvolutionCriteria:
    """Criteria for triggering evolution operations"""
    gradient_variance_threshold: float = 0.5
    error_cluster_threshold: float = 0.3
    similarity_threshold: float = 0.9
    concept_drift_threshold: float = 0.4

@dataclass
class EvolutionEvent:
    """Record of an evolution event"""
    event_id: str
    evolution_type: EvolutionType
    expert_ids: List[str]
    timestamp: float
    criteria_met: Dict[str, float]
    success: bool
    details: Dict[str, Any]

class EvolutionModule:
    """Module for expert evolution through split/merge operations"""
    
    def __init__(self, criteria: Optional[EvolutionCriteria] = None):
        self.expert_registry = get_expert_registry()
        self.criteria = criteria or EvolutionCriteria()
        self.evolution_history = []
        self.simulation_cache = {}
        self.lock = threading.RLock()
        
        print("🧬 EvolutionModule initialized")
    
    def check_split_criteria(self, expert_id: str, 
                           gradient_variances: List[float],
                           error_clusters: List[List[float]]) -> bool:
        """
        Check if expert should be split based on criteria.
        
        Args:
            expert_id: ID of the expert to check
            gradient_variances: List of gradient variances
            error_clusters: List of error clusters
            
        Returns:
            bool: True if split criteria are met
        """
        with self.lock:
            # Check gradient variance
            if gradient_variances:
                avg_variance = sum(gradient_variances) / len(gradient_variances)
                if avg_variance > self.criteria.gradient_variance_threshold:
                    print(f"📈 High gradient variance detected for expert {expert_id}: {avg_variance:.3f}")
                    return True
            
            # Check error clusters
            if error_clusters and len(error_clusters) > 1:
                # Simple check for multimodal error distribution
                cluster_distances = []
                for i in range(len(error_clusters)):
                    for j in range(i + 1, len(error_clusters)):
                        # Calculate distance between cluster centers
                        if error_clusters[i] and error_clusters[j]:
                            distance = self._calculate_vector_distance(
                                error_clusters[i], error_clusters[j]
                            )
                            cluster_distances.append(distance)
                
                if cluster_distances:
                    avg_distance = sum(cluster_distances) / len(cluster_distances)
                    if avg_distance > self.criteria.error_cluster_threshold:
                        print(f"🔄 Multimodal error clusters detected for expert {expert_id}: {avg_distance:.3f}")
                        return True
            
            return False
    
    def check_merge_criteria(self, expert_id_1: str, expert_id_2: str,
                           similarity_score: float) -> bool:
        """
        Check if two experts should be merged based on criteria.
        
        Args:
            expert_id_1: ID of first expert
            expert_id_2: ID of second expert
            similarity_score: Similarity score between experts
            
        Returns:
            bool: True if merge criteria are met
        """
        with self.lock:
            # Check similarity threshold
            if similarity_score > self.criteria.similarity_threshold:
                print(f"🔗 High similarity between experts {expert_id_1} and {expert_id_2}: {similarity_score:.3f}")
                return True
            
            return False
    
    def simulate_evolution(self, expert_id: str, evolution_type: EvolutionType,
                          simulation_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Simulate evolution operation offline before applying.
        
        Args:
            expert_id: ID of the expert to evolve
            evolution_type: Type of evolution operation
            simulation_data: Data for simulation
            
        Returns:
            Dict[str, Any]: Simulation results
        """
        with self.lock:
            # Create cache key
            cache_key = f"{expert_id}_{evolution_type.value}_{hash(str(simulation_data))}"
            
            # Check cache
            if cache_key in self.simulation_cache:
                print(f"キャッシング Simulation result found in cache")
                return self.simulation_cache[cache_key]
            
            # Get expert metadata
            expert = self.expert_registry.get_expert(expert_id)
            if not expert:
                result = {
                    "success": False,
                    "error": f"Expert {expert_id} not found"
                }
                self.simulation_cache[cache_key] = result
                return result
            
            # Perform simulation based on evolution type
            if evolution_type == EvolutionType.SPLIT:
                result = self._simulate_split(expert, simulation_data)
            elif evolution_type == EvolutionType.MERGE:
                result = self._simulate_merge(expert, simulation_data)
            elif evolution_type == EvolutionType.MUTATE:
                result = self._simulate_mutate(expert, simulation_data)
            else:
                result = {
                    "success": False,
                    "error": f"Unknown evolution type: {evolution_type}"
                }
            
            # Cache result
            self.simulation_cache[cache_key] = result
            
            # Keep cache size manageable
            if len(self.simulation_cache) > 100:
                # Remove oldest entries
                oldest_keys = list(self.simulation_cache.keys())[:10]
                for key in oldest_keys:
                    del self.simulation_cache[key]
            
            return result
    
    def _simulate_split(self, expert: ExpertMetadata, 
                       simulation_data: Dict[str, Any]) -> Dict[str, Any]:
        """Simulate expert split operation"""
        try:
            # Extract simulation parameters
            gradient_variances = simulation_data.get("gradient_variances", [])
            error_clusters = simulation_data.get("error_clusters", [])
            
            # Calculate split feasibility
            feasibility_score = 0.0
            
            if gradient_variances:
                avg_variance = sum(gradient_variances) / len(gradient_variances)
                feasibility_score += min(1.0, avg_variance / self.criteria.gradient_variance_threshold)
            
            if error_clusters and len(error_clusters) > 1:
                feasibility_score += 0.5  # Base score for multimodal errors
            
            # Estimate resource requirements
            estimated_memory_increase = expert.memory_footprint_mb * 0.8  # Rough estimate
            
            result = {
                "success": True,
                "operation": "split",
                "feasibility_score": feasibility_score,
                "estimated_new_experts": 2,
                "estimated_memory_increase_mb": estimated_memory_increase,
                "risk_factors": [
                    "Increased memory usage",
                    "Potential training instability",
                    "Coordination overhead"
                ]
            }
            
            return result
            
        except Exception as e:
            return {
                "success": False,
                "error": f"Split simulation failed: {str(e)}"
            }
    
    def _simulate_merge(self, expert: ExpertMetadata,
                       simulation_data: Dict[str, Any]) -> Dict[str, Any]:
        """Simulate expert merge operation"""
        try:
            # Extract simulation parameters
            target_expert_id = simulation_data.get("target_expert_id")
            similarity_score = simulation_data.get("similarity_score", 0.0)
            
            # Get target expert
            target_expert = self.expert_registry.get_expert(target_expert_id)
            if not target_expert:
                return {
                    "success": False,
                    "error": f"Target expert {target_expert_id} not found"
                }
            
            # Calculate merge feasibility
            feasibility_score = similarity_score
            
            # Estimate resource requirements
            combined_memory = expert.memory_footprint_mb + target_expert.memory_footprint_mb
            estimated_memory_savings = combined_memory * 0.3  # Rough estimate
            
            result = {
                "success": True,
                "operation": "merge",
                "feasibility_score": feasibility_score,
                "target_expert": target_expert_id,
                "estimated_memory_savings_mb": estimated_memory_savings,
                "risk_factors": [
                    "Potential capability loss",
                    "Reduced specialization",
                    "Performance regression risk"
                ]
            }
            
            return result
            
        except Exception as e:
            return {
                "success": False,
                "error": f"Merge simulation failed: {str(e)}"
            }
    
    def _simulate_mutate(self, expert: ExpertMetadata,
                        simulation_data: Dict[str, Any]) -> Dict[str, Any]:
        """Simulate expert mutation operation"""
        try:
            # Extract simulation parameters
            mutation_rate = simulation_data.get("mutation_rate", 0.1)
            concept_drift_score = simulation_data.get("concept_drift_score", 0.0)
            
            # Calculate mutation feasibility
            feasibility_score = concept_drift_score
            
            # Estimate resource requirements (typically low for mutation)
            estimated_memory_change = expert.memory_footprint_mb * 0.1 * mutation_rate
            
            result = {
                "success": True,
                "operation": "mutate",
                "feasibility_score": feasibility_score,
                "mutation_rate": mutation_rate,
                "estimated_memory_change_mb": estimated_memory_change,
                "risk_factors": [
                    "Performance instability",
                    "Training time increase",
                    "Validation required"
                ]
            }
            
            return result
            
        except Exception as e:
            return {
                "success": False,
                "error": f"Mutation simulation failed: {str(e)}"
            }
    
    def execute_evolution(self, expert_id: str, evolution_type: EvolutionType,
                         execution_data: Dict[str, Any],
                         model_surgery_fn: Optional[Callable] = None) -> Dict[str, Any]:
        """
        Execute evolution operation on expert.
        
        Args:
            expert_id: ID of the expert to evolve
            evolution_type: Type of evolution operation
            execution_data: Data for execution
            model_surgery_fn: Optional function for model surgery operations
            
        Returns:
            Dict[str, Any]: Execution results
        """
        with self.lock:
            # First simulate the operation
            simulation_result = self.simulate_evolution(expert_id, evolution_type, execution_data)
            
            if not simulation_result.get("success", False):
                return simulation_result
            
            # Get expert metadata
            expert = self.expert_registry.get_expert(expert_id)
            if not expert:
                return {
                    "success": False,
                    "error": f"Expert {expert_id} not found"
                }
            
            # Create evolution event
            event_id = f"evolution_{int(time.time() * 1000000)}"
            event = EvolutionEvent(
                event_id=event_id,
                evolution_type=evolution_type,
                expert_ids=[expert_id],
                timestamp=time.time(),
                criteria_met=execution_data,  # Simplified
                success=False,
                details={}
            )
            
            try:
                # Perform actual evolution operation
                if evolution_type == EvolutionType.SPLIT:
                    result = self._execute_split(expert, execution_data, model_surgery_fn)
                elif evolution_type == EvolutionType.MERGE:
                    result = self._execute_merge(expert, execution_data, model_surgery_fn)
                elif evolution_type == EvolutionType.MUTATE:
                    result = self._execute_mutate(expert, execution_data, model_surgery_fn)
                else:
                    result = {
                        "success": False,
                        "error": f"Unknown evolution type: {evolution_type}"
                    }
                
                # Update event
                event.success = result.get("success", False)
                event.details = result
                
                # Store event
                self.evolution_history.append(event)
                
                # Keep history manageable
                if len(self.evolution_history) > 1000:
                    self.evolution_history = self.evolution_history[-1000:]
                
                print(f"🧬 Evolution {evolution_type.value} executed for expert {expert_id}: {result}")
                return result
                
            except Exception as e:
                event.success = False
                event.details = {"error": str(e)}
                self.evolution_history.append(event)
                return {
                    "success": False,
                    "error": f"Evolution execution failed: {str(e)}"
                }
    
    def _execute_split(self, expert: ExpertMetadata, execution_data: Dict[str, Any],
                      model_surgery_fn: Optional[Callable]) -> Dict[str, Any]:
        """Execute expert split operation"""
        try:
            # In a real implementation, this would involve:
            # 1. Cloning the expert model
            # 2. Applying weight modifications
            # 3. Registering new experts
            # 4. Updating routing policies
            
            # For simulation, we'll just register new experts
            new_expert_ids = []
            
            # Create two new experts based on the original
            for i in range(2):
                new_id = self.expert_registry.register_expert(
                    capabilities=expert.capabilities,
                    embedding_signature=[x + (0.1 * (i + 1)) for x in expert.embedding_signature],
                    device=expert.device,
                    memory_footprint_mb=expert.memory_footprint_mb * 0.6,  # Reduced memory
                    version=f"{expert.version}.split{i+1}"
                )
                new_expert_ids.append(new_id)
            
            # Mark original expert as retiring
            self.expert_registry.update_expert_metadata(
                expert.expert_id,
                status=ExpertStatus.RETIRING
            )
            
            return {
                "success": True,
                "operation": "split",
                "new_expert_ids": new_expert_ids,
                "original_expert_status": "retiring"
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": f"Split execution failed: {str(e)}"
            }
    
    def _execute_merge(self, expert: ExpertMetadata, execution_data: Dict[str, Any],
                      model_surgery_fn: Optional[Callable]) -> Dict[str, Any]:
        """Execute expert merge operation"""
        try:
            target_expert_id = execution_data.get("target_expert_id")
            
            # Get target expert
            if target_expert_id is None:
                return {
                    "success": False,
                    "error": "Target expert ID not provided"
                }
            target_expert = self.expert_registry.get_expert(target_expert_id)
            if not target_expert:
                return {
                    "success": False,
                    "error": f"Target expert {target_expert_id} not found"
                }
            
            # In a real implementation, this would involve:
            # 1. Combining model weights
            # 2. Updating the target expert
            # 3. Archiving the source expert
            
            # For simulation, we'll just update and archive
            # Update target expert capabilities (union of both)
            combined_capabilities = list(set(expert.capabilities + target_expert.capabilities))
            
            self.expert_registry.update_expert_metadata(
                target_expert_id,
                capabilities=combined_capabilities,
                memory_footprint_mb=target_expert.memory_footprint_mb * 1.2,  # Increased memory
                version=f"{target_expert.version}.merged"
            )
            
            # Mark source expert as archived
            self.expert_registry.update_expert_metadata(
                expert.expert_id,
                status=ExpertStatus.ARCHIVED
            )
            
            return {
                "success": True,
                "operation": "merge",
                "target_expert_id": target_expert_id,
                "source_expert_status": "archived",
                "target_expert_updated": True
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": f"Merge execution failed: {str(e)}"
            }
    
    def _execute_mutate(self, expert: ExpertMetadata, execution_data: Dict[str, Any],
                       model_surgery_fn: Optional[Callable]) -> Dict[str, Any]:
        """Execute expert mutation operation"""
        try:
            mutation_rate = execution_data.get("mutation_rate", 0.1)
            
            # In a real implementation, this would involve:
            # 1. Applying neuroevolutionary mutations
            # 2. Updating model weights
            # 3. Validating performance
            
            # For simulation, we'll just update metadata
            self.expert_registry.update_expert_metadata(
                expert.expert_id,
                version=f"{expert.version}.mutated",
                embedding_signature=[
                    x + (mutation_rate * (0.5 - hash(str(x)) % 1000 / 1000.0))
                    for x in expert.embedding_signature
                ]
            )
            
            return {
                "success": True,
                "operation": "mutate",
                "expert_id": expert.expert_id,
                "mutation_applied": True
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": f"Mutation execution failed: {str(e)}"
            }
    
    def _calculate_vector_distance(self, vec1: List[float], vec2: List[float]) -> float:
        """Calculate Euclidean distance between two vectors"""
        if len(vec1) != len(vec2):
            return float('inf')
        
        return math.sqrt(sum((a - b) ** 2 for a, b in zip(vec1, vec2)))
    
    def get_evolution_history(self, limit: int = 50) -> List[EvolutionEvent]:
        """
        Get evolution history.
        
        Args:
            limit: Maximum number of events to return
            
        Returns:
            List[EvolutionEvent]: Recent evolution events
        """
        with self.lock:
            return self.evolution_history[-limit:] if self.evolution_history else []
    
    def get_evolution_report(self) -> Dict[str, Any]:
        """
        Generate evolution report.
        
        Returns:
            Dict[str, Any]: Evolution report
        """
        with self.lock:
            if not self.evolution_history:
                return {
                    "status": "no_data",
                    "message": "No evolution events recorded"
                }
            
            # Calculate statistics
            total_events = len(self.evolution_history)
            successful_events = len([e for e in self.evolution_history if e.success])
            split_events = len([e for e in self.evolution_history if e.evolution_type == EvolutionType.SPLIT])
            merge_events = len([e for e in self.evolution_history if e.evolution_type == EvolutionType.MERGE])
            mutate_events = len([e for e in self.evolution_history if e.evolution_type == EvolutionType.MUTATE])
            
            # Get recent events
            recent_events = self.evolution_history[-10:] if len(self.evolution_history) >= 10 else self.evolution_history
            
            return {
                "status": "success",
                "total_events": total_events,
                "success_rate": successful_events / total_events if total_events > 0 else 0.0,
                "event_breakdown": {
                    "split": split_events,
                    "merge": merge_events,
                    "mutate": mutate_events
                },
                "recent_events": [
                    {
                        "event_id": event.event_id,
                        "type": event.evolution_type.value,
                        "experts": event.expert_ids,
                        "timestamp": event.timestamp,
                        "success": event.success
                    }
                    for event in recent_events
                ]
            }
    
    def clear_simulation_cache(self):
        """Clear simulation cache"""
        with self.lock:
            self.simulation_cache.clear()
            print("🗑️  Evolution simulation cache cleared")

# Global instance
_evolution_module = None

def get_evolution_module() -> EvolutionModule:
    """Get the global evolution module instance"""
    global _evolution_module
    if _evolution_module is None:
        _evolution_module = EvolutionModule()
    return _evolution_module

if __name__ == "__main__":
    # Example usage
    evolution = get_evolution_module()
    
    # Register a test expert
    registry = get_expert_registry()
    expert_id = registry.register_expert(
        capabilities=["nlp", "translation"],
        embedding_signature=[0.1, 0.2, 0.3, 0.4],
        device="cuda:0",
        memory_footprint_mb=1024.0
    )
    
    # Check split criteria
    gradient_vars = [0.6, 0.7, 0.8]  # High variance
    error_clusters = [[0.1, 0.2], [0.8, 0.9]]  # Distinct clusters
    
    should_split = evolution.check_split_criteria(expert_id, gradient_vars, error_clusters)
    print(f"Should split expert {expert_id}: {should_split}")
    
    # Simulate split
    simulation_data = {
        "gradient_variances": gradient_vars,
        "error_clusters": error_clusters
    }
    
    simulation_result = evolution.simulate_evolution(expert_id, EvolutionType.SPLIT, simulation_data)
    print(f"Split simulation result: {simulation_result}")
    
    # Execute split
    execution_result = evolution.execute_evolution(expert_id, EvolutionType.SPLIT, simulation_data)
    print(f"Split execution result: {execution_result}")
    
    # Get evolution report
    report = evolution.get_evolution_report()
    print(f"Evolution report: {report}")