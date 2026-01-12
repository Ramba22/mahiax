"""
Expert Registry & Manager for MAHIA Expert Engine
Handles registration, management, and metadata of experts in the system.
"""

import time
import uuid
import threading
from typing import Dict, Any, Optional, List, Set
from dataclasses import dataclass, asdict
from enum import Enum

# Conditional imports with fallbacks
try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    redis = None

class ExpertStatus(Enum):
    """Expert lifecycle status"""
    ACTIVE = "active"
    PAUSED = "paused"
    RETIRING = "retiring"
    ARCHIVED = "archived"
    INACTIVE = "inactive"

@dataclass
class ExpertMetadata:
    """Metadata for an expert"""
    expert_id: str
    capabilities: List[str]
    embedding_signature: List[float]
    device: str
    memory_footprint_mb: float
    version: str
    health_status: str
    registration_time: float
    last_heartbeat: float
    status: ExpertStatus = ExpertStatus.ACTIVE

class ExpertRegistry:
    """Registry and manager for experts in the MAHIA system"""
    
    def __init__(self, use_persistence: bool = False, redis_host: str = "localhost", redis_port: int = 6379):
        self.experts: Dict[str, ExpertMetadata] = {}
        self.use_persistence = use_persistence
        self.redis_client = None
        self.lock = threading.RLock()
        
        # Initialize Redis if available and requested
        if use_persistence and REDIS_AVAILABLE and redis is not None:
            try:
                self.redis_client = redis.Redis(host=redis_host, port=redis_port, decode_responses=True)
                self.redis_client.ping()
                print("🔗 ExpertRegistry connected to Redis")
            except Exception as e:
                print(f"⚠️  Redis connection failed: {e}")
                self.redis_client = None
        elif use_persistence:
            print("⚠️  Redis not available, using in-memory storage only")
        
        print("📚 ExpertRegistry initialized")
    
    def register_expert(self, capabilities: List[str], embedding_signature: List[float], 
                       device: str, memory_footprint_mb: float, version: str = "1.0.0",
                       expert_id: Optional[str] = None) -> str:
        """
        Register a new expert in the system.
        
        Args:
            capabilities: List of capabilities this expert supports
            embedding_signature: Expert's embedding signature
            device: Device where expert is located (e.g., "cuda:0", "cpu")
            memory_footprint_mb: Memory footprint in MB
            version: Expert version
            expert_id: Optional predefined expert ID
            
        Returns:
            str: Generated or provided expert ID
        """
        if expert_id is None:
            expert_id = str(uuid.uuid4())
            
        with self.lock:
            metadata = ExpertMetadata(
                expert_id=expert_id,
                capabilities=capabilities,
                embedding_signature=embedding_signature,
                device=device,
                memory_footprint_mb=memory_footprint_mb,
                version=version,
                health_status="healthy",
                registration_time=time.time(),
                last_heartbeat=time.time(),
                status=ExpertStatus.ACTIVE
            )
            
            self.experts[expert_id] = metadata
            
            # Persist to Redis if available
            if self.redis_client:
                try:
                    self.redis_client.hset(f"expert:{expert_id}", mapping=asdict(metadata))
                    self.redis_client.sadd("active_experts", expert_id)
                except Exception as e:
                    print(f"⚠️  Redis persistence failed: {e}")
            
            print(f"🆕 Registered expert {expert_id} with capabilities: {capabilities}")
            return expert_id
    
    def update_expert_metadata(self, expert_id: str, **kwargs) -> bool:
        """
        Update expert metadata.
        
        Args:
            expert_id: ID of the expert to update
            **kwargs: Metadata fields to update
            
        Returns:
            bool: True if successful, False if expert not found
        """
        with self.lock:
            if expert_id not in self.experts:
                print(f"⚠️  Expert {expert_id} not found")
                return False
                
            # Update metadata
            expert = self.experts[expert_id]
            for key, value in kwargs.items():
                if hasattr(expert, key):
                    setattr(expert, key, value)
                    
            # Update last heartbeat if not explicitly set
            if 'last_heartbeat' not in kwargs:
                expert.last_heartbeat = time.time()
                
            # Persist to Redis if available
            if self.redis_client:
                try:
                    update_dict = {}
                    for key, value in kwargs.items():
                        if hasattr(expert, key):
                            update_dict[key] = str(value)
                    update_dict['last_heartbeat'] = str(expert.last_heartbeat)
                    self.redis_client.hset(f"expert:{expert_id}", mapping=update_dict)
                except Exception as e:
                    print(f"⚠️  Redis update failed: {e}")
            
            print(f"🔄 Updated expert {expert_id} metadata")
            return True
    
    def get_active_experts(self) -> List[ExpertMetadata]:
        """
        Get all active experts.
        
        Returns:
            List[ExpertMetadata]: List of active expert metadata
        """
        with self.lock:
            active_experts = [
                expert for expert in self.experts.values() 
                if expert.status == ExpertStatus.ACTIVE
            ]
            return active_experts
    
    def mark_inactive(self, expert_id: str) -> bool:
        """
        Mark an expert as inactive.
        
        Args:
            expert_id: ID of the expert to mark inactive
            
        Returns:
            bool: True if successful, False if expert not found
        """
        return self.update_expert_metadata(expert_id, status=ExpertStatus.INACTIVE)
    
    def fetch_expert_for_ctx(self, context_vector: List[float], 
                           required_capabilities: Optional[List[str]] = None) -> Optional[ExpertMetadata]:
        """
        Fetch the most suitable expert for a given context.
        
        Args:
            context_vector: Context vector to match against
            required_capabilities: Required capabilities (optional)
            
        Returns:
            ExpertMetadata: Best matching expert or None if none found
        """
        with self.lock:
            # Filter active experts with required capabilities
            candidates = self.get_active_experts()
            
            if required_capabilities:
                candidates = [
                    expert for expert in candidates
                    if all(cap in expert.capabilities for cap in required_capabilities)
                ]
            
            if not candidates:
                print("⚠️  No suitable experts found for context")
                return None
            
            # Simple similarity matching (cosine similarity would be better in practice)
            best_expert = None
            best_score = -1.0
            
            for expert in candidates:
                # Calculate simple dot product similarity (simplified)
                if len(expert.embedding_signature) == len(context_vector):
                    similarity = sum(a * b for a, b in zip(expert.embedding_signature, context_vector))
                    if similarity > best_score:
                        best_score = similarity
                        best_expert = expert
            
            if best_expert:
                print(f"🎯 Selected expert {best_expert.expert_id} with score {best_score:.3f}")
            
            return best_expert
    
    def heartbeat(self, expert_id: str) -> bool:
        """
        Update expert heartbeat.
        
        Args:
            expert_id: ID of the expert sending heartbeat
            
        Returns:
            bool: True if successful, False if expert not found
        """
        return self.update_expert_metadata(expert_id, last_heartbeat=time.time())
    
    def get_expert(self, expert_id: str) -> Optional[ExpertMetadata]:
        """
        Get expert metadata by ID.
        
        Args:
            expert_id: ID of the expert
            
        Returns:
            ExpertMetadata: Expert metadata or None if not found
        """
        with self.lock:
            return self.experts.get(expert_id)
    
    def get_expert_count(self) -> Dict[str, int]:
        """
        Get count of experts by status.
        
        Returns:
            Dict[str, int]: Count of experts by status
        """
        with self.lock:
            counts = {}
            for expert in self.experts.values():
                status = expert.status.value
                counts[status] = counts.get(status, 0) + 1
            return counts
    
    def cleanup_stale_experts(self, timeout_seconds: int = 300) -> int:
        """
        Remove experts that haven't sent heartbeat within timeout.
        
        Args:
            timeout_seconds: Timeout in seconds
            
        Returns:
            int: Number of experts cleaned up
        """
        with self.lock:
            current_time = time.time()
            stale_experts = [
                expert_id for expert_id, expert in self.experts.items()
                if current_time - expert.last_heartbeat > timeout_seconds
            ]
            
            for expert_id in stale_experts:
                self.mark_inactive(expert_id)
                # Remove from Redis if available
                if self.redis_client:
                    try:
                        self.redis_client.srem("active_experts", expert_id)
                    except Exception as e:
                        print(f"⚠️  Redis cleanup failed: {e}")
            
            cleanup_count = len(stale_experts)
            if cleanup_count > 0:
                print(f"🧹 Cleaned up {cleanup_count} stale experts")
            
            return cleanup_count

# Global instance
_expert_registry = None

def get_expert_registry() -> ExpertRegistry:
    """Get the global expert registry instance"""
    global _expert_registry
    if _expert_registry is None:
        _expert_registry = ExpertRegistry()
    return _expert_registry

if __name__ == "__main__":
    # Example usage
    registry = get_expert_registry()
    
    # Register some experts
    expert1_id = registry.register_expert(
        capabilities=["nlp", "translation"],
        embedding_signature=[0.1, 0.2, 0.3, 0.4],
        device="cuda:0",
        memory_footprint_mb=1024.0,
        version="1.0.0"
    )
    
    expert2_id = registry.register_expert(
        capabilities=["vision", "classification"],
        embedding_signature=[0.4, 0.3, 0.2, 0.1],
        device="cuda:1",
        memory_footprint_mb=2048.0,
        version="1.1.0"
    )
    
    # Update expert metadata
    registry.update_expert_metadata(expert1_id, health_status="degraded")
    
    # Get active experts
    active_experts = registry.get_active_experts()
    print(f"Active experts: {len(active_experts)}")
    
    # Fetch expert for context
    context = [0.15, 0.25, 0.35, 0.45]
    expert = registry.fetch_expert_for_ctx(context, required_capabilities=["nlp"])
    if expert:
        print(f"Selected expert: {expert.expert_id}")
    
    # Show expert counts
    counts = registry.get_expert_count()
    print(f"Expert counts: {counts}")