"""
Lifecycle Manager for MAHIA Expert Engine
Manages expert lifecycle transitions and resource allocation.
"""

import time
import threading
from typing import Dict, Any, Optional, List, Callable
from dataclasses import dataclass
from enum import Enum

# Import expert registry and OptiCore components
from expert_registry import get_expert_registry, ExpertMetadata, ExpertStatus
from opticore.opticore import get_opticore

class LifecycleState(Enum):
    """Expert lifecycle states"""
    ACTIVE = "active"
    PAUSED = "paused"
    RETIRING = "retiring"
    ARCHIVED = "archived"
    DELETED = "deleted"

@dataclass
class LifecycleEvent:
    """Record of a lifecycle event"""
    event_id: str
    expert_id: str
    from_state: LifecycleState
    to_state: LifecycleState
    timestamp: float
    reason: str
    success: bool

class LifecycleManager:
    """Manager for expert lifecycle transitions and resource management"""
    
    def __init__(self):
        self.expert_registry = get_expert_registry()
        self.opticore = get_opticore()
        self.lifecycle_history = []
        self.heartbeat_monitors = {}
        self.lock = threading.RLock()
        
        print("🔄 LifecycleManager initialized")
    
    def transition_state(self, expert_id: str, new_state: LifecycleState,
                        reason: str = "manual_transition") -> bool:
        """
        Transition expert to a new state.
        
        Args:
            expert_id: ID of the expert
            new_state: New state to transition to
            reason: Reason for transition
            
        Returns:
            bool: True if successful, False otherwise
        """
        with self.lock:
            # Get current expert metadata
            expert = self.expert_registry.get_expert(expert_id)
            if not expert:
                print(f"⚠️  Expert {expert_id} not found")
                return False
            
            # Map ExpertStatus to LifecycleState
            current_state = self._map_expert_status_to_lifecycle(expert.status)
            
            # Validate transition
            if not self._is_valid_transition(current_state, new_state):
                print(f"⚠️  Invalid state transition: {current_state.value} → {new_state.value}")
                return False
            
            try:
                # Perform state-specific actions
                if new_state == LifecycleState.ACTIVE:
                    success = self._activate_expert(expert_id)
                elif new_state == LifecycleState.PAUSED:
                    success = self._pause_expert(expert_id)
                elif new_state == LifecycleState.RETIRING:
                    success = self._retire_expert(expert_id)
                elif new_state == LifecycleState.ARCHIVED:
                    success = self._archive_expert(expert_id)
                elif new_state == LifecycleState.DELETED:
                    success = self._delete_expert(expert_id)
                else:
                    success = False
                
                if success:
                    # Update expert registry
                    expert_status = self._map_lifecycle_to_expert_status(new_state)
                    self.expert_registry.update_expert_metadata(expert_id, status=expert_status)
                    
                    # Record lifecycle event
                    event = LifecycleEvent(
                        event_id=f"lifecycle_{int(time.time() * 1000000)}",
                        expert_id=expert_id,
                        from_state=current_state,
                        to_state=new_state,
                        timestamp=time.time(),
                        reason=reason,
                        success=True
                    )
                    self.lifecycle_history.append(event)
                    
                    print(f"🔄 Expert {expert_id} transitioned: {current_state.value} → {new_state.value}")
                    return True
                else:
                    # Record failed event
                    event = LifecycleEvent(
                        event_id=f"lifecycle_{int(time.time() * 1000000)}",
                        expert_id=expert_id,
                        from_state=current_state,
                        to_state=new_state,
                        timestamp=time.time(),
                        reason=reason,
                        success=False
                    )
                    self.lifecycle_history.append(event)
                    
                    print(f"❌ Failed to transition expert {expert_id}: {current_state.value} → {new_state.value}")
                    return False
                    
            except Exception as e:
                # Record failed event
                event = LifecycleEvent(
                    event_id=f"lifecycle_{int(time.time() * 1000000)}",
                    expert_id=expert_id,
                    from_state=current_state,
                    to_state=new_state,
                    timestamp=time.time(),
                    reason=reason,
                    success=False
                )
                self.lifecycle_history.append(event)
                
                print(f"💥 Error during transition for expert {expert_id}: {str(e)}")
                return False
    
    def _map_expert_status_to_lifecycle(self, status: ExpertStatus) -> LifecycleState:
        """Map ExpertStatus to LifecycleState"""
        mapping = {
            ExpertStatus.ACTIVE: LifecycleState.ACTIVE,
            ExpertStatus.PAUSED: LifecycleState.PAUSED,
            ExpertStatus.RETIRING: LifecycleState.RETIRING,
            ExpertStatus.ARCHIVED: LifecycleState.ARCHIVED,
            ExpertStatus.INACTIVE: LifecycleState.ARCHIVED
        }
        return mapping.get(status, LifecycleState.ARCHIVED)
    
    def _map_lifecycle_to_expert_status(self, state: LifecycleState) -> ExpertStatus:
        """Map LifecycleState to ExpertStatus"""
        mapping = {
            LifecycleState.ACTIVE: ExpertStatus.ACTIVE,
            LifecycleState.PAUSED: ExpertStatus.PAUSED,
            LifecycleState.RETIRING: ExpertStatus.RETIRING,
            LifecycleState.ARCHIVED: ExpertStatus.ARCHIVED,
            LifecycleState.DELETED: ExpertStatus.INACTIVE
        }
        return mapping.get(state, ExpertStatus.INACTIVE)
    
    def _is_valid_transition(self, from_state: LifecycleState, to_state: LifecycleState) -> bool:
        """Check if state transition is valid"""
        # Define valid transitions
        valid_transitions = {
            LifecycleState.ACTIVE: [LifecycleState.PAUSED, LifecycleState.RETIRING],
            LifecycleState.PAUSED: [LifecycleState.ACTIVE, LifecycleState.RETIRING],
            LifecycleState.RETIRING: [LifecycleState.ARCHIVED],
            LifecycleState.ARCHIVED: [LifecycleState.DELETED, LifecycleState.ACTIVE],
            LifecycleState.DELETED: []  # Terminal state
        }
        
        return to_state in valid_transitions.get(from_state, [])
    
    def _activate_expert(self, expert_id: str) -> bool:
        """Activate an expert"""
        try:
            # Allocate resources through OptiCore
            expert = self.expert_registry.get_expert(expert_id)
            if not expert:
                return False
            
            # In practice, this would involve:
            # 1. Allocating memory through OptiCore
            # 2. Loading model weights
            # 3. Starting any required services
            
            print(f"⚡ Activating expert {expert_id}")
            return True
        except Exception as e:
            print(f"❌ Error activating expert {expert_id}: {str(e)}")
            return False
    
    def _pause_expert(self, expert_id: str) -> bool:
        """Pause an expert"""
        try:
            # In practice, this would involve:
            # 1. Suspending processing
            # 2. Keeping memory allocated but not active
            
            print(f"⏸️  Pausing expert {expert_id}")
            return True
        except Exception as e:
            print(f"❌ Error pausing expert {expert_id}: {str(e)}")
            return False
    
    def _retire_expert(self, expert_id: str) -> bool:
        """Retire an expert"""
        try:
            # In practice, this would involve:
            # 1. Stopping active processing
            # 2. Preparing for archival
            
            print(f"🌅 Retiring expert {expert_id}")
            return True
        except Exception as e:
            print(f"❌ Error retiring expert {expert_id}: {str(e)}")
            return False
    
    def _archive_expert(self, expert_id: str) -> bool:
        """Archive an expert"""
        try:
            # In practice, this would involve:
            # 1. Freeing most resources
            # 2. Storing model weights to persistent storage
            # 3. Keeping metadata for potential revival
            
            print(f"📦 Archiving expert {expert_id}")
            return True
        except Exception as e:
            print(f"❌ Error archiving expert {expert_id}: {str(e)}")
            return False
    
    def _delete_expert(self, expert_id: str) -> bool:
        """Delete an expert"""
        try:
            # In practice, this would involve:
            # 1. Freeing all resources
            # 2. Removing from registry
            # 3. Deleting persistent storage
            
            print(f"🧨 Deleting expert {expert_id}")
            return True
        except Exception as e:
            print(f"❌ Error deleting expert {expert_id}: {str(e)}")
            return False
    
    def handle_memory_pressure(self, memory_threshold_mb: float = 8000.0) -> List[str]:
        """
        Handle memory pressure by transitioning experts.
        
        Args:
            memory_threshold_mb: Memory threshold in MB
            
        Returns:
            List[str]: List of expert IDs that were transitioned
        """
        with self.lock:
            # Get current memory usage (simplified)
            # In practice, this would query OptiCore for actual memory usage
            active_experts = self.expert_registry.get_active_experts()
            
            # Sort by memory footprint (descending)
            active_experts.sort(key=lambda e: e.memory_footprint_mb, reverse=True)
            
            # Transition high-memory experts to retiring state
            transitioned_experts = []
            
            for expert in active_experts:
                if expert.memory_footprint_mb > memory_threshold_mb / len(active_experts):
                    if self.transition_state(expert.expert_id, LifecycleState.RETIRING, 
                                           "memory_pressure"):
                        transitioned_experts.append(expert.expert_id)
            
            if transitioned_experts:
                print(f"MemoryWarning: Transitioned {len(transitioned_experts)} experts due to memory pressure")
            
            return transitioned_experts
    
    def start_heartbeat_monitoring(self, expert_id: str, interval: int = 30):
        """
        Start monitoring expert heartbeats.
        
        Args:
            expert_id: ID of the expert to monitor
            interval: Check interval in seconds
        """
        with self.lock:
            if expert_id in self.heartbeat_monitors:
                # Stop existing monitor
                self.heartbeat_monitors[expert_id]["stop"] = True
            
            # Start new monitor
            monitor_thread = threading.Thread(
                target=self._heartbeat_monitor_worker,
                args=(expert_id, interval),
                daemon=True
            )
            
            self.heartbeat_monitors[expert_id] = {
                "thread": monitor_thread,
                "stop": False,
                "interval": interval
            }
            
            monitor_thread.start()
            print(f"❤️  Started heartbeat monitoring for expert {expert_id}")
    
    def stop_heartbeat_monitoring(self, expert_id: str):
        """
        Stop monitoring expert heartbeats.
        
        Args:
            expert_id: ID of the expert to stop monitoring
        """
        with self.lock:
            if expert_id in self.heartbeat_monitors:
                self.heartbeat_monitors[expert_id]["stop"] = True
                del self.heartbeat_monitors[expert_id]
                print(f"💔 Stopped heartbeat monitoring for expert {expert_id}")
    
    def _heartbeat_monitor_worker(self, expert_id: str, interval: int):
        """Worker function for heartbeat monitoring"""
        while True:
            with self.lock:
                if expert_id in self.heartbeat_monitors and self.heartbeat_monitors[expert_id]["stop"]:
                    break
            
            # Check heartbeat
            expert = self.expert_registry.get_expert(expert_id)
            if expert:
                time_since_last_heartbeat = time.time() - expert.last_heartbeat
                
                # If no heartbeat for 2 minutes, mark as inactive
                if time_since_last_heartbeat > 120:
                    print(f"⚠️  Expert {expert_id} missed heartbeat for {time_since_last_heartbeat:.0f}s")
                    self.transition_state(expert_id, LifecycleState.PAUSED, "heartbeat_timeout")
                    break
            
            time.sleep(interval)
    
    def graceful_shutdown(self, expert_id: str) -> bool:
        """
        Gracefully shutdown an expert.
        
        Args:
            expert_id: ID of the expert to shutdown
            
        Returns:
            bool: True if successful, False otherwise
        """
        with self.lock:
            # Transition through proper states
            current_expert = self.expert_registry.get_expert(expert_id)
            if not current_expert:
                return False
            
            current_state = self._map_expert_status_to_lifecycle(current_expert.status)
            
            # Follow proper shutdown sequence
            if current_state == LifecycleState.ACTIVE:
                # Active → Paused → Retiring → Archived
                if not self.transition_state(expert_id, LifecycleState.PAUSED, "graceful_shutdown"):
                    return False
                time.sleep(1)  # Brief pause
                
            if current_state in [LifecycleState.ACTIVE, LifecycleState.PAUSED]:
                if not self.transition_state(expert_id, LifecycleState.RETIRING, "graceful_shutdown"):
                    return False
                time.sleep(1)  # Brief retirement period
            
            # Final transition to archived
            return self.transition_state(expert_id, LifecycleState.ARCHIVED, "graceful_shutdown")
    
    def get_lifecycle_history(self, expert_id: Optional[str] = None, limit: int = 50) -> List[LifecycleEvent]:
        """
        Get lifecycle history.
        
        Args:
            expert_id: Optional expert ID to filter by
            limit: Maximum number of events to return
            
        Returns:
            List[LifecycleEvent]: Lifecycle events
        """
        with self.lock:
            if expert_id:
                filtered_events = [e for e in self.lifecycle_history if e.expert_id == expert_id]
            else:
                filtered_events = self.lifecycle_history
            
            return filtered_events[-limit:] if filtered_events else []
    
    def get_lifecycle_report(self) -> Dict[str, Any]:
        """
        Generate lifecycle report.
        
        Returns:
            Dict[str, Any]: Lifecycle report
        """
        with self.lock:
            if not self.lifecycle_history:
                return {
                    "status": "no_data",
                    "message": "No lifecycle events recorded"
                }
            
            # Get current expert states
            experts = self.expert_registry.get_active_experts()
            state_counts = {}
            for expert in experts:
                state = self._map_expert_status_to_lifecycle(expert.status)
                state_counts[state.value] = state_counts.get(state.value, 0) + 1
            
            # Calculate statistics
            total_events = len(self.lifecycle_history)
            successful_events = len([e for e in self.lifecycle_history if e.success])
            
            # Get recent events
            recent_events = self.lifecycle_history[-10:] if len(self.lifecycle_history) >= 10 else self.lifecycle_history
            
            return {
                "status": "success",
                "total_events": total_events,
                "success_rate": successful_events / total_events if total_events > 0 else 0.0,
                "current_expert_states": state_counts,
                "recent_events": [
                    {
                        "expert_id": event.expert_id,
                        "transition": f"{event.from_state.value} → {event.to_state.value}",
                        "timestamp": event.timestamp,
                        "reason": event.reason,
                        "success": event.success
                    }
                    for event in recent_events
                ]
            }

# Global instance
_lifecycle_manager = None

def get_lifecycle_manager() -> LifecycleManager:
    """Get the global lifecycle manager instance"""
    global _lifecycle_manager
    if _lifecycle_manager is None:
        _lifecycle_manager = LifecycleManager()
    return _lifecycle_manager

if __name__ == "__main__":
    # Example usage
    lifecycle = get_lifecycle_manager()
    
    # Register a test expert
    registry = get_expert_registry()
    expert_id = registry.register_expert(
        capabilities=["nlp"],
        embedding_signature=[0.1, 0.2, 0.3, 0.4],
        device="cuda:0",
        memory_footprint_mb=1024.0
    )
    
    print(f"Registered expert: {expert_id}")
    
    # Transition states
    lifecycle.transition_state(expert_id, LifecycleState.PAUSED, "test_pause")
    lifecycle.transition_state(expert_id, LifecycleState.ACTIVE, "test_resume")
    
    # Handle memory pressure
    transitioned = lifecycle.handle_memory_pressure(500.0)
    print(f"Experts transitioned due to memory pressure: {transitioned}")
    
    # Get lifecycle report
    report = lifecycle.get_lifecycle_report()
    print(f"Lifecycle report: {report}")