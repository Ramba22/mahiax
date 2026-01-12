"""
Expert Utilization Balancer for MAHIA Expert Engine
Tracks expert usage and balances load to prevent over/under utilization.
"""

import time
import threading
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass
from collections import defaultdict, deque
import math

# Import expert registry and router
from expert_registry import get_expert_registry, ExpertMetadata, ExpertStatus
from contextual_router import get_contextual_router

@dataclass
class UsageStats:
    """Usage statistics for an expert"""
    expert_id: str
    total_requests: int
    recent_requests: int  # In sliding window
    avg_response_time: float
    error_count: int
    last_used: float
    utilization_score: float

class UtilizationBalancer:
    """Balancer for expert utilization and load distribution"""
    
    def __init__(self, window_size: int = 100, cooldown_period: int = 30):
        self.expert_registry = get_expert_registry()
        self.contextual_router = get_contextual_router()
        self.window_size = window_size
        self.cooldown_period = cooldown_period
        self.usage_counters = defaultdict(lambda: deque(maxlen=window_size))
        self.error_counters = defaultdict(int)
        self.cooldowns = {}  # expert_id -> cooldown_end_time
        self.lock = threading.RLock()
        
        # Policy thresholds
        self.overload_threshold = 0.8  # 80% utilization
        self.underload_threshold = 0.2  # 20% utilization
        self.error_rate_threshold = 0.1  # 10% error rate
        
        print("⚖️  UtilizationBalancer initialized")
    
    def track_usage(self, expert_id: str, response_time: float = 0.0, success: bool = True):
        """
        Track expert usage for load balancing.
        
        Args:
            expert_id: ID of the expert used
            response_time: Response time in seconds
            success: Whether the request was successful
        """
        with self.lock:
            timestamp = time.time()
            self.usage_counters[expert_id].append({
                "timestamp": timestamp,
                "response_time": response_time,
                "success": success
            })
            
            if not success:
                self.error_counters[expert_id] += 1
            
            print(f"📊 Tracked usage for expert {expert_id}: {response_time:.3f}s, {'success' if success else 'failure'}")
    
    def get_usage_stats(self, expert_id: str) -> UsageStats:
        """
        Get usage statistics for an expert.
        
        Args:
            expert_id: ID of the expert
            
        Returns:
            UsageStats: Usage statistics
        """
        with self.lock:
            usage_history = self.usage_counters[expert_id]
            if not usage_history:
                return UsageStats(
                    expert_id=expert_id,
                    total_requests=0,
                    recent_requests=0,
                    avg_response_time=0.0,
                    error_count=self.error_counters[expert_id],
                    last_used=0.0,
                    utilization_score=0.0
                )
            
            # Calculate statistics
            total_requests = len(usage_history)
            recent_requests = len([u for u in usage_history if time.time() - u["timestamp"] < 60])  # Last minute
            successful_requests = len([u for u in usage_history if u["success"]])
            avg_response_time = sum(u["response_time"] for u in usage_history) / total_requests if total_requests > 0 else 0.0
            error_count = self.error_counters[expert_id]
            last_used = max(u["timestamp"] for u in usage_history)
            
            # Calculate utilization score (0.0 to 1.0)
            # Based on recent request rate and success rate
            time_window = 60.0  # 1 minute window
            requests_per_second = recent_requests / time_window
            success_rate = successful_requests / total_requests if total_requests > 0 else 1.0
            
            # Normalize requests per second (assuming max 10 RPS per expert is good)
            normalized_rps = min(1.0, requests_per_second / 10.0)
            
            # Utilization score combines request rate and success rate
            utilization_score = normalized_rps * success_rate
            
            return UsageStats(
                expert_id=expert_id,
                total_requests=total_requests,
                recent_requests=recent_requests,
                avg_response_time=avg_response_time,
                error_count=error_count,
                last_used=last_used,
                utilization_score=utilization_score
            )
    
    def get_all_usage_stats(self) -> List[UsageStats]:
        """
        Get usage statistics for all experts.
        
        Returns:
            List[UsageStats]: Usage statistics for all experts
        """
        with self.lock:
            expert_ids = set(list(self.usage_counters.keys()) + list(self.error_counters.keys()))
            return [self.get_usage_stats(expert_id) for expert_id in expert_ids]
    
    def generate_heatmap(self) -> Dict[str, float]:
        """
        Generate utilization heatmap for all experts.
        
        Returns:
            Dict[str, float]: Expert ID to utilization score mapping
        """
        with self.lock:
            heatmap = {}
            stats = self.get_all_usage_stats()
            for stat in stats:
                heatmap[stat.expert_id] = stat.utilization_score
            return heatmap
    
    def check_balancing_actions(self) -> List[Dict[str, Any]]:
        """
        Check for and suggest balancing actions.
        
        Returns:
            List[Dict[str, Any]]: List of suggested actions
        """
        actions = []
        current_time = time.time()
        
        with self.lock:
            stats = self.get_all_usage_stats()
            
            for stat in stats:
                expert_id = stat.expert_id
                
                # Check if expert is in cooldown
                if expert_id in self.cooldowns and current_time < self.cooldowns[expert_id]:
                    continue
                
                # Calculate error rate
                error_rate = stat.error_count / stat.total_requests if stat.total_requests > 0 else 0.0
                
                # Check for overload
                if stat.utilization_score > self.overload_threshold:
                    actions.append({
                        "action": "throttle",
                        "expert_id": expert_id,
                        "reason": f"High utilization ({stat.utilization_score:.2f})",
                        "severity": "high"
                    })
                
                # Check for underload
                elif stat.utilization_score < self.underload_threshold:
                    actions.append({
                        "action": "boost_usage",
                        "expert_id": expert_id,
                        "reason": f"Low utilization ({stat.utilization_score:.2f})",
                        "severity": "low"
                    })
                
                # Check for high error rate
                if error_rate > self.error_rate_threshold:
                    actions.append({
                        "action": "cooldown",
                        "expert_id": expert_id,
                        "reason": f"High error rate ({error_rate:.2f})",
                        "severity": "critical"
                    })
                
                # Check for dead expert (no usage for a long time)
                time_since_last_use = current_time - stat.last_used
                if time_since_last_use > 300:  # 5 minutes
                    actions.append({
                        "action": "revive",
                        "expert_id": expert_id,
                        "reason": f"Dead expert (not used for {time_since_last_use:.0f}s)",
                        "severity": "medium"
                    })
        
        return actions
    
    def apply_balancing_action(self, action: Dict[str, Any]) -> bool:
        """
        Apply a balancing action.
        
        Args:
            action: Action to apply
            
        Returns:
            bool: True if successful, False otherwise
        """
        action_type = action.get("action")
        expert_id = action.get("expert_id")
        
        if not action_type or not expert_id:
            return False
        
        current_time = time.time()
        
        with self.lock:
            if action_type == "throttle":
                # Implement throttling (e.g., reduce quota)
                print(f"🐢 Throttling expert {expert_id}")
                self.cooldowns[expert_id] = current_time + 10  # 10 second cooldown
                return True
                
            elif action_type == "boost_usage":
                # Implement usage boost (e.g., increase quota or forced sampling)
                print(f"🚀 Boosting usage for expert {expert_id}")
                # This would typically involve adjusting routing policies
                return True
                
            elif action_type == "cooldown":
                # Implement cooldown for problematic expert
                cooldown_duration = self.cooldown_period
                self.cooldowns[expert_id] = current_time + cooldown_duration
                print(f"❄️  Cooling down expert {expert_id} for {cooldown_duration}s")
                return True
                
            elif action_type == "revive":
                # Implement revival for dead expert (e.g., forced sampling)
                print(f"⚡ Reviving expert {expert_id}")
                # This would typically involve adjusting routing to include this expert
                return True
                
            else:
                print(f"⚠️  Unknown action type: {action_type}")
                return False
    
    def run_periodic_balancing(self) -> Dict[str, Any]:
        """
        Run periodic balancing check and apply actions.
        
        Returns:
            Dict[str, Any]: Balancing results
        """
        with self.lock:
            actions = self.check_balancing_actions()
            applied_actions = []
            
            for action in actions:
                if self.apply_balancing_action(action):
                    applied_actions.append(action)
            
            result = {
                "timestamp": time.time(),
                "total_actions": len(actions),
                "applied_actions": applied_actions,
                "active_cooldowns": len([t for t in self.cooldowns.values() if t > time.time()])
            }
            
            if applied_actions:
                print(f"⚖️  Applied {len(applied_actions)} balancing actions")
            
            return result
    
    def get_policy_engine_status(self) -> Dict[str, Any]:
        """
        Get policy engine status and configuration.
        
        Returns:
            Dict[str, Any]: Policy engine status
        """
        with self.lock:
            return {
                "thresholds": {
                    "overload": self.overload_threshold,
                    "underload": self.underload_threshold,
                    "error_rate": self.error_rate_threshold
                },
                "configuration": {
                    "window_size": self.window_size,
                    "cooldown_period": self.cooldown_period
                },
                "current_state": {
                    "tracked_experts": len(self.usage_counters),
                    "active_cooldowns": len([t for t in self.cooldowns.values() if t > time.time()]),
                    "total_errors": sum(self.error_counters.values())
                }
            }
    
    def adjust_policy_thresholds(self, overload: Optional[float] = None, 
                               underload: Optional[float] = None,
                               error_rate: Optional[float] = None):
        """
        Adjust policy thresholds.
        
        Args:
            overload: New overload threshold
            underload: New underload threshold
            error_rate: New error rate threshold
        """
        with self.lock:
            if overload is not None:
                self.overload_threshold = max(0.0, min(1.0, overload))
            if underload is not None:
                self.underload_threshold = max(0.0, min(1.0, underload))
            if error_rate is not None:
                self.error_rate_threshold = max(0.0, min(1.0, error_rate))
            
            print(f"⚙️  Policy thresholds updated: overload={self.overload_threshold}, "
                  f"underload={self.underload_threshold}, error_rate={self.error_rate_threshold}")

# Global instance
_utilization_balancer = None

def get_utilization_balancer() -> UtilizationBalancer:
    """Get the global utilization balancer instance"""
    global _utilization_balancer
    if _utilization_balancer is None:
        _utilization_balancer = UtilizationBalancer()
    return _utilization_balancer

if __name__ == "__main__":
    # Example usage
    balancer = get_utilization_balancer()
    
    # Simulate some expert usage
    balancer.track_usage("expert_1", response_time=0.1, success=True)
    balancer.track_usage("expert_1", response_time=0.2, success=True)
    balancer.track_usage("expert_1", response_time=0.15, success=False)  # Error
    balancer.track_usage("expert_2", response_time=0.05, success=True)
    
    # Get usage stats
    stats1 = balancer.get_usage_stats("expert_1")
    stats2 = balancer.get_usage_stats("expert_2")
    print(f"Expert 1 stats: {stats1}")
    print(f"Expert 2 stats: {stats2}")
    
    # Generate heatmap
    heatmap = balancer.generate_heatmap()
    print(f"Heatmap: {heatmap}")
    
    # Check for balancing actions
    actions = balancer.check_balancing_actions()
    print(f"Balancing actions: {actions}")
    
    # Run periodic balancing
    result = balancer.run_periodic_balancing()
    print(f"Balancing result: {result}")
    
    # Get policy status
    policy_status = balancer.get_policy_engine_status()
    print(f"Policy status: {policy_status}")