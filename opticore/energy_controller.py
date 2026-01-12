"""
Energy Controller for MAHIA OptiCore
Calculation of Power Efficiency Score and adaptation of batch size, frequency or precision.
"""

import time
import threading
from typing import Dict, Any, Optional, List, Tuple
from collections import defaultdict, deque

class EnergyController:
    """Energy efficiency controller with Power Efficiency Score calculation"""
    
    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self.power_history = deque(maxlen=window_size)
        self.performance_history = deque(maxlen=window_size)
        self.efficiency_scores = deque(maxlen=window_size)
        self.lock = threading.Lock()
        self.stats = {
            "efficiency_improvements": 0,
            "optimizations_made": 0,
            "power_savings_wh": 0.0
        }
        
        print(f"💡 EnergyController initialized with window size: {window_size}")
        
    def calculate_efficiency_score(self, power_watts: float, performance_metric: float, 
                                 time_seconds: float) -> float:
        """
        Calculate Power Efficiency Score.
        
        Args:
            power_watts: Power consumption in watts
            performance_metric: Performance metric (e.g., samples per second)
            time_seconds: Time period in seconds
            
        Returns:
            float: Efficiency score (higher is better)
        """
        if power_watts <= 0 or time_seconds <= 0:
            return 0.0
            
        # Energy consumed in watt-hours
        energy_wh = power_watts * (time_seconds / 3600.0)
        
        # Efficiency score: performance per unit energy
        efficiency_score = performance_metric / energy_wh if energy_wh > 0 else 0.0
        
        # Store in history
        timestamp = time.time()
        with self.lock:
            self.power_history.append((timestamp, power_watts))
            self.performance_history.append((timestamp, performance_metric))
            self.efficiency_scores.append((timestamp, efficiency_score))
            
        print(f"⚡ Efficiency Score: {efficiency_score:.2f} (performance/Wh)")
        return efficiency_score
        
    def optimize_step(self, batch_time: float, batch_size: int, 
                     current_power: float) -> Dict[str, Any]:
        """
        Optimize energy efficiency for the current step.
        
        Args:
            batch_time: Time to process batch in seconds
            batch_size: Number of samples in batch
            current_power: Current power consumption in watts
            
        Returns:
            Dict with optimization recommendations
        """
        # Calculate performance metrics
        samples_per_second = batch_size / batch_time if batch_time > 0 else 0
        performance_metric = samples_per_second
        
        # Calculate efficiency score
        efficiency_score = self.calculate_efficiency_score(
            current_power, performance_metric, batch_time)
        
        # Get recommendations
        recommendations = self._get_optimization_recommendations(
            efficiency_score, batch_size, current_power)
            
        # Update stats
        with self.lock:
            if recommendations["should_optimize"]:
                self.stats["optimizations_made"] += 1
                
        return {
            "efficiency_score": efficiency_score,
            "samples_per_second": samples_per_second,
            "recommendations": recommendations
        }
        
    def _get_optimization_recommendations(self, efficiency_score: float, 
                                        current_batch_size: int, 
                                        current_power: float) -> Dict[str, Any]:
        """
        Get optimization recommendations based on efficiency score.
        
        Args:
            efficiency_score: Current efficiency score
            current_batch_size: Current batch size
            current_power: Current power consumption
            
        Returns:
            Dict with optimization recommendations
        """
        recommendations = {
            "should_optimize": False,
            "actions": [],
            "batch_size_change": 0,
            "precision_change": None,
            "frequency_change": 0.0
        }
        
        # Get historical average efficiency
        with self.lock:
            if len(self.efficiency_scores) > 10:
                recent_scores = [score for _, score in list(self.efficiency_scores)[-10:]]
                avg_efficiency = sum(recent_scores) / len(recent_scores)
            else:
                avg_efficiency = efficiency_score
                
        # Check if optimization is needed
        if efficiency_score < avg_efficiency * 0.8:  # 20% below average
            recommendations["should_optimize"] = True
            
            # Determine optimization actions
            if current_batch_size > 16:  # Don't go too small
                recommendations["actions"].append("reduce_batch_size")
                recommendations["batch_size_change"] = -max(1, current_batch_size // 8)
                
            # Consider precision reduction if power is high
            if current_power > 100:  # Watts
                recommendations["actions"].append("reduce_precision")
                recommendations["precision_change"] = "fp16" if current_power > 150 else "fp8"
                
            # Consider frequency reduction if still inefficient
            if efficiency_score < avg_efficiency * 0.6:  # 40% below average
                recommendations["actions"].append("reduce_frequency")
                recommendations["frequency_change"] = -0.1  # 10% reduction
                
        elif efficiency_score > avg_efficiency * 1.2:  # 20% above average
            # We're doing well, could potentially increase performance
            recommendations["actions"].append("increase_performance")
            if current_batch_size < 512:  # Don't go too large
                recommendations["batch_size_change"] = max(1, current_batch_size // 4)
                
        return recommendations
        
    def record_power_savings(self, baseline_power: float, optimized_power: float, 
                           time_period_hours: float) -> float:
        """
        Record power savings from optimization.
        
        Args:
            baseline_power: Baseline power consumption in watts
            optimized_power: Optimized power consumption in watts
            time_period_hours: Time period in hours
            
        Returns:
            float: Power savings in watt-hours
        """
        power_savings = (baseline_power - optimized_power) * time_period_hours
        
        with self.lock:
            self.stats["power_savings_wh"] += power_savings
            if power_savings > 0:
                self.stats["efficiency_improvements"] += 1
                
        if power_savings > 0:
            print(f"💰 Power savings: {power_savings:.2f} Wh "
                  f"({((baseline_power - optimized_power) / baseline_power * 100):.1f}% reduction)")
                  
        return power_savings
        
    def get_stats(self) -> Dict[str, Any]:
        """Get energy controller statistics"""
        with self.lock:
            return self.stats.copy()
            
    def clear_stats(self):
        """Clear statistics"""
        with self.lock:
            self.stats = {
                "efficiency_improvements": 0,
                "optimizations_made": 0,
                "power_savings_wh": 0.0
            }
        print("🗑️  Energy controller statistics cleared")
        
    def get_efficiency_trend(self) -> List[Tuple[float, float]]:
        """
        Get efficiency score trend.
        
        Returns:
            List of (timestamp, efficiency_score) tuples
        """
        with self.lock:
            return list(self.efficiency_scores)

# Global instance
_energy_controller = None

def get_energy_controller() -> EnergyController:
    """Get the global energy controller instance"""
    global _energy_controller
    if _energy_controller is None:
        _energy_controller = EnergyController()
    return _energy_controller

if __name__ == "__main__":
    # Example usage
    controller = get_energy_controller()
    
    # Simulate some optimization steps
    result1 = controller.optimize_step(batch_time=2.5, batch_size=32, current_power=120.0)
    print(f"Optimization result 1: {result1}")
    
    result2 = controller.optimize_step(batch_time=2.0, batch_size=32, current_power=100.0)
    print(f"Optimization result 2: {result2}")
    
    # Record power savings
    controller.record_power_savings(baseline_power=120.0, optimized_power=100.0, time_period_hours=1.0)
    
    # Print stats
    print(f"📊 Stats: {controller.get_stats()}")