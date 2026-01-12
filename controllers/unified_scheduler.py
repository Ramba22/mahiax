"""
Unified Scheduler Pipeline for MAHIA-X
This module provides a consolidated interface for all training schedulers including:
- Learning Rate Scheduler
- Precision Scheduler
- Curriculum Scheduler
"""

import time
from typing import Optional, Dict, Any, List, Callable
import math

class BaseScheduler:
    """Base class for all schedulers"""
    
    def __init__(self, name: str):
        self.name = name
        self.enabled = True
        self.history = []
        
    def enable(self):
        """Enable the scheduler"""
        self.enabled = True
        
    def disable(self):
        """Disable the scheduler"""
        self.enabled = False
        
    def is_enabled(self) -> bool:
        """Check if scheduler is enabled"""
        return self.enabled
        
    def get_name(self) -> str:
        """Get scheduler name"""
        return self.name
        
    def get_history(self) -> List:
        """Get scheduler history"""
        return self.history.copy()
        
    def reset(self):
        """Reset scheduler state"""
        self.history = []


class LRScheduler(BaseScheduler):
    """Unified Learning Rate Scheduler with multiple scheduling strategies"""
    
    def __init__(self, initial_lr: float = 1e-3, min_lr: float = 1e-7, 
                 max_lr: float = 1e-2, scheduler_type: str = "cosine"):
        super().__init__("LRScheduler")
        self.initial_lr = initial_lr
        self.current_lr = initial_lr
        self.min_lr = min_lr
        self.max_lr = max_lr
        self.scheduler_type = scheduler_type
        
        # For cosine annealing
        self.cosine_t_max = 100
        self.cosine_eta_min = min_lr
        
        # For step decay
        self.step_size = 30
        self.gamma = 0.1
        
        # For exponential decay
        self.decay_rate = 0.95
        
        # For warmup
        self.warmup_steps = 1000
        self.warmup_initialized = False
        
        # For adaptive scheduling
        self.adaptive_factor = 0.5
        self.patience = 10
        self.best_metric = float('inf')
        self.waiting_steps = 0
        
    def step(self, step: int = 0, epoch: int = 0, metric: Optional[float] = None) -> float:
        """Update learning rate based on scheduler type and current step
        
        Args:
            step: Current training step
            epoch: Current training epoch
            metric: Current validation metric (for adaptive scheduling)
            
        Returns:
            float: Updated learning rate
        """
        if step < self.warmup_steps and not self.warmup_initialized:
            # Warmup phase
            warmup_lr = self.initial_lr * (step / self.warmup_steps)
            self.current_lr = max(warmup_lr, self.min_lr)
        else:
            self.warmup_initialized = True
            # Apply main scheduling strategy
            if self.scheduler_type == "cosine":
                self._cosine_annealing(step, epoch)
            elif self.scheduler_type == "step":
                self._step_decay(step, epoch)
            elif self.scheduler_type == "exponential":
                self._exponential_decay(step, epoch)
            elif self.scheduler_type == "adaptive" and metric is not None:
                self._adaptive_scheduling(step, epoch, metric)
            # For other types, keep current LR
                
        # Ensure LR stays within bounds
        self.current_lr = max(min(self.current_lr, self.max_lr), self.min_lr)
        
        # Store in history
        self.history.append({
            "step": step,
            "epoch": epoch,
            "lr": self.current_lr,
            "type": self.scheduler_type
        })
        
        return self.current_lr
        
    def _cosine_annealing(self, step: int, epoch: int):
        """Cosine annealing scheduling"""
        # Calculate current cycle position
        cycle_position = step % self.cosine_t_max
        # Calculate cosine annealed LR
        cosine_factor = (1 + math.cos(math.pi * cycle_position / self.cosine_t_max)) / 2
        self.current_lr = self.cosine_eta_min + (self.initial_lr - self.cosine_eta_min) * cosine_factor
        
    def _step_decay(self, step: int, epoch: int):
        """Step decay scheduling"""
        # Calculate number of decays
        num_decays = epoch // self.step_size
        self.current_lr = self.initial_lr * (self.gamma ** num_decays)
        
    def _exponential_decay(self, step: int, epoch: int):
        """Exponential decay scheduling"""
        self.current_lr = self.initial_lr * (self.decay_rate ** epoch)
        
    def _adaptive_scheduling(self, step: int, epoch: int, metric: float):
        """Adaptive scheduling based on validation metric"""
        if metric < self.best_metric:
            self.best_metric = metric
            self.waiting_steps = 0
        else:
            self.waiting_steps += 1
            if self.waiting_steps >= self.patience:
                self.current_lr *= self.adaptive_factor
                self.waiting_steps = 0
                print(f"📉 {self.name}: Reducing LR to {self.current_lr:.2e} due to plateau")
                
    def set_scheduler_type(self, scheduler_type: str):
        """Change scheduler type"""
        self.scheduler_type = scheduler_type
        
    def get_status(self) -> Dict[str, Any]:
        """Get scheduler status"""
        return {
            "name": self.name,
            "enabled": self.enabled,
            "current_lr": self.current_lr,
            "initial_lr": self.initial_lr,
            "min_lr": self.min_lr,
            "max_lr": self.max_lr,
            "scheduler_type": self.scheduler_type,
            "warmup_steps": self.warmup_steps,
            "warmup_initialized": self.warmup_initialized,
            "history_length": len(self.history)
        }


class PrecisionScheduler(BaseScheduler):
    """Unified Precision Scheduler for mixed precision training"""
    
    def __init__(self, initial_precision: str = "fp32", precision_threshold: float = 0.01):
        super().__init__("PrecisionScheduler")
        self.initial_precision = initial_precision
        self.current_precision = initial_precision
        self.precision_threshold = precision_threshold
        
        # Precision switching history
        self.switch_count = 0
        self.last_switch_step = 0
        self.stability_counter = 0
        
        # Performance tracking
        self.loss_history = []
        self.gradient_norm_history = []
        
    def step(self, step: int = 0, loss: Optional[float] = None, gradient_norm: Optional[float] = None, 
             force_precision: Optional[str] = None) -> str:
        """Update precision based on training dynamics
        
        Args:
            step: Current training step
            loss: Current loss value
            gradient_norm: Current gradient norm
            force_precision: Force a specific precision (optional)
            
        Returns:
            str: Current precision setting
        """
        # Store metrics for analysis
        if loss is not None:
            self.loss_history.append(loss)
        if gradient_norm is not None:
            self.gradient_norm_history.append(gradient_norm)
            
        # If precision is forced, use it
        if force_precision is not None:
            if force_precision != self.current_precision:
                self._switch_precision(force_precision, step, "forced")
            return self.current_precision
            
        # Keep only recent history
        max_history = 100
        if len(self.loss_history) > max_history:
            self.loss_history = self.loss_history[-max_history:]
        if len(self.gradient_norm_history) > max_history:
            self.gradient_norm_history = self.gradient_norm_history[-max_history:]
            
        # Analyze training stability
        if len(self.loss_history) >= 10:
            recent_losses = self.loss_history[-10:]
            loss_variance = self._compute_variance(recent_losses)
            
            # If loss is stable and we're using lower precision, consider upgrading
            if loss_variance < self.precision_threshold and self.current_precision == "fp16":
                if step - self.last_switch_step > 100:  # Avoid frequent switching
                    self._switch_precision("fp32", step, "stable_training")
            # If loss is unstable and we're using higher precision, consider downgrading
            elif loss_variance > self.precision_threshold * 10 and self.current_precision == "fp32":
                if step - self.last_switch_step > 100:  # Avoid frequent switching
                    self._switch_precision("fp16", step, "unstable_training")
                    
        # Store in history
        self.history.append({
            "step": step,
            "precision": self.current_precision,
            "loss": loss,
            "gradient_norm": gradient_norm
        })
        
        return self.current_precision
        
    def _compute_variance(self, values: List[float]) -> float:
        """Compute variance of a list of values"""
        if len(values) < 2:
            return 0.0
        mean = sum(values) / len(values)
        variance = sum((x - mean) ** 2 for x in values) / len(values)
        return variance
        
    def _switch_precision(self, new_precision: str, step: int, reason: str):
        """Switch to a new precision setting"""
        old_precision = self.current_precision
        self.current_precision = new_precision
        self.switch_count += 1
        self.last_switch_step = step
        self.stability_counter = 0
        
        print(f"⚡ {self.name}: Switching precision from {old_precision} to {new_precision} at step {step} ({reason})")
        
    def get_status(self) -> Dict[str, Any]:
        """Get scheduler status"""
        return {
            "name": self.name,
            "enabled": self.enabled,
            "current_precision": self.current_precision,
            "initial_precision": self.initial_precision,
            "switch_count": self.switch_count,
            "last_switch_step": self.last_switch_step,
            "loss_history_length": len(self.loss_history),
            "gradient_norm_history_length": len(self.gradient_norm_history)
        }


class CurriculumScheduler(BaseScheduler):
    """Unified Curriculum Scheduler for adaptive difficulty adjustment"""
    
    def __init__(self, initial_difficulty: float = 0.3, min_difficulty: float = 0.1, 
                 max_difficulty: float = 1.0, adjustment_factor: float = 0.1):
        super().__init__("CurriculumScheduler")
        self.current_difficulty = initial_difficulty
        self.min_difficulty = min_difficulty
        self.max_difficulty = max_difficulty
        self.adjustment_factor = adjustment_factor
        
        # Curriculum adjustment history
        self.difficulty_history = [initial_difficulty]
        self.performance_history = []
        self.entropy_history = []
        
        # Performance tracking
        self.best_performance = float('-inf')
        self.worst_performance = float('inf')
        
        # Adaptive parameters
        self.window_size = 10
        self.improvement_threshold = 0.01
        self.degradation_threshold = -0.005
        
    def step(self, step: int = 0, epoch: int = 0, performance: Optional[float] = None, 
             entropy: Optional[float] = None, force_difficulty: Optional[float] = None) -> float:
        """Update curriculum difficulty based on training performance
        
        Args:
            step: Current training step
            epoch: Current training epoch
            performance: Current performance metric
            entropy: Current entropy metric
            force_difficulty: Force a specific difficulty (optional)
            
        Returns:
            float: Current difficulty level
        """
        # If difficulty is forced, use it
        if force_difficulty is not None:
            self.current_difficulty = max(min(force_difficulty, self.max_difficulty), self.min_difficulty)
            self.difficulty_history.append(self.current_difficulty)
            return self.current_difficulty
            
        # Store metrics
        if performance is not None:
            self.performance_history.append(performance)
            # Update best/worst performance
            self.best_performance = max(self.best_performance, performance)
            self.worst_performance = min(self.worst_performance, performance)
            
        if entropy is not None:
            self.entropy_history.append(entropy)
            
        # Keep only recent history
        max_history = 1000
        if len(self.performance_history) > max_history:
            self.performance_history = self.performance_history[-max_history:]
        if len(self.entropy_history) > max_history:
            self.entropy_history = self.entropy_history[-max_history:]
            
        # Analyze recent performance trend
        if len(self.performance_history) >= self.window_size:
            recent_performance = self.performance_history[-self.window_size:]
            performance_trend = self._compute_trend(recent_performance)
            
            # Adjust difficulty based on performance trend
            if performance_trend > self.improvement_threshold:
                # Performance is improving, increase difficulty
                new_difficulty = min(self.current_difficulty + self.adjustment_factor, self.max_difficulty)
                if new_difficulty != self.current_difficulty:
                    self.current_difficulty = new_difficulty
                    print(f"📚 {self.name}: Increasing difficulty to {self.current_difficulty:.2f} (improving performance)")
            elif performance_trend < self.degradation_threshold:
                # Performance is degrading, decrease difficulty
                new_difficulty = max(self.current_difficulty - self.adjustment_factor, self.min_difficulty)
                if new_difficulty != self.current_difficulty:
                    self.current_difficulty = new_difficulty
                    print(f"📚 {self.name}: Decreasing difficulty to {self.current_difficulty:.2f} (degrading performance)")
                    
        # Store in history
        self.difficulty_history.append(self.current_difficulty)
        self.history.append({
            "step": step,
            "epoch": epoch,
            "difficulty": self.current_difficulty,
            "performance": performance,
            "entropy": entropy
        })
        
        return self.current_difficulty
        
    def _compute_trend(self, values: List[float]) -> float:
        """Compute linear trend of a list of values"""
        if len(values) < 2:
            return 0.0
            
        n = len(values)
        x = list(range(n))
        y = values
        
        # Compute means
        mean_x = sum(x) / n
        mean_y = sum(y) / n
        
        # Compute slope using least squares
        numerator = sum((x[i] - mean_x) * (y[i] - mean_y) for i in range(n))
        denominator = sum((x[i] - mean_x) ** 2 for i in range(n))
        
        if denominator > 1e-12:
            slope = numerator / denominator
        else:
            slope = 0.0
            
        return slope
        
    def get_difficulty_trend(self, window_size: int = 10) -> Dict[str, Any]:
        """Get difficulty trend analysis"""
        if len(self.difficulty_history) < 2:
            return {}
            
        # Get recent records
        recent_records = self.difficulty_history[-min(window_size, len(self.difficulty_history)):]
        
        # Compute trend using linear regression
        if len(recent_records) >= 2:
            trend = self._compute_trend(recent_records)
            return {
                "trend": trend,
                "current_difficulty": recent_records[-1] if recent_records else 0.0,
                "avg_difficulty": sum(recent_records) / len(recent_records) if recent_records else 0.0,
                "std_difficulty": self._compute_std(recent_records) if len(recent_records) > 1 else 0.0
            }
            
        return {}
        
    def _compute_std(self, values: List[float]) -> float:
        """Compute standard deviation of a list of values"""
        if len(values) < 2:
            return 0.0
        mean = sum(values) / len(values)
        variance = sum((x - mean) ** 2 for x in values) / len(values)
        return variance ** 0.5
        
    def get_status(self) -> Dict[str, Any]:
        """Get scheduler status"""
        trend = self.get_difficulty_trend()
        
        return {
            "name": self.name,
            "enabled": self.enabled,
            "current_difficulty": self.current_difficulty,
            "min_difficulty": self.min_difficulty,
            "max_difficulty": self.max_difficulty,
            "adjustment_factor": self.adjustment_factor,
            "difficulty_history_length": len(self.difficulty_history),
            "performance_history_length": len(self.performance_history),
            "entropy_history_length": len(self.entropy_history),
            "best_performance": self.best_performance,
            "worst_performance": self.worst_performance,
            "difficulty_trend": trend.get("trend", 0.0) if trend else 0.0
        }


class UnifiedSchedulerPipeline:
    """Main unified interface that consolidates all schedulers"""
    
    def __init__(self):
        # Initialize all schedulers
        self.lr_scheduler = LRScheduler()
        self.precision_scheduler = PrecisionScheduler()
        self.curriculum_scheduler = CurriculumScheduler()
        
        # Scheduler registry
        self.schedulers = {
            "lr": self.lr_scheduler,
            "precision": self.precision_scheduler,
            "curriculum": self.curriculum_scheduler
        }
        
        # Synchronization settings
        self.sync_schedulers = True
        self.coordinated_adjustments = []
        
    def get_scheduler(self, name: str) -> Optional[BaseScheduler]:
        """Get scheduler by name"""
        return self.schedulers.get(name)
        
    def enable_scheduler(self, name: str):
        """Enable scheduler by name"""
        if name in self.schedulers:
            self.schedulers[name].enable()
            
    def disable_scheduler(self, name: str):
        """Disable scheduler by name"""
        if name in self.schedulers:
            self.schedulers[name].disable()
            
    def get_all_schedulers(self) -> Dict[str, BaseScheduler]:
        """Get all schedulers"""
        return self.schedulers.copy()
        
    def step_all(self, step: int = 0, epoch: int = 0, metrics: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Step all schedulers and return current settings
        
        Args:
            step: Current training step
            epoch: Current training epoch
            metrics: Dictionary of current training metrics
            
        Returns:
            dict: Current settings from all schedulers
        """
        if metrics is None:
            metrics = {}
            
        # Extract metrics
        loss = metrics.get("loss")
        metric = metrics.get("metric")
        gradient_norm = metrics.get("gradient_norm")
        performance = metrics.get("performance")
        entropy = metrics.get("entropy")
        
        # Step all schedulers
        current_settings = {}
        
        # Step LR scheduler
        if self.lr_scheduler.is_enabled():
            lr = self.lr_scheduler.step(step, epoch, metric)
            current_settings["lr"] = lr
            
        # Step precision scheduler
        if self.precision_scheduler.is_enabled():
            precision = self.precision_scheduler.step(step, loss, gradient_norm)
            current_settings["precision"] = precision
            
        # Step curriculum scheduler
        if self.curriculum_scheduler.is_enabled():
            difficulty = self.curriculum_scheduler.step(step, epoch, performance, entropy)
            current_settings["difficulty"] = difficulty
            
        # Store coordinated adjustment
        self.coordinated_adjustments.append({
            "step": step,
            "epoch": epoch,
            "settings": current_settings.copy(),
            "metrics": metrics.copy()
        })
        
        # Keep only recent adjustments
        if len(self.coordinated_adjustments) > 1000:
            self.coordinated_adjustments.pop(0)
            
        return current_settings
        
    def force_settings(self, step: int = 0, settings: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Force specific settings across all schedulers
        
        Args:
            step: Current training step
            settings: Dictionary of settings to force
            
        Returns:
            dict: Current settings after forcing
        """
        if settings is None:
            settings = {}
            
        current_settings = {}
        
        # Force LR if specified
        if "lr" in settings and self.lr_scheduler.is_enabled():
            # Note: LR scheduler doesn't directly support forcing, but we can adjust its state
            target_lr = settings["lr"]
            self.lr_scheduler.current_lr = target_lr
            current_settings["lr"] = target_lr
            
        # Force precision if specified
        if "precision" in settings and self.precision_scheduler.is_enabled():
            precision = self.precision_scheduler.step(step, force_precision=settings["precision"])
            current_settings["precision"] = precision
            
        # Force difficulty if specified
        if "difficulty" in settings and self.curriculum_scheduler.is_enabled():
            difficulty = self.curriculum_scheduler.step(step, force_difficulty=settings["difficulty"])
            current_settings["difficulty"] = difficulty
            
        return current_settings
        
    def get_scheduler_status(self) -> Dict[str, Any]:
        """Get status of all schedulers"""
        status = {}
        for name, scheduler in self.schedulers.items():
            if hasattr(scheduler, 'get_status'):
                status[name] = scheduler.get_status()
            else:
                status[name] = {"name": scheduler.get_name(), "enabled": scheduler.is_enabled()}
        return status
        
    def get_coordinated_adjustments(self) -> List[Dict[str, Any]]:
        """Get history of coordinated adjustments"""
        return self.coordinated_adjustments.copy()
        
    def reset_all_schedulers(self):
        """Reset all schedulers"""
        for scheduler in self.schedulers.values():
            scheduler.reset()
        self.coordinated_adjustments = []
            
    def synchronize_schedulers(self, sync: bool = True):
        """Enable or disable scheduler synchronization"""
        self.sync_schedulers = sync


# Example usage
if __name__ == "__main__":
    # Create unified scheduler pipeline
    scheduler_pipeline = UnifiedSchedulerPipeline()
    
    # Example training loop integration
    print("Unified Scheduler Pipeline for MAHIA-X")
    print("=" * 40)
    
    # Simulate training loop
    for epoch in range(5):
        print(f"\nEpoch {epoch + 1}:")
        
        # Simulate multiple steps per epoch
        for step in range(10):
            global_step = epoch * 10 + step
            
            # Simulate training metrics
            metrics = {
                "loss": 1.0 - (global_step * 0.01),  # Decreasing loss
                "metric": 0.5 + (global_step * 0.005),  # Increasing metric
                "gradient_norm": 1.0 - (global_step * 0.001),  # Decreasing gradient norm
                "performance": 0.6 + (global_step * 0.003),  # Increasing performance
                "entropy": 1.0 - (global_step * 0.002)  # Decreasing entropy
            }
            
            # Step all schedulers
            settings = scheduler_pipeline.step_all(global_step, epoch, metrics)
            
            # Print settings every few steps
            if step % 5 == 0:
                print(f"  Step {global_step}:")
                for key, value in settings.items():
                    if key == "lr":
                        print(f"    LR: {value:.2e}")
                    else:
                        print(f"    {key.capitalize()}: {value:.4f}")
    
    # Print final status
    print("\nFinal Scheduler Status:")
    status = scheduler_pipeline.get_scheduler_status()
    for scheduler_name, scheduler_status in status.items():
        print(f"  {scheduler_name}:")
        for key, value in scheduler_status.items():
            if key == "current_lr":
                print(f"    {key}: {value:.2e}")
            else:
                print(f"    {key}: {value}")