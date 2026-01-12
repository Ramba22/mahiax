"""
Standardized Telemetry API for MAHIA-X
This module provides a unified interface for logging all training metrics including:
- Controller events
- Entropy values
- GPU data
- Performance metrics
"""

import time
import json
import os
from typing import Optional, Dict, Any, List, Union
from datetime import datetime
import threading

class TelemetryEvent:
    """Standardized telemetry event structure"""
    
    def __init__(self, event_type: str, data: Dict[str, Any], timestamp: Optional[float] = None):
        self.event_type = event_type
        self.data = data
        self.timestamp = timestamp or time.time()
        self.id = self._generate_id()
        
    def _generate_id(self) -> str:
        """Generate unique event ID"""
        return f"{self.event_type}_{int(self.timestamp * 1000000)}"
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert event to dictionary"""
        return {
            "id": self.id,
            "event_type": self.event_type,
            "timestamp": self.timestamp,
            "data": self.data
        }
        
    def to_json(self) -> str:
        """Convert event to JSON string"""
        return json.dumps(self.to_dict())


class StandardizedTelemetryAPI:
    """Main telemetry API with standardized logging interface"""
    
    def __init__(self, log_dir: str = "./logs", experiment_name: str = "mahia_experiment"):
        self.log_dir = log_dir
        self.experiment_name = experiment_name
        self.session_id = self._generate_session_id()
        self.start_time = time.time()
        
        # Create log directory
        os.makedirs(log_dir, exist_ok=True)
        
        # Event storage
        self.events = []
        self.event_lock = threading.Lock()
        
        # Log files
        self.log_files = {}
        self._initialize_log_files()
        
        # Active loggers
        self.active_loggers = {
            "controller_events": True,
            "entropy_values": True,
            "gpu_data": True,
            "performance_metrics": True,
            "learning_rates": True,
            "precision_changes": True,
            "curriculum_data": True,
            "checkpoint_events": True,
            "error_events": True
        }
        
        # Print initialization message
        print(f"📊 Telemetry API initialized for experiment: {self.experiment_name}")
        print(f"   Session ID: {self.session_id}")
        print(f"   Log directory: {self.log_dir}")
        
    def _generate_session_id(self) -> str:
        """Generate unique session ID"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"{self.experiment_name}_{timestamp}"
        
    def _initialize_log_files(self):
        """Initialize CSV log files for different event types"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        self.log_files = {
            "controller_events": os.path.join(self.log_dir, f"controller_events_{timestamp}.csv"),
            "entropy_values": os.path.join(self.log_dir, f"entropy_values_{timestamp}.csv"),
            "gpu_data": os.path.join(self.log_dir, f"gpu_data_{timestamp}.csv"),
            "performance_metrics": os.path.join(self.log_dir, f"performance_metrics_{timestamp}.csv"),
            "learning_rates": os.path.join(self.log_dir, f"learning_rates_{timestamp}.csv"),
            "precision_changes": os.path.join(self.log_dir, f"precision_changes_{timestamp}.csv"),
            "curriculum_data": os.path.join(self.log_dir, f"curriculum_data_{timestamp}.csv"),
            "checkpoint_events": os.path.join(self.log_dir, f"checkpoint_events_{timestamp}.csv"),
            "error_events": os.path.join(self.log_dir, f"error_events_{timestamp}.csv"),
            "all_events": os.path.join(self.log_dir, f"all_events_{timestamp}.json")
        }
        
        # Create headers for CSV files
        self._write_csv_header("controller_events", ["timestamp", "step", "epoch", "controller", "action", "confidence", "details"])
        self._write_csv_header("entropy_values", ["timestamp", "step", "epoch", "entropy", "gradient_norm", "confidence", "type"])
        self._write_csv_header("gpu_data", ["timestamp", "step", "epoch", "gpu_utilization", "memory_used_mb", "temperature_c", "power_watts"])
        self._write_csv_header("performance_metrics", ["timestamp", "step", "epoch", "loss", "metric", "accuracy", "f1_score"])
        self._write_csv_header("learning_rates", ["timestamp", "step", "epoch", "lr", "scheduler", "reason"])
        self._write_csv_header("precision_changes", ["timestamp", "step", "epoch", "precision", "reason", "stability"])
        self._write_csv_header("curriculum_data", ["timestamp", "step", "epoch", "difficulty", "performance", "entropy", "adjustment"])
        self._write_csv_header("checkpoint_events", ["timestamp", "step", "epoch", "event_type", "path", "is_best", "size_mb"])
        self._write_csv_header("error_events", ["timestamp", "step", "epoch", "error_type", "message", "severity", "traceback"])
        
    def _write_csv_header(self, log_type: str, headers: List[str]):
        """Write CSV header to log file"""
        try:
            with open(self.log_files[log_type], 'w') as f:
                f.write(','.join(headers) + '\n')
        except Exception as e:
            print(f"⚠️  Failed to write header for {log_type}: {e}")
            
    def _append_to_csv(self, log_type: str, data: List[str]):
        """Append data to CSV log file"""
        try:
            with open(self.log_files[log_type], 'a') as f:
                f.write(','.join(map(str, data)) + '\n')
        except Exception as e:
            print(f"⚠️  Failed to append to {log_type}: {e}")
            
    def enable_logger(self, logger_name: str):
        """Enable specific logger"""
        if logger_name in self.active_loggers:
            self.active_loggers[logger_name] = True
            print(f"✅ Enabled logger: {logger_name}")
            
    def disable_logger(self, logger_name: str):
        """Disable specific logger"""
        if logger_name in self.active_loggers:
            self.active_loggers[logger_name] = False
            print(f"🚫 Disabled logger: {logger_name}")
            
    def log_controller_event(self, step: int, epoch: int, controller: str, 
                           action: str, confidence: float = None, details: Dict[str, Any] = None):
        """Log controller event"""
        if not self.active_loggers.get("controller_events", True):
            return
            
        event_data = {
            "step": step,
            "epoch": epoch,
            "controller": controller,
            "action": action,
            "confidence": confidence,
            "details": details or {}
        }
        
        event = TelemetryEvent("controller_event", event_data)
        self._store_event(event)
        
        # Log to CSV
        csv_data = [
            event.timestamp,
            step,
            epoch,
            controller,
            action,
            confidence or "",
            json.dumps(details or {})
        ]
        self._append_to_csv("controller_events", csv_data)
        
    def log_entropy_value(self, step: int, epoch: int, entropy: float, 
                         gradient_norm: float = None, confidence: float = None, 
                         entropy_type: str = "gradient"):
        """Log entropy value"""
        if not self.active_loggers.get("entropy_values", True):
            return
            
        event_data = {
            "step": step,
            "epoch": epoch,
            "entropy": entropy,
            "gradient_norm": gradient_norm,
            "confidence": confidence,
            "type": entropy_type
        }
        
        event = TelemetryEvent("entropy_value", event_data)
        self._store_event(event)
        
        # Log to CSV
        csv_data = [
            event.timestamp,
            step,
            epoch,
            entropy,
            gradient_norm or "",
            confidence or "",
            entropy_type
        ]
        self._append_to_csv("entropy_values", csv_data)
        
    def log_gpu_data(self, step: int, epoch: int, gpu_utilization: float, 
                    memory_used_mb: float, temperature_c: float, power_watts: float = None):
        """Log GPU data"""
        if not self.active_loggers.get("gpu_data", True):
            return
            
        event_data = {
            "step": step,
            "epoch": epoch,
            "gpu_utilization": gpu_utilization,
            "memory_used_mb": memory_used_mb,
            "temperature_c": temperature_c,
            "power_watts": power_watts
        }
        
        event = TelemetryEvent("gpu_data", event_data)
        self._store_event(event)
        
        # Log to CSV
        csv_data = [
            event.timestamp,
            step,
            epoch,
            gpu_utilization,
            memory_used_mb,
            temperature_c,
            power_watts or ""
        ]
        self._append_to_csv("gpu_data", csv_data)
        
    def log_performance_metric(self, step: int, epoch: int, loss: float = None, 
                              metric: float = None, accuracy: float = None, 
                              f1_score: float = None):
        """Log performance metric"""
        if not self.active_loggers.get("performance_metrics", True):
            return
            
        event_data = {
            "step": step,
            "epoch": epoch,
            "loss": loss,
            "metric": metric,
            "accuracy": accuracy,
            "f1_score": f1_score
        }
        
        event = TelemetryEvent("performance_metric", event_data)
        self._store_event(event)
        
        # Log to CSV
        csv_data = [
            event.timestamp,
            step,
            epoch,
            loss or "",
            metric or "",
            accuracy or "",
            f1_score or ""
        ]
        self._append_to_csv("performance_metrics", csv_data)
        
    def log_learning_rate(self, step: int, epoch: int, lr: float, 
                         scheduler: str = "default", reason: str = "scheduled"):
        """Log learning rate change"""
        if not self.active_loggers.get("learning_rates", True):
            return
            
        event_data = {
            "step": step,
            "epoch": epoch,
            "lr": lr,
            "scheduler": scheduler,
            "reason": reason
        }
        
        event = TelemetryEvent("learning_rate", event_data)
        self._store_event(event)
        
        # Log to CSV
        csv_data = [
            event.timestamp,
            step,
            epoch,
            lr,
            scheduler,
            reason
        ]
        self._append_to_csv("learning_rates", csv_data)
        
    def log_precision_change(self, step: int, epoch: int, precision: str, 
                            reason: str = "scheduled", stability: float = None):
        """Log precision change"""
        if not self.active_loggers.get("precision_changes", True):
            return
            
        event_data = {
            "step": step,
            "epoch": epoch,
            "precision": precision,
            "reason": reason,
            "stability": stability
        }
        
        event = TelemetryEvent("precision_change", event_data)
        self._store_event(event)
        
        # Log to CSV
        csv_data = [
            event.timestamp,
            step,
            epoch,
            precision,
            reason,
            stability or ""
        ]
        self._append_to_csv("precision_changes", csv_data)
        
    def log_curriculum_data(self, step: int, epoch: int, difficulty: float, 
                           performance: float = None, entropy: float = None, 
                           adjustment: str = "none"):
        """Log curriculum data"""
        if not self.active_loggers.get("curriculum_data", True):
            return
            
        event_data = {
            "step": step,
            "epoch": epoch,
            "difficulty": difficulty,
            "performance": performance,
            "entropy": entropy,
            "adjustment": adjustment
        }
        
        event = TelemetryEvent("curriculum_data", event_data)
        self._store_event(event)
        
        # Log to CSV
        csv_data = [
            event.timestamp,
            step,
            epoch,
            difficulty,
            performance or "",
            entropy or "",
            adjustment
        ]
        self._append_to_csv("curriculum_data", csv_data)
        
    def log_checkpoint_event(self, step: int, epoch: int, event_type: str, 
                            path: str = None, is_best: bool = False, size_mb: float = None):
        """Log checkpoint event"""
        if not self.active_loggers.get("checkpoint_events", True):
            return
            
        event_data = {
            "step": step,
            "epoch": epoch,
            "event_type": event_type,
            "path": path,
            "is_best": is_best,
            "size_mb": size_mb
        }
        
        event = TelemetryEvent("checkpoint_event", event_data)
        self._store_event(event)
        
        # Log to CSV
        csv_data = [
            event.timestamp,
            step,
            epoch,
            event_type,
            path or "",
            is_best,
            size_mb or ""
        ]
        self._append_to_csv("checkpoint_events", csv_data)
        
    def log_error_event(self, step: int, epoch: int, error_type: str, 
                       message: str, severity: str = "warning", traceback: str = None):
        """Log error event"""
        if not self.active_loggers.get("error_events", True):
            return
            
        event_data = {
            "step": step,
            "epoch": epoch,
            "error_type": error_type,
            "message": message,
            "severity": severity,
            "traceback": traceback
        }
        
        event = TelemetryEvent("error_event", event_data)
        self._store_event(event)
        
        # Log to CSV
        csv_data = [
            event.timestamp,
            step,
            epoch,
            error_type,
            message,
            severity,
            traceback or ""
        ]
        self._append_to_csv("error_events", csv_data)
        
        # Print error message
        severity_icon = {"info": "ℹ️", "warning": "⚠️", "error": "❌", "critical": "🚨"}
        icon = severity_icon.get(severity, "ℹ️")
        print(f"{icon} {error_type}: {message}")
        
    def _store_event(self, event: TelemetryEvent):
        """Store event in memory"""
        with self.event_lock:
            self.events.append(event)
            
            # Keep only recent events in memory
            if len(self.events) > 10000:
                self.events = self.events[-5000:]
                
    def get_events(self, event_type: str = None, limit: int = None) -> List[TelemetryEvent]:
        """Get stored events"""
        with self.event_lock:
            filtered_events = self.events
            if event_type:
                filtered_events = [e for e in filtered_events if e.event_type == event_type]
                
            if limit:
                filtered_events = filtered_events[-limit:]
                
            return filtered_events.copy()
            
    def get_events_by_time_range(self, start_time: float, end_time: float) -> List[TelemetryEvent]:
        """Get events within a time range"""
        with self.event_lock:
            filtered_events = [
                e for e in self.events 
                if start_time <= e.timestamp <= end_time
            ]
            return filtered_events
            
    def save_all_events(self, filepath: str = None):
        """Save all events to JSON file"""
        if filepath is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filepath = os.path.join(self.log_dir, f"all_events_{timestamp}.json")
            
        try:
            with self.event_lock:
                events_data = [e.to_dict() for e in self.events]
                
            session_info = {
                "experiment_name": self.experiment_name,
                "session_id": self.session_id,
                "start_time": self.start_time,
                "end_time": time.time(),
                "total_events": len(events_data)
            }
            
            full_data = {
                "session_info": session_info,
                "events": events_data
            }
            
            with open(filepath, 'w') as f:
                json.dump(full_data, f, indent=2)
                
            print(f"✅ Saved {len(events_data)} events to {filepath}")
            return filepath
        except Exception as e:
            print(f"❌ Failed to save events: {e}")
            return None
            
    def load_events(self, filepath: str):
        """Load events from JSON file"""
        try:
            with open(filepath, 'r') as f:
                full_data = json.load(f)
                
            events_data = full_data.get("events", [])
            loaded_events = []
            
            for event_dict in events_data:
                event = TelemetryEvent(
                    event_type=event_dict["event_type"],
                    data=event_dict["data"],
                    timestamp=event_dict["timestamp"]
                )
                event.id = event_dict["id"]
                loaded_events.append(event)
                
            with self.event_lock:
                self.events.extend(loaded_events)
                
            print(f"✅ Loaded {len(loaded_events)} events from {filepath}")
            return len(loaded_events)
        except Exception as e:
            print(f"❌ Failed to load events: {e}")
            return 0
            
    def get_summary(self) -> Dict[str, Any]:
        """Get telemetry summary"""
        with self.event_lock:
            event_counts = {}
            for event in self.events:
                event_type = event.event_type
                event_counts[event_type] = event_counts.get(event_type, 0) + 1
                
            # Get time range
            if self.events:
                timestamps = [e.timestamp for e in self.events]
                start_time = min(timestamps)
                end_time = max(timestamps)
                duration = end_time - start_time
            else:
                start_time = self.start_time
                end_time = time.time()
                duration = end_time - start_time
                
            return {
                "experiment_name": self.experiment_name,
                "session_id": self.session_id,
                "start_time": start_time,
                "end_time": end_time,
                "duration_seconds": duration,
                "total_events": len(self.events),
                "event_counts": event_counts,
                "active_loggers": self.active_loggers.copy(),
                "log_files": self.log_files.copy()
            }
            
    def print_summary(self):
        """Print telemetry summary"""
        summary = self.get_summary()
        
        print("\n" + "="*50)
        print("TELEMETRY SUMMARY")
        print("="*50)
        print(f"Experiment: {summary['experiment_name']}")
        print(f"Session ID: {summary['session_id']}")
        print(f"Duration: {summary['duration_seconds']:.2f} seconds")
        print(f"Total Events: {summary['total_events']}")
        print("\nEvent Counts:")
        for event_type, count in summary['event_counts'].items():
            print(f"  {event_type}: {count}")
        print("\nActive Loggers:")
        for logger, active in summary['active_loggers'].items():
            status = "✅" if active else "🚫"
            print(f"  {status} {logger}")
        print("="*50)


# Example usage
if __name__ == "__main__":
    # Create telemetry API
    telemetry = StandardizedTelemetryAPI(experiment_name="mahia_test_experiment")
    
    # Simulate logging various events
    print("Testing Standardized Telemetry API")
    print("=" * 40)
    
    # Log some sample events
    for step in range(10):
        # Log controller events
        telemetry.log_controller_event(
            step=step,
            epoch=step // 5,
            controller="ExtendStop",
            action="waiting",
            confidence=0.85,
            details={"loss": 0.5 - step * 0.05}
        )
        
        # Log entropy values
        telemetry.log_entropy_value(
            step=step,
            epoch=step // 5,
            entropy=1.0 - step * 0.1,
            gradient_norm=0.1 + step * 0.01,
            confidence=0.9 - step * 0.02
        )
        
        # Log GPU data
        telemetry.log_gpu_data(
            step=step,
            epoch=step // 5,
            gpu_utilization=50 + step * 2,
            memory_used_mb=1024 + step * 100,
            temperature_c=60 + step,
            power_watts=150 + step * 2
        )
        
        # Log performance metrics
        telemetry.log_performance_metric(
            step=step,
            epoch=step // 5,
            loss=0.5 - step * 0.05,
            metric=0.7 + step * 0.03,
            accuracy=0.8 + step * 0.02
        )
        
        # Log learning rate changes
        if step % 3 == 0:
            telemetry.log_learning_rate(
                step=step,
                epoch=step // 5,
                lr=1e-3 * (0.9 ** (step // 3)),
                scheduler="cosine",
                reason="scheduled"
            )
        
        # Log precision changes
        if step == 5:
            telemetry.log_precision_change(
                step=step,
                epoch=step // 5,
                precision="fp16",
                reason="stability_improved",
                stability=0.95
            )
            
        # Log curriculum data
        telemetry.log_curriculum_data(
            step=step,
            epoch=step // 5,
            difficulty=0.3 + step * 0.05,
            performance=0.7 + step * 0.03,
            entropy=1.0 - step * 0.1,
            adjustment="none" if step % 2 == 0 else "increased"
        )
        
        # Log checkpoint events
        if step == 9:
            telemetry.log_checkpoint_event(
                step=step,
                epoch=step // 5,
                event_type="saved",
                path="./checkpoints/model_step_9.pt",
                is_best=True,
                size_mb=150.5
            )
            
        # Log error events (occasional)
        if step == 7:
            telemetry.log_error_event(
                step=step,
                epoch=step // 5,
                error_type="gradient_explosion",
                message="Gradient norm exceeded threshold",
                severity="warning",
                traceback="Traceback..."
            )
    
    # Print summary
    telemetry.print_summary()
    
    # Save all events
    save_path = telemetry.save_all_events()
    print(f"\n💾 Events saved to: {save_path}")
    
    # Show some events
    print(f"\n📋 Recent events:")
    recent_events = telemetry.get_events(limit=5)
    for event in recent_events:
        print(f"  {datetime.fromtimestamp(event.timestamp).strftime('%H:%M:%S')} - {event.event_type}")