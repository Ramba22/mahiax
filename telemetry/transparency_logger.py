"""
Transparency Logger for MAHIA-X
Implements training logs with Param-Changes/Checkpoint-Meta for full reproducibility and auditability
"""

import json
import time
import os
import hashlib
from typing import Dict, Any, Optional, List, Union
from collections import OrderedDict, defaultdict
from datetime import datetime
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TransparencyLogger:
    """Transparency logger for training logs with parameter changes and checkpoint metadata"""
    
    def __init__(self, log_dir: str = "transparency_logs", experiment_name: str = "default_experiment"):
        """
        Initialize transparency logger
        
        Args:
            log_dir: Directory to store log files
            experiment_name: Name of the experiment
        """
        self.log_dir = log_dir
        self.experiment_name = experiment_name
        self.log_file = os.path.join(log_dir, f"{experiment_name}_transparency_log.json")
        self.checkpoint_meta_file = os.path.join(log_dir, f"{experiment_name}_checkpoint_meta.json")
        
        # Create log directory if it doesn't exist
        os.makedirs(log_dir, exist_ok=True)
        
        # Initialize log storage
        self.training_logs = OrderedDict()
        self.parameter_changes = OrderedDict()
        self.checkpoint_metadata = OrderedDict()
        self.alerts = []
        
        # Initialize log files
        self._initialize_log_files()
        
        print(f"✅ TransparencyLogger initialized for experiment: {experiment_name}")
        print(f"   Log directory: {log_dir}")
        
    def _initialize_log_files(self):
        """Initialize log files with headers"""
        # Create initial log file if it doesn't exist
        if not os.path.exists(self.log_file):
            with open(self.log_file, 'w') as f:
                json.dump({
                    "experiment_name": self.experiment_name,
                    "created_at": datetime.now().isoformat(),
                    "logs": {}
                }, f, indent=2)
                
        # Create initial checkpoint metadata file if it doesn't exist
        if not os.path.exists(self.checkpoint_meta_file):
            with open(self.checkpoint_meta_file, 'w') as f:
                json.dump({
                    "experiment_name": self.experiment_name,
                    "created_at": datetime.now().isoformat(),
                    "checkpoints": {}
                }, f, indent=2)
                
    def log_training_step(self, step: int, metrics: Dict[str, Any], 
                         hyperparameters: Optional[Dict[str, Any]] = None,
                         model_state: Optional[Dict[str, Any]] = None) -> str:
        """
        Log a training step with metrics and optional model state
        
        Args:
            step: Training step number
            metrics: Dictionary of metrics
            hyperparameters: Current hyperparameters
            model_state: Optional model state information
            
        Returns:
            Log entry ID
        """
        timestamp = time.time()
        log_id = f"step_{step}_{int(timestamp * 1000)}"
        
        log_entry = {
            "log_id": log_id,
            "timestamp": timestamp,
            "datetime": datetime.fromtimestamp(timestamp).isoformat(),
            "step": step,
            "metrics": metrics,
            "hyperparameters": hyperparameters,
            "model_state": model_state
        }
        
        # Store in memory
        self.training_logs[log_id] = log_entry
        
        # Write to file
        self._append_to_log_file("training_logs", log_id, log_entry)
        
        # Check for anomalies
        self._check_for_anomalies(log_entry)
        
        return log_id
        
    def log_parameter_change(self, param_name: str, old_value: Any, new_value: Any,
                           reason: str = "", step: Optional[int] = None) -> str:
        """
        Log a parameter change
        
        Args:
            param_name: Name of the parameter
            old_value: Old value
            new_value: New value
            reason: Reason for the change
            step: Training step when change occurred
            
        Returns:
            Parameter change ID
        """
        timestamp = time.time()
        change_id = f"param_change_{param_name}_{int(timestamp * 1000)}"
        
        change_entry = {
            "change_id": change_id,
            "timestamp": timestamp,
            "datetime": datetime.fromtimestamp(timestamp).isoformat(),
            "param_name": param_name,
            "old_value": old_value,
            "new_value": new_value,
            "change": str(new_value) if old_value is None else f"{old_value} -> {new_value}",
            "reason": reason,
            "step": step
        }
        
        # Store in memory
        self.parameter_changes[change_id] = change_entry
        
        # Write to file
        self._append_to_log_file("parameter_changes", change_id, change_entry)
        
        # Generate alert for significant changes
        self._generate_parameter_change_alert(change_entry)
        
        return change_id
        
    def log_checkpoint(self, checkpoint_path: str, step: int, 
                      metrics: Dict[str, Any], model_hash: str = "",
                      metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Log a checkpoint with metadata
        
        Args:
            checkpoint_path: Path to checkpoint file
            step: Training step
            metrics: Metrics at checkpoint
            model_hash: Hash of model weights
            metadata: Additional checkpoint metadata
            
        Returns:
            Checkpoint ID
        """
        timestamp = time.time()
        checkpoint_id = f"checkpoint_{step}_{int(timestamp * 1000)}"
        
        # Calculate file hash if not provided
        if not model_hash and os.path.exists(checkpoint_path):
            model_hash = self._calculate_file_hash(checkpoint_path)
            
        checkpoint_entry = {
            "checkpoint_id": checkpoint_id,
            "timestamp": timestamp,
            "datetime": datetime.fromtimestamp(timestamp).isoformat(),
            "step": step,
            "checkpoint_path": checkpoint_path,
            "model_hash": model_hash,
            "metrics": metrics,
            "metadata": metadata or {},
            "file_size": os.path.getsize(checkpoint_path) if os.path.exists(checkpoint_path) else 0
        }
        
        # Store in memory
        self.checkpoint_metadata[checkpoint_id] = checkpoint_entry
        
        # Write to checkpoint metadata file
        self._append_to_checkpoint_file(checkpoint_id, checkpoint_entry)
        
        # Generate alert for significant metric changes
        self._generate_checkpoint_alert(checkpoint_entry)
        
        return checkpoint_id
        
    def _calculate_file_hash(self, file_path: str) -> str:
        """
        Calculate SHA256 hash of a file
        
        Args:
            file_path: Path to file
            
        Returns:
            SHA256 hash as hex string
        """
        try:
            hash_sha256 = hashlib.sha256()
            with open(file_path, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_sha256.update(chunk)
            return hash_sha256.hexdigest()
        except Exception as e:
            logger.warning(f"Could not calculate hash for {file_path}: {e}")
            return ""
            
    def _append_to_log_file(self, section: str, entry_id: str, entry: Dict[str, Any]):
        """
        Append entry to log file
        
        Args:
            section: Section name (training_logs, parameter_changes)
            entry_id: Entry ID
            entry: Entry data
        """
        try:
            # Read existing data
            if os.path.exists(self.log_file):
                with open(self.log_file, 'r') as f:
                    data = json.load(f)
            else:
                data = {
                    "experiment_name": self.experiment_name,
                    "created_at": datetime.now().isoformat(),
                    "logs": {}
                }
                
            # Add entry to section
            if section not in data["logs"]:
                data["logs"][section] = {}
            data["logs"][section][entry_id] = entry
            
            # Write back to file
            with open(self.log_file, 'w') as f:
                json.dump(data, f, indent=2, default=str)
                
        except Exception as e:
            logger.error(f"Failed to write to log file: {e}")
            
    def _append_to_checkpoint_file(self, checkpoint_id: str, entry: Dict[str, Any]):
        """
        Append checkpoint entry to checkpoint metadata file
        
        Args:
            checkpoint_id: Checkpoint ID
            entry: Checkpoint data
        """
        try:
            # Read existing data
            if os.path.exists(self.checkpoint_meta_file):
                with open(self.checkpoint_meta_file, 'r') as f:
                    data = json.load(f)
            else:
                data = {
                    "experiment_name": self.experiment_name,
                    "created_at": datetime.now().isoformat(),
                    "checkpoints": {}
                }
                
            # Add checkpoint entry
            data["checkpoints"][checkpoint_id] = entry
            
            # Write back to file
            with open(self.checkpoint_meta_file, 'w') as f:
                json.dump(data, f, indent=2, default=str)
                
        except Exception as e:
            logger.error(f"Failed to write to checkpoint metadata file: {e}")
            
    def _check_for_anomalies(self, log_entry: Dict[str, Any]):
        """
        Check for anomalies in training metrics
        
        Args:
            log_entry: Training log entry
        """
        metrics = log_entry.get("metrics", {})
        
        # Check for gradient explosion
        if "gradient_norm" in metrics:
            grad_norm = metrics["gradient_norm"]
            if grad_norm > 100.0:  # Threshold for gradient explosion
                alert = {
                    "type": "gradient_explosion",
                    "severity": "critical",
                    "timestamp": log_entry["timestamp"],
                    "step": log_entry["step"],
                    "gradient_norm": grad_norm,
                    "message": f"Gradient explosion detected: norm={grad_norm:.2f}"
                }
                self.alerts.append(alert)
                logger.critical(f"🚨 {alert['message']}")
                
        # Check for loss divergence
        if "loss" in metrics:
            loss = metrics["loss"]
            if loss > 10.0:  # Threshold for loss divergence
                alert = {
                    "type": "loss_divergence",
                    "severity": "warning",
                    "timestamp": log_entry["timestamp"],
                    "step": log_entry["step"],
                    "loss": loss,
                    "message": f"Loss divergence detected: loss={loss:.2f}"
                }
                self.alerts.append(alert)
                logger.warning(f"⚠️  {alert['message']}")
                
    def _generate_parameter_change_alert(self, change_entry: Dict[str, Any]):
        """
        Generate alert for significant parameter changes
        
        Args:
            change_entry: Parameter change entry
        """
        param_name = change_entry["param_name"]
        old_value = change_entry["old_value"]
        new_value = change_entry["new_value"]
        
        # Check for significant learning rate changes
        if "learning_rate" in param_name.lower():
            try:
                old_lr = float(old_value) if old_value is not None else 0.0
                new_lr = float(new_value)
                if old_lr > 0 and abs(new_lr - old_lr) / old_lr > 0.5:  # 50% change
                    alert = {
                        "type": "lr_change",
                        "severity": "info",
                        "timestamp": change_entry["timestamp"],
                        "param_name": param_name,
                        "change": f"{old_lr} -> {new_lr}",
                        "message": f"Significant learning rate change: {old_lr} -> {new_lr}"
                    }
                    self.alerts.append(alert)
                    logger.info(f"ℹ️  {alert['message']}")
            except (ValueError, TypeError):
                pass  # Ignore if values can't be converted to float
                
    def _generate_checkpoint_alert(self, checkpoint_entry: Dict[str, Any]):
        """
        Generate alert for significant metric changes at checkpoints
        
        Args:
            checkpoint_entry: Checkpoint entry
        """
        metrics = checkpoint_entry.get("metrics", {})
        
        # Check for significant accuracy drops
        if "accuracy" in metrics:
            accuracy = metrics["accuracy"]
            if accuracy < 0.5:  # Threshold for poor performance
                alert = {
                    "type": "low_accuracy",
                    "severity": "warning",
                    "timestamp": checkpoint_entry["timestamp"],
                    "step": checkpoint_entry["step"],
                    "accuracy": accuracy,
                    "message": f"Low accuracy at checkpoint: {accuracy:.4f}"
                }
                self.alerts.append(alert)
                logger.warning(f"⚠️  {alert['message']}")
                
    def get_training_logs(self, limit: Optional[int] = None) -> Dict[str, Any]:
        """
        Get training logs
        
        Args:
            limit: Maximum number of logs to return
            
        Returns:
            Training logs
        """
        if limit:
            # Return last N logs
            logs_list = list(self.training_logs.items())
            return OrderedDict(logs_list[-limit:])
        return self.training_logs
        
    def get_parameter_changes(self, limit: Optional[int] = None) -> Dict[str, Any]:
        """
        Get parameter changes
        
        Args:
            limit: Maximum number of changes to return
            
        Returns:
            Parameter changes
        """
        if limit:
            # Return last N changes
            changes_list = list(self.parameter_changes.items())
            return OrderedDict(changes_list[-limit:])
        return self.parameter_changes
        
    def get_checkpoint_metadata(self, limit: Optional[int] = None) -> Dict[str, Any]:
        """
        Get checkpoint metadata
        
        Args:
            limit: Maximum number of checkpoints to return
            
        Returns:
            Checkpoint metadata
        """
        if limit:
            # Return last N checkpoints
            checkpoints_list = list(self.checkpoint_metadata.items())
            return OrderedDict(checkpoints_list[-limit:])
        return self.checkpoint_metadata
        
    def get_alerts(self, severity: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Get alerts
        
        Args:
            severity: Filter by severity (critical, warning, info)
            
        Returns:
            List of alerts
        """
        if severity:
            return [alert for alert in self.alerts if alert.get("severity") == severity]
        return self.alerts.copy()
        
    def export_full_report(self, report_file: str) -> bool:
        """
        Export full transparency report
        
        Args:
            report_file: Path to export report
            
        Returns:
            True if successful, False otherwise
        """
        try:
            report = {
                "experiment_name": self.experiment_name,
                "generated_at": datetime.now().isoformat(),
                "training_logs": dict(self.training_logs),
                "parameter_changes": dict(self.parameter_changes),
                "checkpoint_metadata": dict(self.checkpoint_metadata),
                "alerts": self.alerts,
                "summary": {
                    "total_training_steps": len(self.training_logs),
                    "total_parameter_changes": len(self.parameter_changes),
                    "total_checkpoints": len(self.checkpoint_metadata),
                    "total_alerts": len(self.alerts)
                }
            }
            
            with open(report_file, 'w') as f:
                json.dump(report, f, indent=2, default=str)
                
            print(f"✅ Full transparency report exported to: {report_file}")
            return True
        except Exception as e:
            logger.error(f"Failed to export transparency report: {e}")
            return False
            
    def generate_summary_report(self) -> str:
        """
        Generate summary report
        
        Returns:
            Formatted summary report
        """
        report = f"""
📊 TRANSPARENCY LOGGER SUMMARY REPORT
====================================
Experiment: {self.experiment_name}
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

📈 Training Progress:
  - Training Steps Logged: {len(self.training_logs)}
  - Parameter Changes: {len(self.parameter_changes)}
  - Checkpoints Saved: {len(self.checkpoint_metadata)}
  - Alerts Generated: {len(self.alerts)}

📝 Recent Activity:
"""
        
        # Show recent training logs
        recent_logs = list(self.training_logs.items())[-3:]
        for log_id, log_entry in recent_logs:
            step = log_entry.get("step", "N/A")
            timestamp = datetime.fromtimestamp(log_entry["timestamp"]).strftime('%H:%M:%S')
            report += f"  • Step {step} at {timestamp}\n"
            
        # Show recent parameter changes
        if self.parameter_changes:
            report += "\n🔧 Parameter Changes:\n"
            recent_changes = list(self.parameter_changes.items())[-3:]
            for change_id, change_entry in recent_changes:
                param_name = change_entry.get("param_name", "Unknown")
                change = change_entry.get("change", "N/A")
                timestamp = datetime.fromtimestamp(change_entry["timestamp"]).strftime('%H:%M:%S')
                report += f"  • {param_name}: {change} at {timestamp}\n"
                
        # Show alerts
        if self.alerts:
            report += "\n🚨 Alerts:\n"
            recent_alerts = self.alerts[-3:]
            for alert in recent_alerts:
                severity = alert.get("severity", "info").upper()
                message = alert.get("message", "No message")
                timestamp = datetime.fromtimestamp(alert["timestamp"]).strftime('%H:%M:%S')
                report += f"  • [{severity}] {message} at {timestamp}\n"
                
        return report


def demo_transparency_logger():
    """Demonstrate transparency logger functionality"""
    print("🚀 Demonstrating Transparency Logger...")
    print("=" * 50)
    
    # Create logger
    logger = TransparencyLogger(
        log_dir="demo_transparency_logs",
        experiment_name="demo_experiment"
    )
    print("✅ Created transparency logger")
    
    # Simulate training steps
    print("📝 Logging training steps...")
    for step in range(1, 6):
        metrics = {
            "loss": 1.0 / step,  # Decreasing loss
            "accuracy": 0.5 + (step * 0.1),  # Increasing accuracy
            "gradient_norm": 5.0 + (step * 0.5),  # Increasing gradient norm
            "learning_rate": 0.001
        }
        
        hyperparameters = {
            "batch_size": 32,
            "learning_rate": 0.001,
            "weight_decay": 1e-5
        }
        
        model_state = {
            "layer_count": 12,
            "parameter_count": 1000000 * step
        }
        
        log_id = logger.log_training_step(
            step=step,
            metrics=metrics,
            hyperparameters=hyperparameters,
            model_state=model_state
        )
        
        print(f"   Logged step {step} with ID: {log_id}")
        
    # Log parameter changes
    print("\n🔧 Logging parameter changes...")
    logger.log_parameter_change(
        param_name="learning_rate",
        old_value=0.001,
        new_value=0.0005,
        reason="Learning rate decay",
        step=3
    )
    
    logger.log_parameter_change(
        param_name="batch_size",
        old_value=32,
        new_value=64,
        reason="Memory optimization",
        step=4
    )
    
    print("   Logged parameter changes")
    
    # Log checkpoints
    print("\n💾 Logging checkpoints...")
    # Create dummy checkpoint files for demo
    dummy_checkpoint_paths = []
    for i in range(1, 4):
        checkpoint_path = f"demo_transparency_logs/checkpoint_{i}.pt"
        with open(checkpoint_path, 'w') as f:
            f.write(f"Dummy checkpoint {i}")
        dummy_checkpoint_paths.append(checkpoint_path)
        
    for i, checkpoint_path in enumerate(dummy_checkpoint_paths, 1):
        metrics = {
            "loss": 1.0 / (i * 2),
            "accuracy": 0.6 + (i * 0.1)
        }
        
        checkpoint_id = logger.log_checkpoint(
            checkpoint_path=checkpoint_path,
            step=i * 2,
            metrics=metrics,
            model_hash=f"dummy_hash_{i}"
        )
        
        print(f"   Logged checkpoint {i} with ID: {checkpoint_id}")
    
    # Show summary report
    print("\n" + "=" * 60)
    print(logger.generate_summary_report())
    
    # Export full report
    report_success = logger.export_full_report("demo_transparency_report.json")
    print(f"✅ Full report export: {'SUCCESS' if report_success else 'FAILED'}")
    
    print("\n" + "=" * 50)
    print("TRANSPARENCY LOGGER DEMO SUMMARY")
    print("=" * 50)
    print("Key Features Implemented:")
    print("  1. Training step logging with metrics")
    print("  2. Parameter change tracking")
    print("  3. Checkpoint metadata with file hashing")
    print("  4. Automated anomaly detection")
    print("  5. Alert generation for critical events")
    print("  6. Comprehensive reporting")
    print("\nBenefits:")
    print("  - Full reproducibility of training process")
    print("  - Audit trail for all parameter changes")
    print("  - Checkpoint verification with hashing")
    print("  - Early detection of training issues")
    print("  - Compliance with research transparency standards")
    
    print("\n✅ Transparency logger demonstration completed!")


if __name__ == "__main__":
    demo_transparency_logger()