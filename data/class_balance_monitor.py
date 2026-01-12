"""
Class Balance Monitor for MAHIA-X
This module implements a class balance monitor for DataLoader with warning generation for imbalanced datasets.
"""

import math
import time
import json
import os
from typing import Dict, Any, Optional, List, Union
from collections import OrderedDict, defaultdict
from datetime import datetime

# Conditional imports with fallbacks
TORCH_AVAILABLE = False
torch = None

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    pass

NUMPY_AVAILABLE = False
np = None

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    pass

class ClassBalanceMonitor:
    """Class balance monitor for DataLoader with warning generation for imbalanced datasets"""
    
    def __init__(self, warning_threshold: float = 0.1, log_file: Optional[str] = None):
        """
        Initialize class balance monitor
        
        Args:
            warning_threshold: Minimum ratio threshold for class balance warnings (default 0.1 = 10%)
            log_file: File to store balance logs (optional)
        """
        self.warning_threshold = warning_threshold
        self.log_file = log_file
        self.balance_logs = OrderedDict()
        self.total_batches = 0
        self.warnings = []
        
        # Load existing logs if file exists
        if self.log_file and os.path.exists(self.log_file):
            self._load_logs_from_file()
            
        print(f"✅ ClassBalanceMonitor initialized with warning threshold: {warning_threshold}")
        
    def monitor_batch(self, 
                     batch_id: str,
                     labels,
                     class_names: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Monitor class balance for a batch
        
        Args:
            batch_id: Unique identifier for the batch
            labels: Labels for the batch (list, tensor, or numpy array)
            class_names: Optional class names for better reporting
            
        Returns:
            Balance analysis results
        """
        # Convert labels to list if needed
        if TORCH_AVAILABLE and isinstance(labels, torch.Tensor):
            label_list = labels.flatten().tolist()
        elif NUMPY_AVAILABLE and isinstance(labels, np.ndarray):
            label_list = labels.flatten().tolist()
        else:
            label_list = list(labels)
            
        if not label_list:
            return {"status": "no_labels", "balance_ratio": 0.0}
            
        # Calculate class distribution
        class_counts = defaultdict(int)
        for label in label_list:
            class_counts[label] += 1
            
        total_samples = len(label_list)
        num_classes = len(class_counts)
        
        # Calculate ratios
        class_ratios = {cls: count / total_samples for cls, count in class_counts.items()}
        
        # Find min and max ratios
        min_ratio = min(class_ratios.values()) if class_ratios else 0.0
        max_ratio = max(class_ratios.values()) if class_ratios else 0.0
        
        # Calculate balance ratio (min/max)
        balance_ratio = min_ratio / max_ratio if max_ratio > 0 else 0.0
        
        # Check for imbalance warning
        is_imbalanced = min_ratio < self.warning_threshold
        
        # Create analysis results
        results = {
            "batch_id": batch_id,
            "total_samples": total_samples,
            "num_classes": num_classes,
            "class_counts": dict(class_counts),
            "class_ratios": class_ratios,
            "min_ratio": min_ratio,
            "max_ratio": max_ratio,
            "balance_ratio": balance_ratio,
            "is_imbalanced": is_imbalanced,
            "timestamp": time.time()
        }
        
        # Generate warning if imbalanced
        if is_imbalanced:
            warning = self._generate_imbalance_warning(batch_id, class_ratios, class_names)
            self.warnings.append(warning)
            results["warning"] = warning
            
        # Log results
        log_entry_id = f"balance_{self.total_batches}_{int(time.time() * 1000)}"
        self.balance_logs[log_entry_id] = results
        self.total_batches += 1
        
        # Save to file if specified
        if self.log_file:
            self._save_logs_to_file()
            
        print(f"✅ Monitored batch {batch_id}: {num_classes} classes, balance ratio = {balance_ratio:.4f}")
        
        if is_imbalanced:
            print(f"⚠️  Imbalance warning for batch {batch_id}")
            
        return results
        
    def _generate_imbalance_warning(self, batch_id: str, class_ratios: Dict, 
                                  class_names: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Generate imbalance warning
        
        Args:
            batch_id: Batch identifier
            class_ratios: Class ratios dictionary
            class_names: Optional class names
            
        Returns:
            Warning dictionary
        """
        # Find the rarest and most common classes
        min_class = min(class_ratios.keys(), key=lambda k: class_ratios[k])
        max_class = max(class_ratios.keys(), key=lambda k: class_ratios[k])
        min_ratio = class_ratios[min_class]
        max_ratio = class_ratios[max_class]
        
        # Get class names if provided
        min_class_name = class_names[min_class] if class_names and min_class < len(class_names) else str(min_class)
        max_class_name = class_names[max_class] if class_names and max_class < len(class_names) else str(max_class)
        
        warning = {
            "type": "class_imbalance",
            "batch_id": batch_id,
            "timestamp": time.time(),
            "severity": "warning",
            "rarest_class": {
                "id": min_class,
                "name": min_class_name,
                "ratio": min_ratio
            },
            "most_common_class": {
                "id": max_class,
                "name": max_class_name,
                "ratio": max_ratio
            },
            "imbalance_ratio": min_ratio / max_ratio if max_ratio > 0 else 0.0,
            "threshold": self.warning_threshold
        }
        
        return warning
        
    def get_overall_balance(self, window_size: Optional[int] = None) -> Dict[str, Any]:
        """
        Get overall class balance across batches
        
        Args:
            window_size: Number of recent batches to consider (None for all)
            
        Returns:
            Overall balance statistics
        """
        if not self.balance_logs:
            return {"status": "no_data"}
            
        # Get recent logs if window size specified
        if window_size:
            logs = list(self.balance_logs.values())[-window_size:]
        else:
            logs = list(self.balance_logs.values())
            
        if not logs:
            return {"status": "no_data"}
            
        # Aggregate class counts across all batches
        total_class_counts = defaultdict(int)
        total_samples = 0
        
        for log in logs:
            class_counts = log.get("class_counts", {})
            for cls, count in class_counts.items():
                total_class_counts[cls] += count
            total_samples += log.get("total_samples", 0)
            
        if total_samples == 0:
            return {"status": "no_samples"}
            
        # Calculate overall ratios
        overall_ratios = {cls: count / total_samples for cls, count in total_class_counts.items()}
        
        # Calculate statistics
        min_ratio = min(overall_ratios.values()) if overall_ratios else 0.0
        max_ratio = max(overall_ratios.values()) if overall_ratios else 0.0
        balance_ratio = min_ratio / max_ratio if max_ratio > 0 else 0.0
        
        # Check for overall imbalance
        is_imbalanced = min_ratio < self.warning_threshold
        
        return {
            "total_batches": len(logs),
            "total_samples": total_samples,
            "num_classes": len(total_class_counts),
            "class_counts": dict(total_class_counts),
            "class_ratios": overall_ratios,
            "min_ratio": min_ratio,
            "max_ratio": max_ratio,
            "balance_ratio": balance_ratio,
            "is_imbalanced": is_imbalanced,
            "warning_threshold": self.warning_threshold
        }
        
    def get_warnings(self, severity: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Get warnings
        
        Args:
            severity: Filter by severity (warning, error, info)
            
        Returns:
            List of warnings
        """
        if severity:
            return [warning for warning in self.warnings if warning.get("severity") == severity]
        return self.warnings
        
    def clear_warnings(self):
        """Clear warnings"""
        self.warnings.clear()
        
    def _save_logs_to_file(self):
        """Save logs to file"""
        if not self.log_file:
            return
            
        try:
            # Create directory if it doesn't exist
            log_dir = os.path.dirname(self.log_file)
            if log_dir and not os.path.exists(log_dir):
                os.makedirs(log_dir)
                
            # Write logs to file
            with open(self.log_file, 'w') as f:
                json.dump(dict(self.balance_logs), f, indent=2, default=str)
        except Exception as e:
            print(f"⚠️  Failed to save balance logs: {e}")
            
    def _load_logs_from_file(self):
        """Load logs from file"""
        if not self.log_file or not os.path.exists(self.log_file):
            return
            
        try:
            with open(self.log_file, 'r') as f:
                loaded_logs = json.load(f)
                self.balance_logs = OrderedDict(loaded_logs)
                self.total_batches = len(self.balance_logs)
        except Exception as e:
            print(f"⚠️  Failed to load balance logs: {e}")
            
    def export_balance_report(self, report_file: str) -> bool:
        """
        Export balance report to file
        
        Args:
            report_file: File to export report to
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Get overall balance
            overall_balance = self.get_overall_balance()
            
            # Prepare report
            report = {
                "generated_at": datetime.now().isoformat(),
                "monitor_config": {
                    "warning_threshold": self.warning_threshold
                },
                "overall_balance": overall_balance,
                "total_batches_monitored": self.total_batches,
                "warnings": self.warnings,
                "summary": {
                    "imbalanced_batches": len([w for w in self.warnings if w.get("type") == "class_imbalance"]),
                    "total_warnings": len(self.warnings)
                }
            }
            
            # Write report
            with open(report_file, 'w') as f:
                json.dump(report, f, indent=2, default=str)
                
            print(f"✅ Balance report exported to: {report_file}")
            return True
        except Exception as e:
            print(f"❌ Failed to export balance report: {e}")
            return False
            
    def clear_logs(self):
        """Clear all logs"""
        self.balance_logs.clear()
        self.total_batches = 0
        if self.log_file and os.path.exists(self.log_file):
            try:
                os.remove(self.log_file)
            except Exception as e:
                print(f"⚠️  Failed to remove log file: {e}")


class DataLoaderMonitor:
    """DataLoader monitor that integrates with ClassBalanceMonitor"""
    
    def __init__(self, balance_monitor: ClassBalanceMonitor):
        """
        Initialize DataLoader monitor
        
        Args:
            balance_monitor: Class balance monitor instance
        """
        self.balance_monitor = balance_monitor
        self.batch_count = 0
        
        print("✅ DataLoaderMonitor initialized")
        
    def monitor_dataloader(self, dataloader, class_names: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Monitor an entire DataLoader for class balance
        
        Args:
            dataloader: PyTorch DataLoader or similar iterable
            class_names: Optional class names for better reporting
            
        Returns:
            Monitoring results
        """
        if not TORCH_AVAILABLE:
            return {"status": "torch_not_available"}
            
        batch_results = []
        total_samples = 0
        
        print("🔄 Monitoring DataLoader for class balance...")
        
        try:
            for batch_idx, batch in enumerate(dataloader):
                # Extract labels (assume batch is tuple of (data, labels) or similar)
                if isinstance(batch, (tuple, list)) and len(batch) >= 2:
                    labels = batch[1]  # Assume labels are second element
                elif hasattr(batch, 'labels'):
                    labels = batch.labels
                else:
                    # Try to infer labels
                    labels = batch
                    
                # Monitor this batch
                batch_id = f"batch_{batch_idx}"
                result = self.balance_monitor.monitor_batch(batch_id, labels, class_names)
                batch_results.append(result)
                
                total_samples += result.get("total_samples", 0)
                
                # Limit monitoring for large datasets
                if batch_idx >= 100:  # Monitor first 100 batches
                    print("⚠️  Stopping monitoring after 100 batches for performance")
                    break
                    
            # Calculate summary statistics
            if batch_results:
                balance_ratios = [r.get("balance_ratio", 0) for r in batch_results]
                avg_balance_ratio = sum(balance_ratios) / len(balance_ratios)
                
                imbalanced_batches = [r for r in batch_results if r.get("is_imbalanced", False)]
                
                results = {
                    "total_batches": len(batch_results),
                    "total_samples": total_samples,
                    "average_balance_ratio": avg_balance_ratio,
                    "imbalanced_batches": len(imbalanced_batches),
                    "imbalanced_batch_ids": [r.get("batch_id") for r in imbalanced_batches],
                    "batch_results": batch_results
                }
                
                print(f"✅ DataLoader monitoring complete: {len(batch_results)} batches, avg balance = {avg_balance_ratio:.4f}")
                return results
            else:
                print("⚠️  No batches monitored")
                return {"status": "no_batches"}
                
        except Exception as e:
            print(f"❌ Error monitoring DataLoader: {e}")
            return {"status": "error", "error": str(e)}
            
    def get_dataloader_summary(self) -> Dict[str, Any]:
        """
        Get DataLoader monitoring summary
        
        Returns:
            Summary dictionary
        """
        overall_balance = self.balance_monitor.get_overall_balance()
        warnings = self.balance_monitor.get_warnings()
        
        return {
            "overall_balance": overall_balance,
            "total_warnings": len(warnings),
            "imbalanced_batches": len([w for w in warnings if w.get("type") == "class_imbalance"]),
            "monitoring_batches": self.balance_monitor.total_batches
        }


def demo_class_balance_monitor():
    """Demonstrate class balance monitor functionality"""
    print("🚀 Demonstrating Class Balance Monitor...")
    print("=" * 60)
    
    # Create class balance monitor
    monitor = ClassBalanceMonitor(warning_threshold=0.15, log_file="balance_demo_log.json")
    print("✅ Created class balance monitor")
    
    # Create DataLoader monitor
    dl_monitor = DataLoaderMonitor(monitor)
    print("✅ Created DataLoader monitor")
    
    # Test with balanced data
    print("\n⚖️  Testing with balanced data...")
    balanced_labels = [0, 1, 2] * 20 + [0, 1, 2] * 15  # 35 of each class
    balanced_result = monitor.monitor_batch("balanced_batch", balanced_labels, ["Class_A", "Class_B", "Class_C"])
    print(f"   Balanced batch result: balance ratio = {balanced_result['balance_ratio']:.4f}")
    print(f"   Is imbalanced: {balanced_result['is_imbalanced']}")
    
    # Test with imbalanced data
    print("\n⚠️  Testing with imbalanced data...")
    imbalanced_labels = [0] * 50 + [1] * 30 + [2] * 5  # Very imbalanced
    imbalanced_result = monitor.monitor_batch("imbalanced_batch", imbalanced_labels, ["Class_A", "Class_B", "Class_C"])
    print(f"   Imbalanced batch result: balance ratio = {imbalanced_result['balance_ratio']:.4f}")
    print(f"   Is imbalanced: {imbalanced_result['is_imbalanced']}")
    if "warning" in imbalanced_result:
        warning = imbalanced_result["warning"]
        print(f"   Warning: Rarest class '{warning['rarest_class']['name']}' has ratio {warning['rarest_class']['ratio']:.4f}")
    
    # Test with another imbalanced batch
    imbalanced_labels2 = [0] * 40 + [1] * 10 + [2] * 2  # Also imbalanced
    imbalanced_result2 = monitor.monitor_batch("imbalanced_batch2", imbalanced_labels2, ["Class_A", "Class_B", "Class_C"])
    print(f"   Second imbalanced batch: balance ratio = {imbalanced_result2['balance_ratio']:.4f}")
    
    # Get overall balance
    print("\n📈 Getting overall balance statistics...")
    overall_balance = monitor.get_overall_balance()
    print(f"   Overall balance: {overall_balance['num_classes']} classes, ratio = {overall_balance['balance_ratio']:.4f}")
    print(f"   Is overall imbalanced: {overall_balance['is_imbalanced']}")
    
    # Show warnings
    warnings = monitor.get_warnings()
    print(f"   Total warnings generated: {len(warnings)}")
    for i, warning in enumerate(warnings[-2:]):  # Show last 2 warnings
        print(f"   Warning {len(warnings)-1+i}: {warning['type']} in {warning['batch_id']}")
    
    # Test with numpy array labels
    print("\n🧮 Testing with numpy array labels...")
    if NUMPY_AVAILABLE:
        numpy_labels = np.array([0, 0, 1, 1, 1, 2, 2, 2, 2])
        numpy_result = monitor.monitor_batch("numpy_batch", numpy_labels, ["A", "B", "C"])
        print(f"   Numpy batch result: balance ratio = {numpy_result['balance_ratio']:.4f}")
    else:
        print("   Numpy not available, skipping test")
    
    # Test with torch tensor labels
    print("\n🔥 Testing with torch tensor labels...")
    if TORCH_AVAILABLE:
        torch_labels = torch.tensor([0, 0, 0, 1, 1, 2])
        torch_result = monitor.monitor_batch("torch_batch", torch_labels, ["X", "Y", "Z"])
        print(f"   Torch batch result: balance ratio = {torch_result['balance_ratio']:.4f}")
    else:
        print("   PyTorch not available, skipping test")
    
    # Export report
    report_success = monitor.export_balance_report("balance_report.json")
    print(f"✅ Report export: {'SUCCESS' if report_success else 'FAILED'}")
    
    print("\n" + "=" * 60)
    print("CLASS BALANCE MONITOR DEMO SUMMARY")
    print("=" * 60)
    print("Key Features Implemented:")
    print("  1. Real-time class balance monitoring for batches")
    print("  2. Automatic imbalance detection with configurable thresholds")
    print(" 3. Comprehensive warning generation for rare classes")
    print("  4. Statistical analysis of class distributions")
    print("  5. Integration with DataLoader for dataset-level monitoring")
    print("  6. Persistent logging and reporting")
    print("\nBenefits:")
    print("  - Early detection of dataset imbalance issues")
    print("  - Automated quality control for training data")
    print("  - Better understanding of class distributions")
    print("  - Improved model training stability")
    print("  - Reduced bias in model predictions")
    
    print("\n✅ Class Balance Monitor demonstration completed!")


if __name__ == "__main__":
    demo_class_balance_monitor()