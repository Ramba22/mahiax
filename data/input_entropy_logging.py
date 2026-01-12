"""
Input Entropy Logging for MAHIA-X
This module implements input entropy logging to store data entropy per batch for curriculum analysis.
"""

import math
import time
import json
import os
from typing import Dict, Any, Optional, List, Union
from collections import OrderedDict
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

class InputEntropyLogger:
    """Input entropy logger for storing data entropy per batch for curriculum analysis"""
    
    def __init__(self, log_file: Optional[str] = None, max_entries: int = 10000):
        """
        Initialize input entropy logger
        
        Args:
            log_file: File to store entropy logs (optional)
            max_entries: Maximum number of entries to keep in memory
        """
        self.log_file = log_file
        self.max_entries = max_entries
        self.entropy_logs = OrderedDict()
        self.total_entries = 0
        
        # Load existing logs if file exists
        if self.log_file and os.path.exists(self.log_file):
            self._load_logs_from_file()
            
        print("✅ InputEntropyLogger initialized")
        
    def log_batch_entropy(self, 
                         batch_id: str,
                         entropy: float,
                         batch_size: int,
                         data_type: str = "unknown",
                         metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Log entropy for a batch
        
        Args:
            batch_id: Unique identifier for the batch
            entropy: Entropy value
            batch_size: Number of samples in batch
            data_type: Type of data (text, image, tabular, etc.)
            metadata: Additional metadata (optional)
            
        Returns:
            Log entry ID
        """
        # Create log entry
        log_entry = {
            "batch_id": batch_id,
            "entropy": entropy,
            "batch_size": batch_size,
            "data_type": data_type,
            "timestamp": time.time(),
            "metadata": metadata or {}
        }
        
        # Generate entry ID
        entry_id = f"entropy_{self.total_entries}_{int(time.time() * 1000)}"
        log_entry["entry_id"] = entry_id
        
        # Store log entry
        self.entropy_logs[entry_id] = log_entry
        self.total_entries += 1
        
        # Remove oldest entries if we exceed max_entries
        if len(self.entropy_logs) > self.max_entries:
            # Remove oldest entry
            oldest_key = next(iter(self.entropy_logs))
            del self.entropy_logs[oldest_key]
            
        # Save to file if specified
        if self.log_file:
            self._save_logs_to_file()
            
        print(f"✅ Logged batch entropy: {batch_id} (entropy={entropy:.4f}, size={batch_size})")
        return entry_id
        
    def calculate_text_entropy(self, text_tokens) -> float:
        """
        Calculate entropy for text tokens
        
        Args:
            text_tokens: Text tokens (list of integers, tensor, or numpy array)
            
        Returns:
            Entropy value
        """
        if not text_tokens:
            return 0.0
            
        # Convert to list if needed
        if TORCH_AVAILABLE and isinstance(text_tokens, torch.Tensor):
            token_list = text_tokens.flatten().tolist()
        elif NUMPY_AVAILABLE and isinstance(text_tokens, np.ndarray):
            token_list = text_tokens.flatten().tolist()
        else:
            token_list = list(text_tokens)
            
        if not token_list:
            return 0.0
            
        # Calculate frequency distribution
        token_counts = {}
        total_tokens = len(token_list)
        
        for token in token_list:
            token_counts[token] = token_counts.get(token, 0) + 1
            
        # Calculate entropy
        entropy = 0.0
        for count in token_counts.values():
            probability = count / total_tokens
            if probability > 0:
                entropy -= probability * (np.log2(probability) if NUMPY_AVAILABLE else (math.log(probability) / math.log(2)))
                
        return entropy
        
    def calculate_feature_entropy(self, features) -> float:
        """
        Calculate entropy for continuous features using binning
        
        Args:
            features: Feature values (list, tensor, or numpy array)
            
        Returns:
            Entropy value
        """
        if not features:
            return 0.0
            
        # Convert to list if needed
        if TORCH_AVAILABLE and isinstance(features, torch.Tensor):
            feature_list = features.flatten().tolist()
        elif NUMPY_AVAILABLE and isinstance(features, np.ndarray):
            feature_list = features.flatten().tolist()
        else:
            feature_list = list(features)
            
        if not feature_list:
            return 0.0
            
        # Create bins for continuous features
        if NUMPY_AVAILABLE:
            hist, bin_edges = np.histogram(feature_list, bins=50)
        else:
            # Simple binning without numpy
            min_val = min(feature_list)
            max_val = max(feature_list)
            bin_width = (max_val - min_val) / 50 if max_val > min_val else 1.0
            hist = [0] * 50
            
            for val in feature_list:
                bin_idx = min(int((val - min_val) / bin_width), 49)
                hist[bin_idx] += 1
                
        # Calculate entropy from histogram
        total = sum(hist)
        if total == 0:
            return 0.0
            
        entropy = 0.0
        for count in hist:
            if count > 0:
                probability = count / total
                if probability > 0:
                    entropy -= probability * (np.log2(probability) if NUMPY_AVAILABLE else (math.log(probability) / math.log(2)))
                    
        return entropy
        
    def get_entropy_statistics(self, data_type: Optional[str] = None) -> Dict[str, float]:
        """
        Get entropy statistics
        
        Args:
            data_type: Filter by data type (optional)
            
        Returns:
            Statistics dictionary
        """
        if not self.entropy_logs:
            return {"count": 0}
            
        # Filter logs by data type if specified
        if data_type:
            filtered_logs = [log for log in self.entropy_logs.values() if log.get("data_type") == data_type]
        else:
            filtered_logs = list(self.entropy_logs.values())
            
        if not filtered_logs:
            return {"count": 0}
            
        # Calculate statistics
        entropies = [log["entropy"] for log in filtered_logs]
        batch_sizes = [log["batch_size"] for log in filtered_logs]
        
        stats = {
            "count": len(filtered_logs),
            "mean_entropy": np.mean(entropies) if NUMPY_AVAILABLE and entropies else sum(entropies) / len(entropies),
            "std_entropy": np.std(entropies) if NUMPY_AVAILABLE and entropies else self._calculate_std(entropies),
            "min_entropy": min(entropies),
            "max_entropy": max(entropies),
            "mean_batch_size": np.mean(batch_sizes) if NUMPY_AVAILABLE and batch_sizes else sum(batch_sizes) / len(batch_sizes),
            "total_samples": sum(batch_sizes)
        }
        
        return stats
        
    def _calculate_std(self, values: List[float]) -> float:
        """Calculate standard deviation without numpy"""
        if not values:
            return 0.0
            
        mean = sum(values) / len(values)
        variance = sum((x - mean) ** 2 for x in values) / len(values)
        return variance ** 0.5
        
    def get_entropy_trend(self, window_size: int = 100) -> List[float]:
        """
        Get entropy trend over time
        
        Args:
            window_size: Size of moving average window
            
        Returns:
            List of entropy values
        """
        if not self.entropy_logs:
            return []
            
        # Get recent entropies
        entropies = [log["entropy"] for log in list(self.entropy_logs.values())[-window_size:]]
        return entropies
        
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
                json.dump(dict(self.entropy_logs), f, indent=2, default=str)
        except Exception as e:
            print(f"⚠️  Failed to save entropy logs: {e}")
            
    def _load_logs_from_file(self):
        """Load logs from file"""
        if not self.log_file or not os.path.exists(self.log_file):
            return
            
        try:
            with open(self.log_file, 'r') as f:
                loaded_logs = json.load(f)
                self.entropy_logs = OrderedDict(loaded_logs)
                self.total_entries = len(self.entropy_logs)
        except Exception as e:
            print(f"⚠️  Failed to load entropy logs: {e}")
            
    def export_entropy_report(self, report_file: str) -> bool:
        """
        Export entropy report to file
        
        Args:
            report_file: File to export report to
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Get statistics by data type
            data_types = set(log.get("data_type", "unknown") for log in self.entropy_logs.values())
            type_stats = {}
            for data_type in data_types:
                type_stats[data_type] = self.get_entropy_statistics(data_type)
                
            # Prepare report
            report = {
                "generated_at": datetime.now().isoformat(),
                "overall_stats": self.get_entropy_statistics(),
                "stats_by_type": type_stats,
                "total_entries": self.total_entries,
                "recent_trend": self.get_entropy_trend(50)
            }
            
            # Write report
            with open(report_file, 'w') as f:
                json.dump(report, f, indent=2, default=str)
                
            print(f"✅ Entropy report exported to: {report_file}")
            return True
        except Exception as e:
            print(f"❌ Failed to export entropy report: {e}")
            return False
            
    def clear_logs(self):
        """Clear all logs"""
        self.entropy_logs.clear()
        self.total_entries = 0
        if self.log_file and os.path.exists(self.log_file):
            try:
                os.remove(self.log_file)
            except Exception as e:
                print(f"⚠️  Failed to remove log file: {e}")


class CurriculumAnalyzer:
    """Curriculum analyzer using input entropy for difficulty assessment"""
    
    def __init__(self, entropy_logger: InputEntropyLogger):
        """
        Initialize curriculum analyzer
        
        Args:
            entropy_logger: Input entropy logger instance
        """
        self.entropy_logger = entropy_logger
        self.difficulty_thresholds = {
            "very_easy": 0.5,
            "easy": 1.0,
            "medium": 2.0,
            "hard": 3.0,
            "very_hard": 4.0
        }
        
        print("✅ CurriculumAnalyzer initialized")
        
    def assess_batch_difficulty(self, batch_entropy: float) -> str:
        """
        Assess difficulty level of a batch based on entropy
        
        Args:
            batch_entropy: Entropy value for the batch
            
        Returns:
            Difficulty level
        """
        if batch_entropy <= self.difficulty_thresholds["very_easy"]:
            return "very_easy"
        elif batch_entropy <= self.difficulty_thresholds["easy"]:
            return "easy"
        elif batch_entropy <= self.difficulty_thresholds["medium"]:
            return "medium"
        elif batch_entropy <= self.difficulty_thresholds["hard"]:
            return "hard"
        else:
            return "very_hard"
            
    def get_curriculum_recommendation(self) -> Dict[str, Any]:
        """
        Get curriculum recommendation based on entropy trends
        
        Returns:
            Recommendation dictionary
        """
        # Get recent entropy trend
        trend = self.entropy_logger.get_entropy_trend(100)
        
        if not trend:
            return {"recommendation": "insufficient_data", "confidence": 0.0}
            
        # Calculate trend statistics
        mean_entropy = np.mean(trend) if NUMPY_AVAILABLE and trend else sum(trend) / len(trend)
        std_entropy = np.std(trend) if NUMPY_AVAILABLE and trend else self.entropy_logger._calculate_std(trend)
        
        # Assess current difficulty level
        current_difficulty = self.assess_batch_difficulty(mean_entropy)
        
        # Determine recommendation based on trend
        if len(trend) >= 2:
            recent_improvement = trend[-1] - trend[-10] if len(trend) >= 10 else trend[-1] - trend[0]
            
            if recent_improvement > 0.1:  # Entropy increasing = getting harder
                recommendation = "increase_difficulty"
            elif recent_improvement < -0.1:  # Entropy decreasing = getting easier
                recommendation = "decrease_difficulty"
            else:
                recommendation = "maintain_current"
        else:
            recommendation = "maintain_current"
            
        return {
            "recommendation": recommendation,
            "current_difficulty": current_difficulty,
            "mean_entropy": mean_entropy,
            "entropy_std": std_entropy,
            "confidence": 1.0 - min(std_entropy / 5.0, 1.0)  # Confidence decreases with high variance
        }
        
    def update_difficulty_thresholds(self, new_thresholds: Dict[str, float]):
        """
        Update difficulty thresholds
        
        Args:
            new_thresholds: New threshold values
        """
        self.difficulty_thresholds.update(new_thresholds)
        print(f"✅ Updated difficulty thresholds: {self.difficulty_thresholds}")


def demo_input_entropy_logging():
    """Demonstrate input entropy logging functionality"""
    print("🚀 Demonstrating Input Entropy Logging...")
    print("=" * 60)
    
    # Create entropy logger
    logger = InputEntropyLogger("entropy_demo_log.json", max_entries=1000)
    print("✅ Created input entropy logger")
    
    # Create curriculum analyzer
    analyzer = CurriculumAnalyzer(logger)
    print("✅ Created curriculum analyzer")
    
    # Generate sample text data and calculate entropy
    print("\n🔤 Testing text entropy calculation...")
    
    # Low entropy text (repetitive)
    low_entropy_text = [1, 1, 1, 2, 2, 1, 1, 1, 2, 2] * 5  # Very repetitive
    low_entropy = logger.calculate_text_entropy(low_entropy_text)
    print(f"   Low entropy text: {low_entropy:.4f}")
    
    # High entropy text (diverse)
    high_entropy_text = list(range(50))  # All different tokens
    high_entropy = logger.calculate_text_entropy(high_entropy_text)
    print(f"   High entropy text: {high_entropy:.4f}")
    
    # Log some batches
    print("\n📝 Logging sample batches...")
    batch1 = logger.log_batch_entropy(
        batch_id="batch_001",
        entropy=low_entropy,
        batch_size=32,
        data_type="text",
        metadata={"source": "training", "epoch": 1}
    )
    print(f"   Logged batch 1: {batch1}")
    
    batch2 = logger.log_batch_entropy(
        batch_id="batch_002",
        entropy=high_entropy,
        batch_size=32,
        data_type="text",
        metadata={"source": "training", "epoch": 1}
    )
    print(f"   Logged batch 2: {batch2}")
    
    # Generate sample feature data
    print("\n📊 Testing feature entropy calculation...")
    
    # Low entropy features (similar values)
    low_entropy_features = [1.0] * 100 + [1.1] * 50  # Mostly same values
    low_feat_entropy = logger.calculate_feature_entropy(low_entropy_features)
    print(f"   Low entropy features: {low_feat_entropy:.4f}")
    
    # High entropy features (diverse values)
    if NUMPY_AVAILABLE:
        high_entropy_features = np.random.normal(0, 1, 150)  # Normal distribution
    else:
        high_entropy_features = [i * 0.1 for i in range(150)]  # Linear distribution
    high_feat_entropy = logger.calculate_feature_entropy(high_entropy_features)
    print(f"   High entropy features: {high_feat_entropy:.4f}")
    
    # Log feature batches
    batch3 = logger.log_batch_entropy(
        batch_id="batch_003",
        entropy=low_feat_entropy,
        batch_size=64,
        data_type="tabular",
        metadata={"source": "validation", "fold": 1}
    )
    print(f"   Logged batch 3: {batch3}")
    
    batch4 = logger.log_batch_entropy(
        batch_id="batch_004",
        entropy=high_feat_entropy,
        batch_size=64,
        data_type="tabular",
        metadata={"source": "validation", "fold": 1}
    )
    print(f"   Logged batch 4: {batch4}")
    
    # Get statistics
    print("\n📈 Getting entropy statistics...")
    overall_stats = logger.get_entropy_statistics()
    print(f"   Overall stats: {overall_stats['count']} batches, mean entropy = {overall_stats['mean_entropy']:.4f}")
    
    text_stats = logger.get_entropy_statistics("text")
    print(f"   Text stats: {text_stats['count']} batches, mean entropy = {text_stats['mean_entropy']:.4f}")
    
    tabular_stats = logger.get_entropy_statistics("tabular")
    print(f"   Tabular stats: {tabular_stats['count']} batches, mean entropy = {tabular_stats['mean_entropy']:.4f}")
    
    # Test curriculum analysis
    print("\n🎓 Testing curriculum analysis...")
    difficulty_low = analyzer.assess_batch_difficulty(low_entropy)
    print(f"   Low entropy batch difficulty: {difficulty_low}")
    
    difficulty_high = analyzer.assess_batch_difficulty(high_entropy)
    print(f"   High entropy batch difficulty: {difficulty_high}")
    
    # Get curriculum recommendation
    recommendation = analyzer.get_curriculum_recommendation()
    print(f"   Curriculum recommendation: {recommendation['recommendation']}")
    print(f"   Confidence: {recommendation['confidence']:.4f}")
    
    # Export report
    report_success = logger.export_entropy_report("entropy_report.json")
    print(f"✅ Report export: {'SUCCESS' if report_success else 'FAILED'}")
    
    print("\n" + "=" * 60)
    print("INPUT ENTROPY LOGGING DEMO SUMMARY")
    print("=" * 60)
    print("Key Features Implemented:")
    print("  1. Text entropy calculation using token frequency")
    print("  2. Feature entropy calculation using histogram binning")
    print("  3. Batch-level entropy logging with metadata")
    print("  4. Statistical analysis of entropy trends")
    print("  5. Curriculum difficulty assessment")
    print("  6. Automated recommendations for curriculum adjustment")
    print("  7. Comprehensive reporting and export")
    print("\nBenefits:")
    print("  - Data-driven curriculum scheduling")
    print("  - Automated difficulty assessment")
    print("  - Insight into data complexity patterns")
    print("  - Improved training efficiency")
    print("  - Better model convergence monitoring")
    
    print("\n✅ Input Entropy Logging demonstration completed!")


if __name__ == "__main__":
    demo_input_entropy_logging()