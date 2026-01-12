"""
Bias/Drift Detection for Auto-Generated Data in MAHIA-X
This module implements bias and drift detection for auto-generated data with holdout validation.
"""

import math
import time
import json
import os
from typing import Dict, Any, Optional, List, Union
from collections import OrderedDict, defaultdict
from scipy import stats
from datetime import datetime

# Conditional imports with fallbacks
NUMPY_AVAILABLE = False
np = None

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    pass

SCIPY_AVAILABLE = False
stats = None

try:
    from scipy import stats
    SCIPY_AVAILABLE = True
except ImportError:
    pass

class BiasDriftDetector:
    """Bias and drift detector for auto-generated data"""
    
    def __init__(self, holdout_ratio: float = 0.1, validation_window: int = 1000):
        """
        Initialize bias and drift detector
        
        Args:
            holdout_ratio: Ratio of data to hold out for validation
            validation_window: Size of validation window for drift detection
        """
        self.holdout_ratio = holdout_ratio
        self.validation_window = validation_window
        self.holdout_data = []
        self.validation_metrics = OrderedDict()
        self.alerts = []
        
        print(f"✅ BiasDriftDetector initialized with holdout ratio: {holdout_ratio}")
        
    def add_data_sample(self, data_sample: Dict[str, Any], is_holdout: bool = False):
        """
        Add data sample for bias/drift detection
        
        Args:
            data_sample: Data sample with features and labels
            is_holdout: Whether this is holdout data (reference distribution)
        """
        # Add to holdout set if specified
        if is_holdout:
            self.holdout_data.append(data_sample)
            
            # Keep only recent samples (last validation_window)
            if len(self.holdout_data) > self.validation_window:
                self.holdout_data = self.holdout_data[-self.validation_window:]
                
    def detect_bias(self, current_data: List[Dict[str, Any]], 
                   reference_data: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
        """
        Detect bias in current data compared to reference data
        
        Args:
            current_data: Current data samples
            reference_data: Reference data (uses holdout if None)
            
        Returns:
            Bias detection results
        """
        # Use holdout data if reference not provided
        if reference_data is None:
            reference_data = self.holdout_data
            
        if not reference_data or not current_data:
            return {"bias_detected": False, "confidence": 0.0, "details": "No data available"}
            
        # Extract labels for bias detection
        ref_labels = [sample.get("label") for sample in reference_data if sample.get("label") is not None]
        curr_labels = [sample.get("label") for sample in current_data if sample.get("label") is not None]
        
        if not ref_labels or not curr_labels:
            return {"bias_detected": False, "confidence": 0.0, "details": "No labels found"}
            
        # Calculate label distributions
        ref_dist = self._calculate_label_distribution(ref_labels)
        curr_dist = self._calculate_label_distribution(curr_labels)
        
        # Compare distributions using statistical tests
        bias_results = self._compare_distributions_statistical(ref_labels, curr_labels)
        
        # Calculate distribution difference
        distribution_diff = self._calculate_distribution_difference(ref_dist, curr_dist)
        
        # Determine if bias is significant
        bias_detected = bias_results["p_value"] < 0.05 or distribution_diff > 0.1
        
        result = {
            "bias_detected": bias_detected,
            "confidence": 1.0 - min(bias_results["p_value"], distribution_diff),
            "p_value": bias_results["p_value"],
            "test_statistic": bias_results["test_statistic"],
            "test_type": bias_results["test_type"],
            "reference_distribution": ref_dist,
            "current_distribution": curr_dist,
            "distribution_difference": distribution_diff,
            "sample_count": {
                "reference": len(ref_labels),
                "current": len(curr_labels)
            }
        }
        
        # Store validation metric
        metric_id = f"bias_check_{int(time.time() * 1000)}"
        self.validation_metrics[metric_id] = result
        
        # Generate alert if bias detected
        if bias_detected:
            alert = {
                "type": "bias_detected",
                "timestamp": time.time(),
                "severity": "warning",
                "details": result
            }
            self.alerts.append(alert)
            
        return result
        
    def _calculate_label_distribution(self, labels: List) -> Dict[Any, float]:
        """Calculate label distribution"""
        if not labels:
            return {}
            
        total = len(labels)
        distribution = defaultdict(int)
        
        for label in labels:
            distribution[label] += 1
            
        # Normalize
        normalized_dist = {label: count / total for label, count in distribution.items()}
        return normalized_dist
        
    def _compare_distributions_statistical(self, dist1: List, dist2: List) -> Dict[str, Any]:
        """Compare two distributions using statistical tests"""
        try:
            # Use Chi-square test for categorical data
            if SCIPY_AVAILABLE and hasattr(stats, 'chi2_contingency'):
                # Create contingency table
                all_labels = list(set(dist1 + dist2))
                contingency = np.array([
                    [dist1.count(label) for label in all_labels],
                    [dist2.count(label) for label in all_labels]
                ])
                
                chi2, p_value, dof, expected = stats.chi2_contingency(contingency)
                return {
                    "test_statistic": chi2,
                    "p_value": p_value,
                    "test_type": "chi_square"
                }
            else:
                # Fallback to simple comparison
                return {
                    "test_statistic": 0.0,
                    "p_value": 0.5,  # Neutral p-value
                    "test_type": "fallback"
                }
        except Exception as e:
            # Fallback if statistical test fails
            return {
                "test_statistic": 0.0,
                "p_value": 0.5,  # Neutral p-value
                "test_type": "fallback",
                "error": str(e)
            }
            
    def _calculate_distribution_difference(self, dist1: Dict[Any, float], 
                                        dist2: Dict[Any, float]) -> float:
        """Calculate difference between two distributions using Jensen-Shannon divergence"""
        try:
            # Get all unique keys
            all_keys = set(dist1.keys()) | set(dist2.keys())
            
            # Calculate Jensen-Shannon divergence
            js_divergence = 0.0
            for key in all_keys:
                p1 = dist1.get(key, 0.0)
                p2 = dist2.get(key, 0.0)
                m = (p1 + p2) / 2.0
                
                if p1 > 0 and m > 0:
                    js_divergence += p1 * np.log(p1 / m) if NUMPY_AVAILABLE else p1 * math.log(p1 / m)
                if p2 > 0 and m > 0:
                    js_divergence += p2 * np.log(p2 / m) if NUMPY_AVAILABLE else p2 * math.log(p2 / m)
                    
            js_divergence /= 2.0
            return js_divergence
        except Exception as e:
            # Fallback to simple difference
            return 0.1
            
    def detect_drift(self, current_features: List[List[float]], 
                    reference_features: Optional[List[List[float]]] = None) -> Dict[str, Any]:
        """
        Detect data drift in features
        
        Args:
            current_features: Current feature vectors
            reference_features: Reference features (uses holdout if None)
            
        Returns:
            Drift detection results
        """
        # Use holdout data if reference not provided
        if reference_features is None:
            reference_features = [sample.get("features", []) for sample in self.holdout_data 
                                if sample.get("features") is not None]
                                
        if not reference_features or not current_features:
            return {"drift_detected": False, "drift_score": 0.0, "details": "No features available"}
            
        # Convert to numpy arrays if possible
        if NUMPY_AVAILABLE:
            try:
                ref_array = np.array(reference_features)
                curr_array = np.array(current_features)
            except Exception:
                # Fallback to lists
                ref_array = reference_features
                curr_array = current_features
        else:
            ref_array = reference_features
            curr_array = current_features
            
        # Calculate drift using statistical tests
        drift_results = self._detect_feature_drift(ref_array, curr_array)
        
        # Determine if drift is significant
        drift_detected = drift_results["max_p_value"] < 0.05 or drift_results["mean_drift_score"] > 0.1
        
        result = {
            "drift_detected": drift_detected,
            "confidence": 1.0 - min(drift_results["max_p_value"], drift_results["mean_drift_score"]),
            "mean_drift_score": drift_results["mean_drift_score"],
            "max_p_value": drift_results["max_p_value"],
            "feature_drift_scores": drift_results["feature_drift_scores"],
            "sample_count": {
                "reference": len(reference_features),
                "current": len(current_features)
            }
        }
        
        # Store validation metric
        metric_id = f"drift_check_{int(time.time() * 1000)}"
        self.validation_metrics[metric_id] = result
        
        # Generate alert if drift detected
        if drift_detected:
            alert = {
                "type": "drift_detected",
                "timestamp": time.time(),
                "severity": "warning",
                "details": result
            }
            self.alerts.append(alert)
            
        return result
        
    def _detect_feature_drift(self, ref_features, curr_features) -> Dict[str, Any]:
        """Detect drift in individual features"""
        feature_drift_scores = []
        p_values = []
        
        # Determine number of features
        if NUMPY_AVAILABLE and hasattr(ref_features, 'shape'):
            num_features = ref_features.shape[1] if len(ref_features.shape) > 1 else 1
        else:
            num_features = len(ref_features[0]) if ref_features and len(ref_features) > 0 else 1
            
        # Test each feature for drift
        for i in range(num_features):
            try:
                # Extract feature column
                if NUMPY_AVAILABLE and hasattr(ref_features, 'shape'):
                    ref_feature = ref_features[:, i] if len(ref_features.shape) > 1 else ref_features
                    curr_feature = curr_features[:, i] if len(curr_features.shape) > 1 else curr_features
                else:
                    ref_feature = [sample[i] if i < len(sample) else 0 for sample in ref_features]
                    curr_feature = [sample[i] if i < len(sample) else 0 for sample in curr_features]
                    
                # Use Kolmogorov-Smirnov test for continuous features
                if SCIPY_AVAILABLE and hasattr(stats, 'ks_2samp'):
                    ks_statistic, p_value = stats.ks_2samp(ref_feature, curr_feature)
                    drift_score = ks_statistic
                    p_values.append(p_value)
                else:
                    # Fallback to mean difference
                    ref_mean = np.mean(ref_feature) if NUMPY_AVAILABLE else sum(ref_feature) / len(ref_feature)
                    curr_mean = np.mean(curr_feature) if NUMPY_AVAILABLE else sum(curr_feature) / len(curr_feature)
                    drift_score = abs(ref_mean - curr_mean)
                    p_values.append(0.1)  # Neutral p-value
                    
                feature_drift_scores.append(drift_score)
                
            except Exception as e:
                # Fallback if feature test fails
                feature_drift_scores.append(0.0)
                p_values.append(1.0)
                
        # Calculate summary statistics
        mean_drift_score = np.mean(feature_drift_scores) if NUMPY_AVAILABLE and feature_drift_scores else 0.0
        max_p_value = max(p_values) if p_values else 1.0
        
        return {
            "feature_drift_scores": feature_drift_scores,
            "p_values": p_values,
            "mean_drift_score": mean_drift_score,
            "max_p_value": max_p_value
        }
        
    def get_validation_metrics(self) -> Dict[str, Any]:
        """Get validation metrics"""
        return dict(self.validation_metrics)
        
    def get_alerts(self, severity: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Get alerts
        
        Args:
            severity: Filter by severity (warning, error, info)
            
        Returns:
            List of alerts
        """
        if severity:
            return [alert for alert in self.alerts if alert.get("severity") == severity]
        return self.alerts
        
    def clear_validation_metrics(self):
        """Clear validation metrics"""
        self.validation_metrics.clear()
        
    def clear_alerts(self):
        """Clear alerts"""
        self.alerts.clear()
        
    def export_validation_report(self, report_file: str) -> bool:
        """
        Export validation report to file
        
        Args:
            report_file: File to export report to
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Prepare report
            report = {
                "generated_at": datetime.now().isoformat(),
                "validation_metrics": dict(self.validation_metrics),
                "alerts": self.alerts,
                "summary": {
                    "total_metrics": len(self.validation_metrics),
                    "total_alerts": len(self.alerts),
                    "bias_detections": len([m for m in self.validation_metrics.values() if m.get("bias_detected")]),
                    "drift_detections": len([m for m in self.validation_metrics.values() if m.get("drift_detected")])
                }
            }
            
            # Write report
            with open(report_file, 'w') as f:
                json.dump(report, f, indent=2, default=str)
                
            print(f"✅ Validation report exported to: {report_file}")
            return True
        except Exception as e:
            print(f"❌ Failed to export validation report: {e}")
            return False


class HoldoutValidator:
    """Holdout validator for continuous validation of auto-generated data"""
    
    def __init__(self, holdout_size: int = 1000, validation_frequency: int = 100):
        """
        Initialize holdout validator
        
        Args:
            holdout_size: Size of holdout set
            validation_frequency: How often to run validation (every N samples)
        """
        self.holdout_size = holdout_size
        self.validation_frequency = validation_frequency
        self.holdout_buffer = []
        self.validation_count = 0
        self.detector = BiasDriftDetector()
        
        print(f"✅ HoldoutValidator initialized with size: {holdout_size}")
        
    def add_sample(self, sample: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Add sample to holdout buffer and validate if needed
        
        Args:
            sample: Data sample
            
        Returns:
            Validation results if validation was run, None otherwise
        """
        self.holdout_buffer.append(sample)
        
        # Keep only recent samples
        if len(self.holdout_buffer) > self.holdout_size:
            self.holdout_buffer = self.holdout_buffer[-self.holdout_size:]
            
        # Run validation if needed
        self.validation_count += 1
        if self.validation_count % self.validation_frequency == 0:
            return self._run_validation()
            
        return None
        
    def _run_validation(self) -> Dict[str, Any]:
        """Run validation on holdout buffer"""
        if len(self.holdout_buffer) < 10:  # Need minimum samples
            return {"status": "insufficient_data"}
            
        # Split data for validation
        split_point = len(self.holdout_buffer) // 2
        reference_data = self.holdout_buffer[:split_point]
        current_data = self.holdout_buffer[split_point:]
        
        # Run bias detection
        bias_results = self.detector.detect_bias(current_data, reference_data)
        
        # Run drift detection if features available
        ref_features = [sample.get("features", []) for sample in reference_data if sample.get("features")]
        curr_features = [sample.get("features", []) for sample in current_data if sample.get("features")]
        
        if ref_features and curr_features:
            drift_results = self.detector.detect_drift(curr_features, ref_features)
        else:
            drift_results = {"drift_detected": False, "drift_score": 0.0}
            
        results = {
            "timestamp": time.time(),
            "sample_count": len(self.holdout_buffer),
            "bias_results": bias_results,
            "drift_results": drift_results,
            "validation_id": f"val_{int(time.time() * 1000)}"
        }
        
        return results
        
    def get_detector(self) -> BiasDriftDetector:
        """Get bias/drift detector"""
        return self.detector


def demo_bias_drift_detection():
    """Demonstrate bias and drift detection functionality"""
    print("🚀 Demonstrating Bias/Drift Detection for Auto-Generated Data...")
    print("=" * 60)
    
    # Create bias/drift detector
    detector = BiasDriftDetector(holdout_ratio=0.15, validation_window=500)
    print("✅ Created bias/drift detector")
    
    # Create holdout validator
    validator = HoldoutValidator(holdout_size=200, validation_frequency=50)
    print("✅ Created holdout validator")
    
    # Add some holdout data (reference distribution)
    print("🔄 Adding reference data to holdout set...")
    for i in range(100):
        # Balanced 3-class distribution
        sample = {
            "label": i % 3,
            "features": [i * 0.1, i * 0.2, i * 0.3, np.random.random() if NUMPY_AVAILABLE else 0.5],
            "source": "reference"
        }
        detector.add_data_sample(sample, is_holdout=True)
    print("✅ Added 100 reference samples to holdout set")
    
    # Test bias detection with balanced data
    print("\n🔍 Testing bias detection with balanced data...")
    balanced_data = []
    for i in range(60):
        sample = {
            "label": i % 3,  # Balanced 3-class
            "features": [i * 0.15, i * 0.25, i * 0.35, np.random.random() if NUMPY_AVAILABLE else 0.5],
            "source": "current"
        }
        balanced_data.append(sample)
        
        # Add to validator
        validation_result = validator.add_sample(sample)
        if validation_result and validation_result.get("status") != "insufficient_data":
            print(f"   Validation result: Bias={validation_result['bias_results']['bias_detected']}, "
                  f"Drift={validation_result['drift_results']['drift_detected']}")
    
    bias_result_balanced = detector.detect_bias(balanced_data)
    print(f"✅ Bias detection (balanced): {'BIAS DETECTED' if bias_result_balanced['bias_detected'] else 'NO BIAS'}")
    print(f"   P-value: {bias_result_balanced['p_value']:.4f}")
    print(f"   Confidence: {bias_result_balanced['confidence']:.4f}")
    
    # Test bias detection with imbalanced data
    print("\n🔍 Testing bias detection with imbalanced data...")
    imbalanced_data = []
    for i in range(60):
        # Imbalanced: mostly class 0
        label = 0 if i < 50 else (1 if i < 55 else 2)
        sample = {
            "label": label,
            "features": [i * 0.12, i * 0.22, i * 0.32, np.random.random() if NUMPY_AVAILABLE else 0.5],
            "source": "current_imbalanced"
        }
        imbalanced_data.append(sample)
    
    bias_result_imbalanced = detector.detect_bias(imbalanced_data)
    print(f"✅ Bias detection (imbalanced): {'BIAS DETECTED' if bias_result_imbalanced['bias_detected'] else 'NO BIAS'}")
    print(f"   P-value: {bias_result_imbalanced['p_value']:.4f}")
    print(f"   Confidence: {bias_result_imbalanced['confidence']:.4f}")
    
    # Test drift detection
    print("\n🔍 Testing drift detection...")
    # Reference features (normal distribution)
    ref_features = []
    for i in range(80):
        if NUMPY_AVAILABLE:
            features = [np.random.normal(0, 1), np.random.normal(2, 0.5), np.random.normal(-1, 1.5)]
        else:
            features = [0.5, 2.2, -0.8]  # Fixed values if numpy not available
        ref_features.append(features)
        
    # Current features (shifted distribution)
    curr_features = []
    for i in range(80):
        if NUMPY_AVAILABLE:
            features = [np.random.normal(0.5, 1), np.random.normal(2.5, 0.5), np.random.normal(-0.5, 1.5)]
        else:
            features = [0.8, 2.7, -0.3]  # Shifted values if numpy not available
        curr_features.append(features)
    
    drift_result = detector.detect_drift(curr_features, ref_features)
    print(f"✅ Drift detection: {'DRIFT DETECTED' if drift_result['drift_detected'] else 'NO DRIFT'}")
    print(f"   Mean drift score: {drift_result['mean_drift_score']:.4f}")
    print(f"   Max p-value: {drift_result['max_p_value']:.4f}")
    print(f"   Confidence: {drift_result['confidence']:.4f}")
    
    # Show validation metrics
    metrics = detector.get_validation_metrics()
    print(f"\n📊 Validation metrics recorded: {len(metrics)}")
    
    # Show alerts
    alerts = detector.get_alerts()
    print(f"🚨 Alerts generated: {len(alerts)}")
    
    # Export validation report
    report_success = detector.export_validation_report("validation_report.json")
    print(f"✅ Report export: {'SUCCESS' if report_success else 'FAILED'}")
    
    print("\n" + "=" * 60)
    print("BIAS/DRIFT DETECTION DEMO SUMMARY")
    print("=" * 60)
    print("Key Features Implemented:")
    print("  1. Statistical bias detection using Chi-square tests")
    print("  2. Feature drift detection using Kolmogorov-Smirnov tests")
    print("  3. Holdout validation for reference distribution")
    print("  4. Continuous monitoring with configurable frequency")
    print("  5. Automated alert generation for detected issues")
    print("  6. Comprehensive reporting and export")
    print("\nBenefits:")
    print("  - Early detection of data quality issues")
    print("  - Statistical validation of generated data")
    print("  - Automated monitoring of data distributions")
    print("  - Compliance with data quality standards")
    print("  - Reduced manual validation effort")
    
    print("\n✅ Bias/Drift Detection demonstration completed!")


if __name__ == "__main__":
    demo_bias_drift_detection()