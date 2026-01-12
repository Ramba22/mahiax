"""
Bias Detection Toolkit for MAHIA
Automated Gender/Ethics Bias analysis for AI models and datasets
"""

import math
import time
import json
import os
from typing import Dict, Any, Optional, List, Union, Tuple
from collections import OrderedDict, defaultdict
from scipy import stats
import re

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

class BiasDetectionToolkit:
    """Comprehensive toolkit for automated bias detection in AI models and datasets"""
    
    def __init__(self, 
                 sensitivity_threshold: float = 0.05,
                 confidence_threshold: float = 0.95,
                 demographic_categories: Optional[List[str]] = None):
        """
        Initialize bias detection toolkit
        
        Args:
            sensitivity_threshold: P-value threshold for statistical significance
            confidence_threshold: Confidence threshold for bias detection
            demographic_categories: List of demographic categories to analyze
        """
        self.sensitivity_threshold = sensitivity_threshold
        self.confidence_threshold = confidence_threshold
        
        # Default demographic categories
        self.demographic_categories = demographic_categories or [
            "gender", "race", "age", "religion", "sexual_orientation", 
            "socioeconomic_status", "disability_status"
        ]
        
        # Bias detection results storage
        self.bias_analysis_results = OrderedDict()
        self.bias_alerts = []
        
        # Protected groups for different categories
        self.protected_groups = {
            "gender": ["male", "female", "non_binary", "other"],
            "race": ["white", "black", "asian", "hispanic", "other"],
            "age": ["young", "middle_aged", "senior"],
            "religion": ["christian", "muslim", "jewish", "hindu", "buddhist", "atheist", "other"],
            "sexual_orientation": ["heterosexual", "homosexual", "bisexual", "other"],
            "socioeconomic_status": ["low", "middle", "high"],
            "disability_status": ["abled", "disabled"]
        }
        
        print("✅ BiasDetectionToolkit initialized")
        print(f"   Sensitivity threshold: {sensitivity_threshold}")
        print(f"   Confidence threshold: {confidence_threshold}")
        print(f"   Demographic categories: {len(self.demographic_categories)}")
        
    def analyze_text_bias(self, 
                         texts: List[str], 
                         labels: List[Any],
                         demographic_metadata: List[Dict[str, str]]) -> Dict[str, Any]:
        """
        Analyze bias in text data based on demographic metadata
        
        Args:
            texts: List of text samples
            labels: List of corresponding labels/predictions
            demographic_metadata: List of demographic information for each sample
            
        Returns:
            Dictionary with bias analysis results
        """
        if not texts or not labels or not demographic_metadata:
            return {"error": "Empty input data"}
            
        if len(texts) != len(labels) or len(texts) != len(demographic_metadata):
            return {"error": "Input data length mismatch"}
            
        analysis_id = f"text_bias_analysis_{int(time.time() * 1000)}"
        start_time = time.time()
        
        # Initialize results
        results = {
            "analysis_id": analysis_id,
            "timestamp": start_time,
            "sample_count": len(texts),
            "demographics_analyzed": [],
            "bias_metrics": {},
            "recommendations": []
        }
        
        # Analyze each demographic category
        for category in self.demographic_categories:
            if category in self.protected_groups:
                category_results = self._analyze_demographic_bias(
                    category, texts, labels, demographic_metadata
                )
                results["bias_metrics"][category] = category_results
                results["demographics_analyzed"].append(category)
                
        # Calculate overall bias score
        results["overall_bias_score"] = self._calculate_overall_bias_score(results["bias_metrics"])
        
        # Generate recommendations
        results["recommendations"] = self._generate_bias_recommendations(results["bias_metrics"])
        
        # Store results
        self.bias_analysis_results[analysis_id] = results
        
        # Generate alerts for significant bias
        self._generate_bias_alerts(analysis_id, results)
        
        analysis_time = time.time() - start_time
        print(f"✅ Text bias analysis completed in {analysis_time:.2f}s")
        print(f"   Analyzed {len(texts)} samples across {len(results['demographics_analyzed'])} demographics")
        print(f"   Overall bias score: {results['overall_bias_score']:.3f}")
        
        return results
        
    def _analyze_demographic_bias(self, 
                                category: str,
                                texts: List[str],
                                labels: List[Any],
                                demographic_metadata: List[Dict[str, str]]) -> Dict[str, Any]:
        """
        Analyze bias for a specific demographic category
        
        Args:
            category: Demographic category to analyze
            texts: List of text samples
            labels: List of corresponding labels/predictions
            demographic_metadata: List of demographic information
            
        Returns:
            Dictionary with category-specific bias metrics
        """
        try:
            # Group data by demographic values
            demographic_groups = defaultdict(list)
            
            for i, metadata in enumerate(demographic_metadata):
                if category in metadata:
                    group_value = metadata[category]
                    demographic_groups[group_value].append({
                        "text": texts[i],
                        "label": labels[i],
                        "index": i
                    })
                    
            if len(demographic_groups) < 2:
                return {
                    "status": "insufficient_data",
                    "message": f"Insufficient demographic groups for {category}"
                }
                
            # Calculate metrics for each group
            group_metrics = {}
            all_labels = []
            
            for group_name, group_data in demographic_groups.items():
                group_labels = [item["label"] for item in group_data]
                group_metrics[group_name] = {
                    "sample_count": len(group_data),
                    "labels": group_labels,
                    "label_distribution": self._calculate_label_distribution(group_labels),
                    "average_label": self._calculate_average_label(group_labels) if group_labels else 0
                }
                all_labels.extend(group_labels)
                
            # Perform statistical tests
            statistical_results = self._perform_statistical_tests(
                list(demographic_groups.keys()),
                [group_metrics[group]["labels"] for group in demographic_groups.keys()]
            )
            
            # Calculate bias metrics
            bias_metrics = {
                "groups_analyzed": list(demographic_groups.keys()),
                "group_metrics": group_metrics,
                "statistical_tests": statistical_results,
                "representation_balance": self._calculate_representation_balance(demographic_groups),
                "outcome_disparity": self._calculate_outcome_disparity(group_metrics),
                "bias_detected": statistical_results.get("significant_bias", False),
                "bias_confidence": statistical_results.get("max_confidence", 0.0)
            }
            
            return bias_metrics
            
        except Exception as e:
            return {
                "status": "error",
                "message": f"Error analyzing {category} bias: {str(e)}"
            }
            
    def _calculate_label_distribution(self, labels: List[Any]) -> Dict[Any, float]:
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
        
    def _calculate_average_label(self, labels: List[Union[int, float]]) -> float:
        """Calculate average of numeric labels"""
        if not labels:
            return 0.0
            
        try:
            return sum(labels) / len(labels)
        except:
            return 0.0
            
    def _perform_statistical_tests(self, 
                                 group_names: List[str],
                                 group_label_lists: List[List[Any]]) -> Dict[str, Any]:
        """
        Perform statistical tests to detect bias
        
        Args:
            group_names: Names of demographic groups
            group_label_lists: Label lists for each group
            
        Returns:
            Dictionary with statistical test results
        """
        results = {
            "tests_performed": [],
            "p_values": {},
            "test_statistics": {},
            "significant_bias": False,
            "max_confidence": 0.0
        }
        
        try:
            # Convert labels to numeric if possible
            numeric_groups = []
            for labels in group_label_lists:
                try:
                    numeric_labels = [float(label) for label in labels]
                    numeric_groups.append(numeric_labels)
                except:
                    # If conversion fails, skip statistical tests for this category
                    return results
                    
            # Perform ANOVA test if we have 2+ groups
            if len(numeric_groups) >= 2 and SCIPY_AVAILABLE:
                results["tests_performed"].append("anova")
                try:
                    f_stat, p_value = stats.f_oneway(*numeric_groups)
                    results["p_values"]["anova"] = p_value
                    results["test_statistics"]["anova"] = f_stat
                    
                    if p_value < self.sensitivity_threshold:
                        results["significant_bias"] = True
                        results["max_confidence"] = 1.0 - p_value
                except Exception as e:
                    print(f"⚠️  ANOVA test failed: {e}")
                    
            # Perform pairwise t-tests
            if len(numeric_groups) >= 2 and SCIPY_AVAILABLE:
                pairwise_significant = False
                max_pairwise_confidence = 0.0
                
                for i in range(len(numeric_groups)):
                    for j in range(i + 1, len(numeric_groups)):
                        try:
                            t_stat, p_value = stats.ttest_ind(numeric_groups[i], numeric_groups[j])
                            test_name = f"ttest_{group_names[i]}_vs_{group_names[j]}"
                            results["tests_performed"].append(test_name)
                            results["p_values"][test_name] = p_value
                            results["test_statistics"][test_name] = t_stat
                            
                            if p_value < self.sensitivity_threshold:
                                pairwise_significant = True
                                max_pairwise_confidence = max(max_pairwise_confidence, 1.0 - p_value)
                        except Exception as e:
                            print(f"⚠️  Pairwise t-test failed: {e}")
                            
                if pairwise_significant:
                    results["significant_bias"] = True
                    results["max_confidence"] = max(results["max_confidence"], max_pairwise_confidence)
                    
        except Exception as e:
            print(f"⚠️  Statistical tests failed: {e}")
            
        return results
        
    def _calculate_representation_balance(self, 
                                       demographic_groups: Dict[str, List[Dict]]) -> Dict[str, Any]:
        """
        Calculate representation balance across demographic groups
        
        Args:
            demographic_groups: Dictionary mapping group names to sample data
            
        Returns:
            Dictionary with representation balance metrics
        """
        if not demographic_groups:
            return {}
            
        total_samples = sum(len(samples) for samples in demographic_groups.values())
        if total_samples == 0:
            return {}
            
        balance_metrics = {}
        group_sizes = {group: len(samples) for group, samples in demographic_groups.items()}
        
        # Calculate representation ratios
        for group, size in group_sizes.items():
            balance_metrics[group] = {
                "count": size,
                "ratio": size / total_samples,
                "percentage": (size / total_samples) * 100
            }
            
        # Calculate balance score (1 = perfectly balanced)
        expected_ratio = 1.0 / len(demographic_groups)
        balance_score = 1.0 - sum(abs(metrics["ratio"] - expected_ratio) 
                                 for metrics in balance_metrics.values())
        balance_score = max(0.0, balance_score)  # Ensure non-negative
        
        balance_metrics["overall_balance_score"] = balance_score
        balance_metrics["is_balanced"] = balance_score > 0.8  # Threshold for "balanced"
        
        return balance_metrics
        
    def _calculate_outcome_disparity(self, 
                                   group_metrics: Dict[str, Dict]) -> Dict[str, Any]:
        """
        Calculate outcome disparity across groups
        
        Args:
            group_metrics: Metrics for each demographic group
            
        Returns:
            Dictionary with outcome disparity metrics
        """
        if not group_metrics:
            return {}
            
        # Extract average outcomes
        avg_outcomes = {group: metrics["average_label"] 
                       for group, metrics in group_metrics.items()}
        
        if not avg_outcomes:
            return {}
            
        # Calculate disparity metrics
        outcomes = list(avg_outcomes.values())
        max_outcome = max(outcomes)
        min_outcome = min(outcomes)
        
        disparity_metrics = {
            "max_outcome": max_outcome,
            "min_outcome": min_outcome,
            "outcome_range": max_outcome - min_outcome,
            "average_outcome": sum(outcomes) / len(outcomes) if outcomes else 0
        }
        
        # Calculate coefficient of variation (normalized measure of disparity)
        if disparity_metrics["average_outcome"] > 0:
            std_dev = (sum((outcome - disparity_metrics["average_outcome"]) ** 2 
                          for outcome in outcomes) / len(outcomes)) ** 0.5
            disparity_metrics["coefficient_of_variation"] = (
                std_dev / disparity_metrics["average_outcome"]
            )
        else:
            disparity_metrics["coefficient_of_variation"] = 0.0
            
        # Determine if disparity is significant
        disparity_metrics["significant_disparity"] = (
            disparity_metrics["outcome_range"] > 0.1  # Threshold for significance
        )
        
        return disparity_metrics
        
    def _calculate_overall_bias_score(self, 
                                    bias_metrics: Dict[str, Dict]) -> float:
        """
        Calculate overall bias score from individual category metrics
        
        Args:
            bias_metrics: Dictionary of bias metrics by category
            
        Returns:
            Overall bias score (0-1, where 1 indicates high bias)
        """
        if not bias_metrics:
            return 0.0
            
        scores = []
        
        for category, metrics in bias_metrics.items():
            if "bias_confidence" in metrics:
                # Higher confidence in bias detection = higher bias score
                scores.append(metrics["bias_confidence"])
            elif "outcome_disparity" in metrics:
                disparity = metrics["outcome_disparity"]
                if "significant_disparity" in disparity and disparity["significant_disparity"]:
                    # Significant disparity contributes to bias score
                    scores.append(min(1.0, disparity["coefficient_of_variation"]))
                    
        if scores:
            return sum(scores) / len(scores)
        else:
            return 0.0
            
    def _generate_bias_recommendations(self, 
                                     bias_metrics: Dict[str, Dict]) -> List[str]:
        """
        Generate recommendations based on bias analysis results
        
        Args:
            bias_metrics: Dictionary of bias metrics by category
            
        Returns:
            List of recommendation strings
        """
        recommendations = []
        
        for category, metrics in bias_metrics.items():
            if metrics.get("bias_detected", False):
                confidence = metrics.get("bias_confidence", 0)
                if confidence > 0.8:
                    recommendations.append(
                        f"High confidence bias detected in {category} category. "
                        f"Consider collecting more balanced training data."
                    )
                elif confidence > 0.5:
                    recommendations.append(
                        f"Moderate confidence bias detected in {category} category. "
                        f"Review model predictions for this demographic."
                    )
                    
            if "representation_balance" in metrics:
                balance = metrics["representation_balance"]
                if "overall_balance_score" in balance:
                    score = balance["overall_balance_score"]
                    if score < 0.5:
                        recommendations.append(
                            f"Poor representation balance in {category} category. "
                            f"Collect more diverse training data."
                        )
                        
            if "outcome_disparity" in metrics:
                disparity = metrics["outcome_disparity"]
                if disparity.get("significant_disparity", False):
                    recommendations.append(
                        f"Significant outcome disparity detected in {category} category. "
                        f"Consider bias mitigation techniques like reweighting or adversarial debiasing."
                    )
                    
        # General recommendations
        if not recommendations:
            recommendations.append(
                "No significant bias detected at current thresholds. "
                "Continue monitoring for bias in production."
            )
            
        return recommendations
        
    def _generate_bias_alerts(self, 
                            analysis_id: str,
                            results: Dict[str, Any]):
        """
        Generate alerts for significant bias findings
        
        Args:
            analysis_id: ID of the analysis
            results: Analysis results
        """
        overall_score = results.get("overall_bias_score", 0)
        
        if overall_score > 0.7:
            alert = {
                "type": "high_bias_detected",
                "analysis_id": analysis_id,
                "timestamp": time.time(),
                "severity": "critical",
                "score": overall_score,
                "message": f"High bias detected (score: {overall_score:.3f}). Immediate attention required."
            }
            self.bias_alerts.append(alert)
            print(f"🚨 CRITICAL: High bias detected (score: {overall_score:.3f})")
            
        elif overall_score > 0.5:
            alert = {
                "type": "moderate_bias_detected",
                "analysis_id": analysis_id,
                "timestamp": time.time(),
                "severity": "warning",
                "score": overall_score,
                "message": f"Moderate bias detected (score: {overall_score:.3f}). Review recommended."
            }
            self.bias_alerts.append(alert)
            print(f"⚠️  WARNING: Moderate bias detected (score: {overall_score:.3f})")
            
    def analyze_model_predictions(self,
                                true_labels: List[Any],
                                predicted_labels: List[Any],
                                demographic_metadata: List[Dict[str, str]],
                                prediction_scores: Optional[List[float]] = None) -> Dict[str, Any]:
        """
        Analyze bias in model predictions
        
        Args:
            true_labels: Ground truth labels
            predicted_labels: Model predictions
            demographic_metadata: Demographic information for each sample
            prediction_scores: Optional prediction confidence scores
            
        Returns:
            Dictionary with prediction bias analysis
        """
        if not true_labels or not predicted_labels or not demographic_metadata:
            return {"error": "Empty input data"}
            
        if len(true_labels) != len(predicted_labels) or len(true_labels) != len(demographic_metadata):
            return {"error": "Input data length mismatch"}
            
        analysis_id = f"model_bias_analysis_{int(time.time() * 1000)}"
        start_time = time.time()
        
        # Initialize results
        results = {
            "analysis_id": analysis_id,
            "timestamp": start_time,
            "sample_count": len(true_labels),
            "metrics": {},
            "bias_analysis": {}
        }
        
        # Calculate performance metrics
        results["metrics"] = self._calculate_prediction_metrics(
            true_labels, predicted_labels, prediction_scores
        )
        
        # Analyze bias in predictions
        for category in self.demographic_categories:
            if category in self.protected_groups:
                category_bias = self._analyze_prediction_bias_by_category(
                    category, true_labels, predicted_labels, demographic_metadata
                )
                results["bias_analysis"][category] = category_bias
                
        # Calculate fairness metrics
        results["fairness_metrics"] = self._calculate_fairness_metrics(
            true_labels, predicted_labels, demographic_metadata
        )
        
        # Store results
        self.bias_analysis_results[analysis_id] = results
        
        # Generate alerts
        self._generate_model_bias_alerts(analysis_id, results)
        
        analysis_time = time.time() - start_time
        print(f"✅ Model bias analysis completed in {analysis_time:.2f}s")
        
        return results
        
    def _calculate_prediction_metrics(self,
                                   true_labels: List[Any],
                                   predicted_labels: List[Any],
                                   prediction_scores: Optional[List[float]] = None) -> Dict[str, float]:
        """Calculate standard prediction metrics"""
        metrics = {}
        
        try:
            # Accuracy
            correct = sum(1 for true, pred in zip(true_labels, predicted_labels) if true == pred)
            metrics["accuracy"] = correct / len(true_labels) if true_labels else 0
            
            # If we have numeric labels, calculate additional metrics
            try:
                numeric_true = [float(label) for label in true_labels]
                numeric_pred = [float(label) for label in predicted_labels]
                
                # MSE
                mse = sum((t - p) ** 2 for t, p in zip(numeric_true, numeric_pred)) / len(numeric_true)
                metrics["mse"] = mse
                
                # MAE
                mae = sum(abs(t - p) for t, p in zip(numeric_true, numeric_pred)) / len(numeric_true)
                metrics["mae"] = mae
                
            except:
                pass  # Non-numeric labels
                
        except Exception as e:
            print(f"⚠️  Error calculating prediction metrics: {e}")
            
        return metrics
        
    def _analyze_prediction_bias_by_category(self,
                                           category: str,
                                           true_labels: List[Any],
                                           predicted_labels: List[Any],
                                           demographic_metadata: List[Dict[str, str]]) -> Dict[str, Any]:
        """Analyze prediction bias for a specific demographic category"""
        try:
            # Group by demographic values
            demographic_groups = defaultdict(list)
            
            for i, metadata in enumerate(demographic_metadata):
                if category in metadata:
                    group_value = metadata[category]
                    demographic_groups[group_value].append({
                        "true": true_labels[i],
                        "pred": predicted_labels[i],
                        "index": i
                    })
                    
            if len(demographic_groups) < 2:
                return {"status": "insufficient_data"}
                
            # Calculate metrics per group
            group_metrics = {}
            
            for group_name, group_data in demographic_groups.items():
                group_true = [item["true"] for item in group_data]
                group_pred = [item["pred"] for item in group_data]
                
                group_metrics[group_name] = self._calculate_prediction_metrics(group_true, group_pred)
                
            # Compare group metrics
            bias_analysis = {
                "groups": list(demographic_groups.keys()),
                "group_metrics": group_metrics,
                "accuracy_disparity": self._calculate_accuracy_disparity(group_metrics),
                "performance_gap": self._calculate_performance_gap(group_metrics)
            }
            
            return bias_analysis
            
        except Exception as e:
            return {"error": f"Error analyzing {category} bias: {str(e)}"}
            
    def _calculate_accuracy_disparity(self, group_metrics: Dict[str, Dict]) -> float:
        """Calculate accuracy disparity across groups"""
        if not group_metrics:
            return 0.0
            
        accuracies = [metrics.get("accuracy", 0) for metrics in group_metrics.values()]
        if not accuracies:
            return 0.0
            
        return max(accuracies) - min(accuracies)
        
    def _calculate_performance_gap(self, group_metrics: Dict[str, Dict]) -> Dict[str, Any]:
        """Calculate performance gap metrics"""
        if not group_metrics:
            return {}
            
        accuracies = [metrics.get("accuracy", 0) for metrics in group_metrics.values()]
        if not accuracies:
            return {}
            
        best_group = max(group_metrics.keys(), key=lambda k: group_metrics[k].get("accuracy", 0))
        worst_group = min(group_metrics.keys(), key=lambda k: group_metrics[k].get("accuracy", 0))
        
        return {
            "best_performing_group": best_group,
            "worst_performing_group": worst_group,
            "performance_gap": max(accuracies) - min(accuracies),
            "relative_gap": (max(accuracies) - min(accuracies)) / max(accuracies) if max(accuracies) > 0 else 0
        }
        
    def _calculate_fairness_metrics(self,
                                  true_labels: List[Any],
                                  predicted_labels: List[Any],
                                  demographic_metadata: List[Dict[str, str]]) -> Dict[str, Any]:
        """Calculate standard fairness metrics"""
        fairness_metrics = {}
        
        try:
            # For binary classification, calculate demographic parity
            # This is a simplified implementation
            unique_labels = list(set(true_labels + predicted_labels))
            if len(unique_labels) == 2:  # Binary classification
                positive_label = unique_labels[1]  # Assume second label is positive
                
                # Group by demographics and calculate positive prediction rates
                demographic_rates = defaultdict(list)
                
                for i, metadata in enumerate(demographic_metadata):
                    pred = predicted_labels[i]
                    for category in self.demographic_categories:
                        if category in metadata:
                            group = metadata[category]
                            demographic_rates[f"{category}_{group}"].append(
                                1 if pred == positive_label else 0
                            )
                            
                # Calculate demographic parity
                positive_rates = {}
                for group, rates in demographic_rates.items():
                    positive_rates[group] = sum(rates) / len(rates) if rates else 0
                    
                if positive_rates:
                    max_rate = max(positive_rates.values())
                    min_rate = min(positive_rates.values())
                    fairness_metrics["demographic_parity_difference"] = max_rate - min_rate
                    
        except Exception as e:
            print(f"⚠️  Error calculating fairness metrics: {e}")
            
        return fairness_metrics
        
    def _generate_model_bias_alerts(self,
                                  analysis_id: str,
                                  results: Dict[str, Any]):
        """Generate alerts for model bias findings"""
        try:
            # Check for high accuracy disparity
            for category, bias_analysis in results.get("bias_analysis", {}).items():
                disparity = bias_analysis.get("accuracy_disparity", 0)
                if disparity > 0.1:  # 10% threshold
                    alert = {
                        "type": "accuracy_disparity",
                        "analysis_id": analysis_id,
                        "category": category,
                        "timestamp": time.time(),
                        "severity": "warning",
                        "disparity": disparity,
                        "message": f"High accuracy disparity ({disparity:.3f}) in {category}"
                    }
                    self.bias_alerts.append(alert)
                    print(f"⚠️  WARNING: High accuracy disparity in {category}: {disparity:.3f}")
                    
            # Check fairness metrics
            fairness_metrics = results.get("fairness_metrics", {})
            dp_diff = fairness_metrics.get("demographic_parity_difference", 0)
            if dp_diff > 0.1:
                alert = {
                    "type": "demographic_parity_violation",
                    "analysis_id": analysis_id,
                    "timestamp": time.time(),
                    "severity": "warning",
                    "dp_difference": dp_diff,
                    "message": f"Demographic parity violation (Δ={dp_diff:.3f})"
                }
                self.bias_alerts.append(alert)
                print(f"⚠️  WARNING: Demographic parity violation: Δ={dp_diff:.3f}")
                
        except Exception as e:
            print(f"⚠️  Error generating model bias alerts: {e}")
            
    def get_bias_alerts(self, severity: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Get bias alerts
        
        Args:
            severity: Filter by severity ("warning", "critical", etc.)
            
        Returns:
            List of bias alerts
        """
        if severity:
            return [alert for alert in self.bias_alerts if alert.get("severity") == severity]
        return self.bias_alerts.copy()
        
    def clear_alerts(self):
        """Clear all bias alerts"""
        self.bias_alerts.clear()
        print("🗑️  Bias alerts cleared")
        
    def export_analysis_report(self, 
                             analysis_id: str, 
                             filepath: str) -> bool:
        """
        Export bias analysis report to JSON file
        
        Args:
            analysis_id: ID of analysis to export
            filepath: Path to save report
            
        Returns:
            True if successful, False otherwise
        """
        try:
            if analysis_id not in self.bias_analysis_results:
                print(f"⚠️  Analysis ID {analysis_id} not found")
                return False
                
            report_data = {
                "toolkit_version": "1.0",
                "export_timestamp": time.time(),
                "analysis_results": self.bias_analysis_results[analysis_id],
                "alerts": self.bias_alerts
            }
            
            with open(filepath, 'w') as f:
                json.dump(report_data, f, indent=2, default=str)
                
            print(f"✅ Bias analysis report exported to {filepath}")
            return True
            
        except Exception as e:
            print(f"❌ Failed to export report: {e}")
            return False
            
    def generate_summary_report(self) -> str:
        """
        Generate a summary report of all bias analyses
        
        Returns:
            Formatted summary report string
        """
        if not self.bias_analysis_results:
            return "No bias analyses performed yet."
            
        report = f"""
📊 BIAS DETECTION TOOLKIT SUMMARY REPORT
=====================================

Total Analyses Performed: {len(self.bias_analysis_results)}
Active Alerts: {len(self.bias_alerts)}

📈 Recent Analyses:
"""
        
        # Show last 3 analyses
        recent_analyses = list(self.bias_analysis_results.items())[-3:]
        for analysis_id, results in recent_analyses:
            timestamp = time.ctime(results.get("timestamp", 0))
            sample_count = results.get("sample_count", 0)
            bias_score = results.get("overall_bias_score", 0)
            report += f"  • {analysis_id}\n"
            report += f"    Time: {timestamp}\n"
            report += f"    Samples: {sample_count:,}\n"
            report += f"    Bias Score: {bias_score:.3f}\n\n"
            
        # Show active alerts
        if self.bias_alerts:
            report += "🚨 Active Alerts:\n"
            for alert in self.bias_alerts[-5:]:  # Show last 5 alerts
                severity = alert.get("severity", "unknown").upper()
                message = alert.get("message", "No message")
                report += f"  • [{severity}] {message}\n"
                
        return report

# Example usage
def example_bias_detection():
    """Example of bias detection toolkit usage"""
    print("🔧 Setting up bias detection toolkit example...")
    
    # Create toolkit
    toolkit = BiasDetectionToolkit(
        sensitivity_threshold=0.05,
        confidence_threshold=0.95
    )
    
    # Create sample data for text bias analysis
    print("\n📝 Analyzing text bias...")
    
    sample_texts = [
        "John is a successful engineer",
        "Mary is a successful engineer", 
        "David is a successful engineer",
        "Sarah is a successful engineer",
        "Michael is a successful engineer",
        "Lisa is a successful engineer"
    ]
    
    sample_labels = [0.8, 0.9, 0.85, 0.7, 0.82, 0.75]  # Confidence scores
    
    sample_metadata = [
        {"gender": "male", "age": "young"},
        {"gender": "female", "age": "young"},
        {"gender": "male", "age": "middle_aged"},
        {"gender": "female", "age": "middle_aged"},
        {"gender": "male", "age": "senior"},
        {"gender": "female", "age": "senior"}
    ]
    
    # Analyze text bias
    text_bias_results = toolkit.analyze_text_bias(
        texts=sample_texts,
        labels=sample_labels,
        demographic_metadata=sample_metadata
    )
    
    print("✅ Text bias analysis completed")
    
    # Analyze model predictions
    print("\n🤖 Analyzing model prediction bias...")
    
    true_labels = [1, 0, 1, 1, 0, 1]
    predicted_labels = [1, 1, 1, 0, 0, 1]  # Some errors
    prediction_scores = [0.9, 0.8, 0.85, 0.7, 0.95, 0.75]
    
    model_bias_results = toolkit.analyze_model_predictions(
        true_labels=true_labels,
        predicted_labels=predicted_labels,
        demographic_metadata=sample_metadata,
        prediction_scores=prediction_scores
    )
    
    print("✅ Model bias analysis completed")
    
    # Print summary report
    print("\n" + "="*60)
    print(toolkit.generate_summary_report())
    
    # Export report
    if text_bias_results.get("analysis_id"):
        toolkit.export_analysis_report(
            text_bias_results["analysis_id"],
            "bias_analysis_report.json"
        )
    
    print("\n✅ Bias detection toolkit example completed!")

if __name__ == "__main__":
    example_bias_detection()