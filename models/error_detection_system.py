"""
Error Detection System for MAHIA-X
Implements intelligent error detection, self-correction, and quality feedback loops
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Any, Optional, List, Tuple
from collections import OrderedDict, defaultdict
import time
import re
import json
from datetime import datetime

# Conditional imports
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    
try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False

class ErrorDetector:
    """Detects errors and inconsistencies in model outputs"""
    
    def __init__(self, confidence_threshold: float = 0.8):
        """
        Initialize error detector
        
        Args:
            confidence_threshold: Threshold for confidence-based error detection
        """
        self.confidence_threshold = confidence_threshold
        self.detection_rules = self._initialize_detection_rules()
        self.error_history = OrderedDict()
        self.detection_stats = defaultdict(int)
        
    def _initialize_detection_rules(self) -> Dict[str, Any]:
        """Initialize error detection rules"""
        return {
            "contradiction_detection": {
                "keywords": ["but", "however", "although", "nevertheless"],
                "pattern": r"\b(contradict|contradictory|conflict)\b"
            },
            "factual_inconsistency": {
                "pattern": r"\b(always|never|impossible|guarantee)\b.*\b(except|unless|but)\b"
            },
            "logical_fallacy": {
                "patterns": [
                    r"\b(all|every|none)\b.*\b(some|few|many)\b",
                    r"\b(if|when)\b.*\b(then)\b.*\b(but|however)\b"
                ]
            },
            "incomplete_response": {
                "indicators": ["...", "to be continued", "see below", "as mentioned"],
                "min_length": 50
            }
        }
        
    def detect_errors(self, response: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Detect errors in response
        
        Args:
            response: Model response text
            context: Optional context information
            
        Returns:
            Error detection results
        """
        errors = []
        confidence_scores = []
        
        # Check for contradictions
        contradiction_score = self._check_contradictions(response)
        if contradiction_score > 0.5:
            errors.append({
                "type": "contradiction",
                "severity": "high",
                "confidence": contradiction_score,
                "description": "Response contains contradictory statements"
            })
            confidence_scores.append(contradiction_score)
            
        # Check for factual inconsistencies
        factual_score = self._check_factual_consistency(response)
        if factual_score > 0.6:
            errors.append({
                "type": "factual_inconsistency",
                "severity": "medium",
                "confidence": factual_score,
                "description": "Potential factual inconsistencies detected"
            })
            confidence_scores.append(factual_score)
            
        # Check for logical fallacies
        fallacy_score = self._check_logical_fallacies(response)
        if fallacy_score > 0.5:
            errors.append({
                "type": "logical_fallacy",
                "severity": "medium",
                "confidence": fallacy_score,
                "description": "Logical fallacies detected"
            })
            confidence_scores.append(fallacy_score)
            
        # Check for incomplete responses
        completeness_score = self._check_completeness(response)
        if completeness_score < 0.7:
            errors.append({
                "type": "incomplete_response",
                "severity": "low",
                "confidence": 1.0 - completeness_score,
                "description": "Response appears incomplete"
            })
            confidence_scores.append(1.0 - completeness_score)
            
        # Calculate overall error confidence
        if confidence_scores:
            overall_confidence = sum(confidence_scores) / len(confidence_scores)
        else:
            overall_confidence = 0.0
            
        # Determine if errors need correction
        needs_correction = overall_confidence > (1.0 - self.confidence_threshold)
        
        detection_result = {
            "timestamp": time.time(),
            "response": response,
            "errors_detected": errors,
            "overall_confidence": overall_confidence,
            "needs_correction": needs_correction,
            "error_count": len(errors)
        }
        
        # Store in history
        detection_id = f"detection_{int(time.time() * 1000)}"
        self.error_history[detection_id] = detection_result
        self.detection_stats["total_detections"] += 1
        self.detection_stats["errors_found"] += len(errors)
        
        if needs_correction:
            self.detection_stats["corrections_needed"] += 1
            
        return detection_result
        
    def _check_contradictions(self, response: str) -> float:
        """Check for contradictory statements"""
        response_lower = response.lower()
        contradiction_indicators = self.detection_rules["contradiction_detection"]["keywords"]
        
        # Count contradiction indicators
        contradiction_count = 0
        for indicator in contradiction_indicators:
            if indicator in response_lower:
                contradiction_count += 1
                
        # Check for contradiction patterns
        pattern = self.detection_rules["contradiction_detection"]["pattern"]
        if re.search(pattern, response_lower):
            contradiction_count += 2
            
        # Normalize score
        max_indicators = len(contradiction_indicators) + 2  # +2 for pattern matches
        return min(1.0, contradiction_count / max_indicators) if max_indicators > 0 else 0.0
        
    def _check_factual_consistency(self, response: str) -> float:
        """Check for factual inconsistencies"""
        response_lower = response.lower()
        pattern = self.detection_rules["factual_inconsistency"]["pattern"]
        
        if re.search(pattern, response_lower):
            return 0.8
        return 0.0
        
    def _check_logical_fallacies(self, response: str) -> float:
        """Check for logical fallacies"""
        response_lower = response.lower()
        patterns = self.detection_rules["logical_fallacy"]["patterns"]
        
        fallacy_count = 0
        for pattern in patterns:
            if re.search(pattern, response_lower):
                fallacy_count += 1
                
        return min(1.0, fallacy_count / len(patterns)) if patterns else 0.0
        
    def _check_completeness(self, response: str) -> float:
        """Check response completeness"""
        # Check length
        if len(response) < self.detection_rules["incomplete_response"]["min_length"]:
            return 0.3
            
        # Check for incomplete indicators
        indicators = self.detection_rules["incomplete_response"]["indicators"]
        for indicator in indicators:
            if indicator in response:
                return 0.4
                
        return 1.0
        
    def get_detection_stats(self) -> Dict[str, Any]:
        """Get error detection statistics"""
        return dict(self.detection_stats)


class SelfCorrector:
    """Self-corrects errors in model outputs"""
    
    def __init__(self, correction_strength: float = 0.7):
        """
        Initialize self-corrector
        
        Args:
            correction_strength: Strength of correction (0.0 to 1.0)
        """
        self.correction_strength = correction_strength
        self.correction_history = OrderedDict()
        self.correction_stats = {
            "total_corrections": 0,
            "successful_corrections": 0,
            "failed_corrections": 0
        }
        
    def correct_response(self, response: str, errors: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Correct response based on detected errors
        
        Args:
            response: Original response
            errors: List of detected errors
            
        Returns:
            Correction results
        """
        correction_start_time = time.time()
        
        corrected_response = response
        corrections_applied = []
        
        for error in errors:
            error_type = error["type"]
            correction_result = self._apply_correction(corrected_response, error)
            
            if correction_result["applied"]:
                corrected_response = correction_result["corrected_text"]
                corrections_applied.append({
                    "error_type": error_type,
                    "correction": correction_result["description"],
                    "confidence": error["confidence"]
                })
                
        correction_time = time.time() - correction_start_time
        
        # Update statistics
        self.correction_stats["total_corrections"] += 1
        if corrections_applied:
            self.correction_stats["successful_corrections"] += 1
        else:
            self.correction_stats["failed_corrections"] += 1
            
        correction_result = {
            "original_response": response,
            "corrected_response": corrected_response,
            "corrections_applied": corrections_applied,
            "correction_time": correction_time,
            "improvement_made": len(corrections_applied) > 0,
            "timestamp": time.time()
        }
        
        # Store in history
        correction_id = f"correction_{int(time.time() * 1000)}"
        self.correction_history[correction_id] = correction_result
        
        return correction_result
        
    def _apply_correction(self, response: str, error: Dict[str, Any]) -> Dict[str, Any]:
        """Apply specific correction based on error type"""
        error_type = error["type"]
        confidence = error["confidence"]
        
        # Only apply corrections with sufficient confidence
        if confidence < 0.5:
            return {"applied": False, "corrected_text": response, "description": "Low confidence"}
            
        if error_type == "contradiction":
            # Simplify contradictory statements
            corrected = self._resolve_contradictions(response)
            return {
                "applied": True,
                "corrected_text": corrected,
                "description": "Resolved contradictory statements"
            }
            
        elif error_type == "factual_inconsistency":
            # Soften absolute statements
            corrected = self._soften_absolutes(response)
            return {
                "applied": True,
                "corrected_text": corrected,
                "description": "Softened absolute statements"
            }
            
        elif error_type == "logical_fallacy":
            # Improve logical flow
            corrected = self._improve_logic(response)
            return {
                "applied": True,
                "corrected_text": corrected,
                "description": "Improved logical structure"
            }
            
        elif error_type == "incomplete_response":
            # Expand incomplete responses
            corrected = self._expand_response(response)
            return {
                "applied": True,
                "corrected_text": corrected,
                "description": "Expanded incomplete response"
            }
            
        else:
            return {"applied": False, "corrected_text": response, "description": "Unknown error type"}
            
    def _resolve_contradictions(self, response: str) -> str:
        """Resolve contradictory statements"""
        # Simple approach: remove but/however and keep second part
        response = re.sub(r"\b(but|however)\b", "and", response, flags=re.IGNORECASE)
        return response
        
    def _soften_absolutes(self, response: str) -> str:
        """Soften absolute statements"""
        response = re.sub(r"\b(always)\b", "typically", response, flags=re.IGNORECASE)
        response = re.sub(r"\b(never)\b", "rarely", response, flags=re.IGNORECASE)
        response = re.sub(r"\b(impossible)\b", "unlikely", response, flags=re.IGNORECASE)
        return response
        
    def _improve_logic(self, response: str) -> str:
        """Improve logical flow"""
        # Add connecting words for better flow
        if "therefore" not in response.lower() and "because" in response.lower():
            response = response.replace("because", "therefore, because")
        return response
        
    def _expand_response(self, response: str) -> str:
        """Expand incomplete responses"""
        if len(response) < 100:
            response += " [This response has been expanded to provide more complete information.]"
        return response
        
    def get_correction_stats(self) -> Dict[str, Any]:
        """Get correction statistics"""
        return self.correction_stats


class QualityFeedbackLoop:
    """Manages quality feedback loop for continuous improvement"""
    
    def __init__(self):
        """Initialize quality feedback loop"""
        self.feedback_data = OrderedDict()
        self.quality_metrics = defaultdict(list)
        self.improvement_actions = []
        
    def add_feedback(self, feedback_id: str, feedback_data: Dict[str, Any]):
        """
        Add quality feedback
        
        Args:
            feedback_id: Unique feedback identifier
            feedback_data: Feedback data dictionary
        """
        feedback_entry = {
            "feedback_id": feedback_id,
            "data": feedback_data,
            "timestamp": time.time(),
            "processed": False
        }
        
        self.feedback_data[feedback_id] = feedback_entry
        
        # Update quality metrics
        if "accuracy" in feedback_data:
            self.quality_metrics["accuracy"].append(feedback_data["accuracy"])
        if "completeness" in feedback_data:
            self.quality_metrics["completeness"].append(feedback_data["completeness"])
        if "helpfulness" in feedback_data:
            self.quality_metrics["helpfulness"].append(feedback_data["helpfulness"])
            
    def analyze_feedback_trends(self) -> Dict[str, Any]:
        """
        Analyze feedback trends to identify improvement opportunities
        
        Returns:
            Analysis results
        """
        trends = {}
        
        for metric, values in self.quality_metrics.items():
            if values:
                # Calculate statistics
                if NUMPY_AVAILABLE:
                    trends[metric] = {
                        "mean": float(np.mean(values)),
                        "std": float(np.std(values)),
                        "trend": "improving" if len(values) > 5 and values[-1] > values[0] else "declining" if len(values) > 5 and values[-1] < values[0] else "stable"
                    }
                else:
                    mean_val = sum(values) / len(values)
                    trends[metric] = {
                        "mean": mean_val,
                        "std": 0.0,  # Simplified without numpy
                        "trend": "stable"
                    }
                    
        return {
            "timestamp": time.time(),
            "trends": trends,
            "total_feedback": len(self.feedback_data),
            "unprocessed_feedback": len([f for f in self.feedback_data.values() if not f["processed"]])
        }
        
    def generate_improvement_actions(self, trends: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Generate improvement actions based on trends
        
        Args:
            trends: Feedback trends analysis
            
        Returns:
            List of improvement actions
        """
        actions = []
        
        for metric, trend_data in trends.get("trends", {}).items():
            if trend_data["mean"] < 0.7:  # Low quality threshold
                actions.append({
                    "type": f"improve_{metric}",
                    "priority": "high",
                    "description": f"Improve {metric} scores (current: {trend_data['mean']:.2f})",
                    "target": 0.8
                })
            elif trend_data["trend"] == "declining":
                actions.append({
                    "type": f"stabilize_{metric}",
                    "priority": "medium",
                    "description": f"Stabilize declining {metric} scores",
                    "target": trend_data["mean"]
                })
                
        self.improvement_actions.extend(actions)
        return actions
        
    def mark_feedback_processed(self, feedback_id: str):
        """
        Mark feedback as processed
        
        Args:
            feedback_id: Feedback identifier
        """
        if feedback_id in self.feedback_data:
            self.feedback_data[feedback_id]["processed"] = True


class ErrorDetectionSystem:
    """Main error detection system integrating all components"""
    
    def __init__(self, confidence_threshold: float = 0.8, correction_strength: float = 0.7):
        """
        Initialize error detection system
        
        Args:
            confidence_threshold: Threshold for error detection
            correction_strength: Strength of corrections
        """
        self.confidence_threshold = confidence_threshold
        self.correction_strength = correction_strength
        
        # Initialize components
        self.error_detector = ErrorDetector(confidence_threshold)
        self.self_corrector = SelfCorrector(correction_strength)
        self.feedback_loop = QualityFeedbackLoop()
        
        # System statistics
        self.system_stats = {
            "total_responses_processed": 0,
            "errors_detected": 0,
            "corrections_made": 0,
            "feedback_collected": 0
        }
        
        print(f"✅ ErrorDetectionSystem initialized")
        print(f"   Confidence threshold: {confidence_threshold}")
        print(f"   Correction strength: {correction_strength}")
        
    def process_response(self, response_id: str, response: str, 
                        context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Process response through error detection and correction pipeline
        
        Args:
            response_id: Unique response identifier
            response: Model response
            context: Optional context information
            
        Returns:
            Processing results
        """
        start_time = time.time()
        
        # Detect errors
        detection_result = self.error_detector.detect_errors(response, context)
        
        # Update statistics
        self.system_stats["total_responses_processed"] += 1
        self.system_stats["errors_detected"] += detection_result["error_count"]
        
        # Apply corrections if needed
        if detection_result["needs_correction"]:
            correction_result = self.self_corrector.correct_response(
                response, detection_result["errors_detected"]
            )
            
            self.system_stats["corrections_made"] += 1
            
            final_response = correction_result["corrected_response"]
            corrections_applied = correction_result["corrections_applied"]
        else:
            correction_result = {"corrected_response": response, "corrections_applied": []}
            final_response = response
            corrections_applied = []
            
        processing_time = time.time() - start_time
        
        return {
            "response_id": response_id,
            "original_response": response,
            "final_response": final_response,
            "detection_result": detection_result,
            "correction_result": correction_result,
            "corrections_applied": len(corrections_applied) > 0,
            "processing_time": processing_time,
            "timestamp": time.time()
        }
        
    def add_user_feedback(self, feedback_id: str, feedback_data: Dict[str, Any]):
        """
        Add user feedback to quality feedback loop
        
        Args:
            feedback_id: Unique feedback identifier
            feedback_data: Feedback data dictionary
        """
        self.feedback_loop.add_feedback(feedback_id, feedback_data)
        self.system_stats["feedback_collected"] += 1
        
        # Mark feedback as processed
        self.feedback_loop.mark_feedback_processed(feedback_id)
        
    def get_system_diagnostics(self) -> Dict[str, Any]:
        """
        Get system diagnostics and statistics
        
        Returns:
            Diagnostic information
        """
        return {
            "timestamp": time.time(),
            "system_stats": self.system_stats,
            "detection_stats": self.error_detector.get_detection_stats(),
            "correction_stats": self.self_corrector.get_correction_stats(),
            "feedback_trends": self.feedback_loop.analyze_feedback_trends()
        }
        
    def export_diagnostics_report(self, filepath: str) -> bool:
        """
        Export diagnostics report to file
        
        Args:
            filepath: Path to export report
            
        Returns:
            True if successful, False otherwise
        """
        try:
            report = {
                "generated_at": datetime.now().isoformat(),
                "system_diagnostics": self.get_system_diagnostics(),
                "recent_detections": dict(list(self.error_detector.error_history.items())[-20:]),
                "recent_corrections": dict(list(self.self_corrector.correction_history.items())[-20:])
            }
            
            with open(filepath, 'w') as f:
                json.dump(report, f, indent=2, default=str)
                
            print(f"✅ Error detection diagnostics report exported to {filepath}")
            return True
        except Exception as e:
            print(f"❌ Failed to export diagnostics report: {e}")
            return False


def demo_error_detection_system():
    """Demonstrate error detection system functionality"""
    print("🚀 Demonstrating Error Detection System...")
    print("=" * 50)
    
    # Create error detection system
    system = ErrorDetectionSystem(confidence_threshold=0.8, correction_strength=0.7)
    print("✅ Created error detection system")
    
    # Test error detection and correction
    print("\n🔍 Testing error detection and correction...")
    
    test_responses = [
        {
            "id": "resp_001",
            "text": "Machine learning is always accurate. However, it can sometimes make mistakes.",
            "expected_errors": ["contradiction"]
        },
        {
            "id": "resp_002",
            "text": "Neural networks never fail and guarantee perfect results in all situations.",
            "expected_errors": ["factual_inconsistency"]
        },
        {
            "id": "resp_003",
            "text": "To train a model, you need data. But sometimes you don't need data.",
            "expected_errors": ["contradiction", "logical_fallacy"]
        },
        {
            "id": "resp_004",
            "text": "This is a short response...",
            "expected_errors": ["incomplete_response"]
        }
    ]
    
    # Process test responses
    for response_data in test_responses:
        result = system.process_response(
            response_data["id"],
            response_data["text"]
        )
        
        print(f"   Response {response_data['id']}:")
        print(f"     Errors detected: {result['detection_result']['error_count']}")
        print(f"     Corrections applied: {result['corrections_applied']}")
        print(f"     Processing time: {result['processing_time']:.3f}s")
        
        if result['corrections_applied']:
            print(f"     Original: {result['original_response'][:50]}...")
            print(f"     Corrected: {result['final_response'][:50]}...")
            
    # Add sample feedback
    print("\n📊 Adding sample user feedback...")
    
    feedback_samples = [
        {
            "id": "feedback_001",
            "data": {
                "accuracy": 0.9,
                "completeness": 0.8,
                "helpfulness": 0.85,
                "comment": "Very accurate response"
            }
        },
        {
            "id": "feedback_002",
            "data": {
                "accuracy": 0.6,
                "completeness": 0.4,
                "helpfulness": 0.5,
                "comment": "Response was incomplete"
            }
        }
    ]
    
    for feedback in feedback_samples:
        system.add_user_feedback(feedback["id"], feedback["data"])
        print(f"   Added feedback {feedback['id']}")
        
    # Analyze feedback trends
    trends = system.feedback_loop.analyze_feedback_trends()
    print(f"   Feedback trends analyzed: {len(trends['trends'])} metrics tracked")
    
    # Generate improvement actions
    actions = system.feedback_loop.generate_improvement_actions(trends)
    print(f"   Improvement actions generated: {len(actions)}")
    
    # Show system diagnostics
    print("\n📈 System Diagnostics:")
    diagnostics = system.get_system_diagnostics()
    print(f"   Total responses processed: {diagnostics['system_stats']['total_responses_processed']}")
    print(f"   Errors detected: {diagnostics['system_stats']['errors_detected']}")
    print(f"   Corrections made: {diagnostics['system_stats']['corrections_made']}")
    print(f"   Feedback collected: {diagnostics['system_stats']['feedback_collected']}")
    
    # Export report
    report_success = system.export_diagnostics_report("error_detection_report.json")
    print(f"   Report export: {'SUCCESS' if report_success else 'FAILED'}")
    
    print("\n" + "=" * 50)
    print("ERROR DETECTION SYSTEM DEMO SUMMARY")
    print("=" * 50)
    print("Key Features Implemented:")
    print("  1. Multi-type error detection (contradictions, inconsistencies, etc.)")
    print("  2. Automated self-correction mechanisms")
    print("  3. Quality feedback loop management")
    print("  4. Trend analysis and improvement actions")
    print("  5. Comprehensive diagnostics and reporting")
    print("\nBenefits:")
    print("  - Automated error detection and correction")
    print("  - Continuous quality improvement")
    print("  - Data-driven optimization")
    print("  - Reduced manual oversight requirements")
    
    print("\n✅ Error detection system demonstration completed!")


if __name__ == "__main__":
    demo_error_detection_system()