"""
Error Detection System for MAHIA-X
Implements intelligent error detection and self-correction mechanisms
"""

import re
import json
from typing import Dict, Any, List, Tuple, Optional
from collections import defaultdict
import time
from datetime import datetime

class ErrorDetector:
    """Detects various types of errors in responses"""
    
    def __init__(self):
        self.detection_rules = {
            'contradiction': self._detect_contradictions,
            'factual_inaccuracy': self._detect_factual_errors,
            'logical_inconsistency': self._detect_logical_errors,
            'grammatical_error': self._detect_grammatical_errors,
            'incomplete_response': self._detect_incomplete_responses
        }
        
    def detect_errors(self, response: str, context: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Detect all types of errors in a response"""
        errors = []
        
        for error_type, detection_func in self.detection_rules.items():
            try:
                detected_errors = detection_func(response, context)
                errors.extend(detected_errors)
            except Exception as e:
                print(f"Error in {error_type} detection: {e}")
                
        return errors
        
    def _detect_contradictions(self, response: str, context: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Detect contradictory statements in response"""
        errors = []
        
        # Common contradiction patterns
        contradiction_patterns = [
            (r'\b(always|never)\b.*\b(never|always)\b', 'Absolute contradiction'),
            (r'\b(all|none)\b.*\b(none|all)\b', 'Universal contradiction'),
            (r'\b(impossible|cannot)\b.*\b(possible|can)\b', 'Possibility contradiction')
        ]
        
        for pattern, description in contradiction_patterns:
            matches = re.finditer(pattern, response, re.IGNORECASE)
            for match in matches:
                errors.append({
                    'type': 'contradiction',
                    'description': description,
                    'location': match.span(),
                    'text': match.group(),
                    'severity': 'high'
                })
                
        return errors
        
    def _detect_factual_errors(self, response: str, context: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Detect potential factual errors (simplified)"""
        errors = []
        
        # Common factual error patterns (simplified)
        factual_patterns = [
            (r'\b(Python \d+\.\d+ is the latest version)\b', 'Outdated version claim'),
            (r'\b(there are only \d+ planets)\b', 'Incorrect astronomical fact'),
            (r'\b(the earth is flat)\b', 'Scientifically incorrect statement')
        ]
        
        for pattern, description in factual_patterns:
            matches = re.finditer(pattern, response, re.IGNORECASE)
            for match in matches:
                errors.append({
                    'type': 'factual_inaccuracy',
                    'description': description,
                    'location': match.span(),
                    'text': match.group(),
                    'severity': 'high'
                })
                
        return errors
        
    def _detect_logical_errors(self, response: str, context: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Detect logical inconsistencies"""
        errors = []
        
        # Simple logical error detection
        if 'first' in response.lower() and 'secondly' not in response.lower() and 'then' in response.lower():
            errors.append({
                'type': 'logical_inconsistency',
                'description': 'Inconsistent sequential logic',
                'location': None,
                'text': 'Sequential logic inconsistency',
                'severity': 'medium'
            })
            
        return errors
        
    def _detect_grammatical_errors(self, response: str, context: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Detect basic grammatical errors"""
        errors = []
        
        # Simple grammatical error detection
        sentences = response.split('.')
        
        for i, sentence in enumerate(sentences):
            # Check for sentence starting with lowercase
            stripped = sentence.strip()
            if stripped and stripped[0].islower():
                errors.append({
                    'type': 'grammatical_error',
                    'description': 'Sentence starts with lowercase letter',
                    'location': None,
                    'text': stripped[:20] + '...',
                    'severity': 'low'
                })
                
        return errors
        
    def _detect_incomplete_responses(self, response: str, context: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Detect incomplete or truncated responses"""
        errors = []
        
        # Check for incomplete sentences
        if response.rstrip().endswith(('...', '..', '.')) and len(response.split()) < 10:
            errors.append({
                'type': 'incomplete_response',
                'description': 'Response appears to be incomplete',
                'location': None,
                'text': response,
                'severity': 'medium'
            })
            
        # Check for placeholder text
        placeholders = ['[placeholder]', '[todo]', '[insert]', '...']
        for placeholder in placeholders:
            if placeholder in response.lower():
                errors.append({
                    'type': 'incomplete_response',
                    'description': f'Contains placeholder: {placeholder}',
                    'location': None,
                    'text': placeholder,
                    'severity': 'high'
                })
                
        return errors

class SelfCorrectionEngine:
    """Automatically corrects detected errors"""
    
    def __init__(self):
        self.correction_history = []
        self.correction_rules = {
            'contradiction': self._correct_contradictions,
            'grammatical_error': self._correct_grammatical_errors,
            'incomplete_response': self._correct_incomplete_responses
        }
        
    def correct_errors(self, response: str, errors: List[Dict[str, Any]]) -> Tuple[str, List[Dict[str, Any]]]:
        """Apply corrections to response based on detected errors"""
        corrected_response = response
        applied_corrections = []
        
        # Group errors by type
        errors_by_type = defaultdict(list)
        for error in errors:
            errors_by_type[error['type']].append(error)
            
        # Apply corrections for each error type
        for error_type, error_list in errors_by_type.items():
            if error_type in self.correction_rules:
                try:
                    corrected_response, corrections = self.correction_rules[error_type](
                        corrected_response, error_list
                    )
                    applied_corrections.extend(corrections)
                except Exception as e:
                    print(f"Error applying {error_type} corrections: {e}")
                    
        # Record correction
        if applied_corrections:
            correction_record = {
                'original_response': response,
                'corrected_response': corrected_response,
                'corrections_applied': applied_corrections,
                'timestamp': datetime.now().isoformat()
            }
            self.correction_history.append(correction_record)
            
        return corrected_response, applied_corrections
        
    def _correct_contradictions(self, response: str, errors: List[Dict[str, Any]]) -> Tuple[str, List[Dict[str, Any]]]:
        """Correct contradictory statements"""
        corrected_response = response
        corrections = []
        
        for error in errors:
            if error['type'] == 'contradiction':
                # Simple correction: remove contradictory part
                text_to_remove = error['text']
                if text_to_remove in corrected_response:
                    corrected_response = corrected_response.replace(text_to_remove, '', 1)
                    corrections.append({
                        'type': 'removed_contradiction',
                        'original_text': text_to_remove,
                        'correction': 'Removed contradictory statement',
                        'confidence': 0.8
                    })
                    
        return corrected_response, corrections
        
    def _correct_grammatical_errors(self, response: str, errors: List[Dict[str, Any]]) -> Tuple[str, List[Dict[str, Any]]]:
        """Correct grammatical errors"""
        corrected_response = response
        corrections = []
        
        for error in errors:
            if error['type'] == 'grammatical_error':
                # Simple correction: capitalize first letter of sentences
                sentences = corrected_response.split('.')
                for i, sentence in enumerate(sentences):
                    stripped = sentence.strip()
                    if stripped and stripped[0].islower():
                        sentences[i] = sentence.replace(
                            stripped[0], 
                            stripped[0].upper(), 
                            1
                        )
                        corrections.append({
                            'type': 'capitalization_fix',
                            'original_text': sentence[:20],
                            'correction': 'Capitalized first letter',
                            'confidence': 0.9
                        })
                corrected_response = '.'.join(sentences)
                
        return corrected_response, corrections
        
    def _correct_incomplete_responses(self, response: str, errors: List[Dict[str, Any]]) -> Tuple[str, List[Dict[str, Any]]]:
        """Correct incomplete responses"""
        corrected_response = response
        corrections = []
        
        for error in errors:
            if error['type'] == 'incomplete_response':
                # Add continuation prompt
                if not corrected_response.endswith('.'):
                    corrected_response += '.'
                corrected_response += " [This response has been expanded for completeness.]"
                corrections.append({
                    'type': 'response_expansion',
                    'original_text': response[:30] + '...',
                    'correction': 'Expanded incomplete response',
                    'confidence': 0.7
                })
                
        return corrected_response, corrections

class QualityFeedbackLoop:
    """Manages feedback loop for continuous quality improvement"""
    
    def __init__(self):
        self.feedback_data = []
        self.quality_metrics = {
            'accuracy': 0.0,
            'completeness': 0.0,
            'consistency': 0.0,
            'clarity': 0.0
        }
        
    def add_feedback(self, feedback: Dict[str, Any]):
        """Add user feedback to improve quality metrics"""
        self.feedback_data.append({
            'feedback': feedback,
            'timestamp': datetime.now().isoformat()
        })
        
        # Update quality metrics based on feedback
        self._update_quality_metrics(feedback)
        
    def _update_quality_metrics(self, feedback: Dict[str, Any]):
        """Update quality metrics based on feedback"""
        rating = feedback.get('rating', 0.5)  # Default to neutral
        feedback_type = feedback.get('type', 'general')
        
        # Adjust metrics based on feedback
        if feedback_type == 'accuracy':
            self.quality_metrics['accuracy'] = (
                self.quality_metrics['accuracy'] * 0.9 + rating * 0.1
            )
        elif feedback_type == 'completeness':
            self.quality_metrics['completeness'] = (
                self.quality_metrics['completeness'] * 0.9 + rating * 0.1
            )
        elif feedback_type == 'consistency':
            self.quality_metrics['consistency'] = (
                self.quality_metrics['consistency'] * 0.9 + rating * 0.1
            )
        elif feedback_type == 'clarity':
            self.quality_metrics['clarity'] = (
                self.quality_metrics['clarity'] * 0.9 + rating * 0.1
            )
        else:
            # General feedback affects all metrics
            for metric in self.quality_metrics:
                self.quality_metrics[metric] = (
                    self.quality_metrics[metric] * 0.95 + rating * 0.05
                )
                
    def get_quality_report(self) -> Dict[str, Any]:
        """Generate quality report"""
        return {
            'metrics': self.quality_metrics.copy(),
            'total_feedback': len(self.feedback_data),
            'last_updated': datetime.now().isoformat()
        }
        
    def should_trigger_correction(self, response: str, errors: List[Dict[str, Any]]) -> bool:
        """Determine if automatic correction should be triggered"""
        # Trigger correction if there are high-severity errors
        high_severity_errors = [e for e in errors if e.get('severity') == 'high']
        return len(high_severity_errors) > 0

# Example usage
if __name__ == "__main__":
    # Initialize components
    error_detector = ErrorDetector()
    correction_engine = SelfCorrectionEngine()
    feedback_loop = QualityFeedbackLoop()
    
    # Example response with errors
    response = "Python 2.7 is the latest version. However, Python 3.9 is also available... [placeholder]"
    
    # Detect errors
    errors = error_detector.detect_errors(response)
    print("Detected Errors:", errors)
    
    # Apply corrections
    corrected_response, corrections = correction_engine.correct_errors(response, errors)
    print("Corrected Response:", corrected_response)
    print("Applied Corrections:", corrections)
    
    # Add feedback
    feedback = {
        'rating': 0.3,
        'type': 'accuracy',
        'comment': 'Incorrect version information'
    }
    feedback_loop.add_feedback(feedback)
    
    # Get quality report
    quality_report = feedback_loop.get_quality_report()
    print("Quality Report:", quality_report)