"""
Ethics Engine for MAHIA-X
Implements bias reduction, privacy protection, and ethical guidelines
"""

import re
import json
import hashlib
from typing import Dict, Any, List, Optional, Set
from collections import defaultdict
from datetime import datetime
import time

class PrivacyProtection:
    """Protects user privacy through data anonymization"""
    
    def __init__(self):
        self.sensitive_patterns = [
            r'\b\d{3}-\d{2}-\d{4}\b',  # SSN pattern
            r'\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b',  # Credit card
            r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',  # Email
            r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b',  # Phone number
            r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b'  # IP address
        ]
        
    def anonymize_text(self, text: str) -> str:
        """Anonymize sensitive information in text"""
        anonymized_text = text
        
        # Replace sensitive patterns with placeholders
        for pattern in self.sensitive_patterns:
            anonymized_text = re.sub(
                pattern, 
                '[REDACTED]', 
                anonymized_text
            )
            
        return anonymized_text
        
    def hash_identifier(self, identifier: str) -> str:
        """Hash identifiers for privacy protection"""
        return hashlib.sha256(identifier.encode()).hexdigest()

class BiasDetector:
    """Detects and reduces bias in responses"""
    
    def __init__(self):
        # Simplified bias detection patterns
        self.bias_patterns = {
            'gender': [
                r'\b(he is|she is) (better|worse|smarter|dumber)\b',
                r'\b(boys|girls) (are|should be)\b'
            ],
            'racial': [
                r'\b(black|white|asian) people (are|should)\b',
                r'\b(certain races) (are|should be)\b'
            ],
            'socioeconomic': [
                r'\b(rich|poor) people (are|should)\b',
                r'\b(wealthy|low-income) (are|should be)\b'
            ]
        }
        
        self.bias_dictionary = {
            'gender': [
                'he', 'she', 'him', 'her', 'his', 'hers',
                'man', 'woman', 'men', 'women', 'boy', 'girl'
            ],
            'racial': [
                'black', 'white', 'asian', 'hispanic', 'african', 'european'
            ]
        }
        
    def detect_bias(self, text: str) -> List[Dict[str, Any]]:
        """Detect potential bias in text"""
        biases = []
        
        # Check for explicit bias patterns
        for bias_type, patterns in self.bias_patterns.items():
            for pattern in patterns:
                matches = re.finditer(pattern, text, re.IGNORECASE)
                for match in matches:
                    biases.append({
                        'type': bias_type,
                        'pattern': pattern,
                        'text': match.group(),
                        'position': match.span(),
                        'confidence': 0.9
                    })
                    
        # Check for potential bias through word frequency
        words = text.lower().split()
        word_counts = defaultdict(int)
        for word in words:
            word_counts[word] += 1
            
        # Check bias categories
        for category, bias_words in self.bias_dictionary.items():
            category_count = sum(word_counts[word] for word in bias_words)
            total_words = len(words)
            
            if total_words > 0 and category_count / total_words > 0.1:  # 10% threshold
                biases.append({
                    'type': category,
                    'description': f'High frequency of {category}-related terms',
                    'frequency': category_count / total_words,
                    'confidence': 0.7
                })
                
        return biases
        
    def reduce_bias(self, text: str) -> str:
        """Reduce detected bias in text"""
        # Simple bias reduction: replace biased terms with neutral ones
        bias_replacements = {
            'he': 'they',
            'she': 'they',
            'him': 'them',
            'her': 'them',
            'his': 'their',
            'hers': 'theirs',
            'man': 'person',
            'woman': 'person',
            'men': 'people',
            'women': 'people'
        }
        
        reduced_text = text
        for biased_term, neutral_term in bias_replacements.items():
            # Case-insensitive replacement
            pattern = re.compile(re.escape(biased_term), re.IGNORECASE)
            reduced_text = pattern.sub(neutral_term, reduced_text)
            
        return reduced_text

class EthicsGuidelines:
    """Enforces ethical guidelines in responses"""
    
    def __init__(self):
        self.guidelines = {
            'harm_prevention': [
                'do not provide instructions for illegal activities',
                'do not encourage harmful behavior',
                'avoid promoting violence'
            ],
            'privacy_respect': [
                'do not share personal information',
                'respect user confidentiality',
                'protect sensitive data'
            ],
            'fairness': [
                'treat all users equally',
                'avoid discrimination',
                'provide balanced perspectives'
            ]
        }
        
        self.prohibited_patterns = [
            r'\b(how to hack|hack instructions)\b',
            r'\b(illegal drugs|controlled substances)\b',
            r'\b(violent acts|harm to others)\b'
        ]
        
    def check_ethical_compliance(self, text: str) -> Dict[str, Any]:
        """Check if text complies with ethical guidelines"""
        compliance_report = {
            'compliant': True,
            'violations': [],
            'guidelines_checked': list(self.guidelines.keys())
        }
        
        # Check prohibited patterns
        for pattern in self.prohibited_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                compliance_report['compliant'] = False
                compliance_report['violations'].append({
                    'type': 'prohibited_content',
                    'pattern': pattern,
                    'description': 'Content violates ethical guidelines'
                })
                
        return compliance_report
        
    def apply_ethical_filter(self, text: str) -> str:
        """Apply ethical filtering to text"""
        # Remove prohibited content
        filtered_text = text
        for pattern in self.prohibited_patterns:
            filtered_text = re.sub(pattern, '[CONTENT REMOVED FOR ETHICAL REASONS]', 
                                 filtered_text, flags=re.IGNORECASE)
                                 
        return filtered_text

class EthicsEngine:
    """Main ethics engine coordinating privacy, bias, and ethical guidelines"""
    
    def __init__(self):
        self.privacy_protection = PrivacyProtection()
        self.bias_detector = BiasDetector()
        self.ethics_guidelines = EthicsGuidelines()
        self.audit_log = []
        
    def process_response(self, response: str, user_id: Optional[str] = None) -> Dict[str, Any]:
        """Process response through all ethical checks"""
        processing_result: Dict[str, Any] = {
            'original_response': response,
            'timestamp': datetime.now().isoformat()
        }
        
        # Apply privacy protection
        anonymized_response = self.privacy_protection.anonymize_text(response)
        processing_result['anonymized_response'] = anonymized_response
        
        # Detect bias
        biases = self.bias_detector.detect_bias(anonymized_response)
        processing_result['detected_biases'] = biases
        
        # Reduce bias if found
        if biases:
            bias_reduced_response = self.bias_detector.reduce_bias(anonymized_response)
            processing_result['bias_reduced_response'] = bias_reduced_response
        else:
            bias_reduced_response = anonymized_response
            
        # Check ethical compliance
        compliance = self.ethics_guidelines.check_ethical_compliance(bias_reduced_response)
        processing_result['ethical_compliance'] = compliance
        
        # Apply ethical filtering if needed
        if not compliance['compliant']:
            final_response = self.ethics_guidelines.apply_ethical_filter(bias_reduced_response)
            processing_result['final_response'] = final_response
            processing_result['filtered'] = True
        else:
            final_response = bias_reduced_response
            processing_result['final_response'] = final_response
            processing_result['filtered'] = False
            
        # Log processing
        self._log_processing(processing_result, user_id)
        
        return processing_result
        
    def _log_processing(self, result: Dict[str, Any], user_id: Optional[str] = None):
        """Log processing results for audit"""
        log_entry = {
            'timestamp': result['timestamp'],
            'user_id': self.privacy_protection.hash_identifier(user_id) if user_id else None,
            'biases_detected': len(result.get('detected_biases', [])),
            'filtered': result.get('filtered', False),
            'compliant': result.get('ethical_compliance', {}).get('compliant', True)
        }
        
        self.audit_log.append(log_entry)
        
        # Keep audit log manageable
        if len(self.audit_log) > 1000:
            self.audit_log = self.audit_log[-500:]
            
    def get_audit_report(self) -> Dict[str, Any]:
        """Generate audit report"""
        if not self.audit_log:
            return {'report': 'No audit data available'}
            
        # Calculate statistics
        total_processed = len(self.audit_log)
        bias_detected = sum(1 for entry in self.audit_log if entry['biases_detected'] > 0)
        filtered = sum(1 for entry in self.audit_log if entry['filtered'])
        non_compliant = sum(1 for entry in self.audit_log if not entry['compliant'])
        
        return {
            'total_responses_processed': total_processed,
            'bias_detected_rate': bias_detected / total_processed if total_processed > 0 else 0,
            'filtering_rate': filtered / total_processed if total_processed > 0 else 0,
            'non_compliance_rate': non_compliant / total_processed if total_processed > 0 else 0,
            'audit_period': {
                'start': self.audit_log[0]['timestamp'] if self.audit_log else None,
                'end': self.audit_log[-1]['timestamp'] if self.audit_log else None
            }
        }

# Example usage
if __name__ == "__main__":
    # Initialize ethics engine
    ethics_engine = EthicsEngine()
    
    # Example responses to process
    test_responses = [
        "John Smith with SSN 123-45-6789 should contact mary@example.com",
        "Men are naturally better at programming than women",
        "How to hack into a computer system?",
        "The weather is nice today"
    ]
    
    # Process each response
    for i, response in enumerate(test_responses):
        print(f"\n--- Processing Response {i+1} ---")
        print(f"Original: {response}")
        
        result = ethics_engine.process_response(response, f"user_{i}")
        print(f"Final Response: {result['final_response']}")
        print(f"Biases Detected: {len(result['detected_biases'])}")
        print(f"Filtered: {result['filtered']}")
        print(f"Compliant: {result['ethical_compliance']['compliant']}")
        
    # Get audit report
    audit_report = ethics_engine.get_audit_report()
    print("\n--- Audit Report ---")
    print(json.dumps(audit_report, indent=2))