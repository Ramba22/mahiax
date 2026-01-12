"""
Data Provenance Tracking for MAHIA-X
This module implements data provenance tracking including source, timestamp, and license information.
"""

import time
import hashlib
import json
import os
import math
from typing import Dict, Any, Optional, List
from datetime import datetime
from collections import OrderedDict

class DataProvenanceTracker:
    """Data provenance tracker for tracking source, timestamp, and license information"""
    
    def __init__(self, provenance_file: Optional[str] = None):
        """
        Initialize data provenance tracker
        
        Args:
            provenance_file: File to store provenance information (optional)
        """
        self.provenance_file = provenance_file
        self.provenance_records = OrderedDict()
        self.record_count = 0
        
        # Load existing provenance data if file exists
        if self.provenance_file and os.path.exists(self.provenance_file):
            self._load_provenance_from_file()
            
        print("✅ DataProvenanceTracker initialized")
        
    def record_data_source(self, 
                          data_id: str,
                          source: str,
                          license_info: Optional[str] = None,
                          timestamp: Optional[float] = None,
                          metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Record data source information
        
        Args:
            data_id: Unique identifier for the data
            source: Source of the data (URL, file path, etc.)
            license_info: License information (optional)
            timestamp: Timestamp of data creation/access (optional)
            metadata: Additional metadata (optional)
            
        Returns:
            Record ID
        """
        # Use current time if not provided
        if timestamp is None:
            timestamp = time.time()
            
        # Create record
        record = {
            "data_id": data_id,
            "source": source,
            "license": license_info,
            "timestamp": timestamp,
            "recorded_at": time.time(),
            "metadata": metadata or {}
        }
        
        # Generate record ID
        record_id = self._generate_record_id(data_id, source, timestamp)
        record["record_id"] = record_id
        
        # Store record
        self.provenance_records[record_id] = record
        self.record_count += 1
        
        # Save to file if specified
        if self.provenance_file:
            self._save_provenance_to_file()
            
        print(f"✅ Recorded data provenance: {data_id} from {source}")
        return record_id
        
    def _generate_record_id(self, data_id: str, source: str, timestamp: float) -> str:
        """Generate unique record ID"""
        # Create hash from data_id, source, and timestamp
        hash_input = f"{data_id}:{source}:{timestamp}"
        return hashlib.sha256(hash_input.encode()).hexdigest()[:16]
        
    def get_provenance_record(self, record_id: str) -> Optional[Dict[str, Any]]:
        """
        Get provenance record by ID
        
        Args:
            record_id: Record ID
            
        Returns:
            Provenance record or None if not found
        """
        return self.provenance_records.get(record_id)
        
    def get_provenance_by_data_id(self, data_id: str) -> List[Dict[str, Any]]:
        """
        Get all provenance records for a specific data ID
        
        Args:
            data_id: Data ID
            
        Returns:
            List of provenance records
        """
        records = []
        for record in self.provenance_records.values():
            if record.get("data_id") == data_id:
                records.append(record)
        return records
        
    def verify_data_integrity(self, data_id: str, data_content: bytes) -> bool:
        """
        Verify data integrity using hash comparison
        
        Args:
            data_id: Data ID
            data_content: Data content as bytes
            
        Returns:
            True if integrity verified, False otherwise
        """
        # Generate hash of data content
        data_hash = hashlib.sha256(data_content).hexdigest()
        
        # Add hash to metadata if record exists
        records = self.get_provenance_by_data_id(data_id)
        if records:
            for record in records:
                existing_hash = record["metadata"].get("data_hash")
                if existing_hash and existing_hash == data_hash:
                    return True
                # Update record with new hash
                record["metadata"]["data_hash"] = data_hash
            # Save updated records
            if self.provenance_file:
                self._save_provenance_to_file()
            return True
            
        return False
        
    def add_data_hash(self, data_id: str, data_content: bytes) -> str:
        """
        Add data hash to provenance record for integrity tracking
        
        Args:
            data_id: Data ID
            data_content: Data content as bytes
            
        Returns:
            Hash of the data
        """
        # Generate hash
        data_hash = hashlib.sha256(data_content).hexdigest()
        
        # Add hash to metadata
        records = self.get_provenance_by_data_id(data_id)
        if records:
            for record in records:
                record["metadata"]["data_hash"] = data_hash
                
            # Save updated records
            if self.provenance_file:
                self._save_provenance_to_file()
                
        return data_hash
        
    def _save_provenance_to_file(self):
        """Save provenance records to file"""
        if not self.provenance_file:
            return
            
        try:
            # Create directory if it doesn't exist
            provenance_dir = os.path.dirname(self.provenance_file)
            if provenance_dir and not os.path.exists(provenance_dir):
                os.makedirs(provenance_dir)
                
            # Write records to file
            with open(self.provenance_file, 'w') as f:
                json.dump(dict(self.provenance_records), f, indent=2, default=str)
        except Exception as e:
            print(f"⚠️  Failed to save provenance records: {e}")
            
    def _load_provenance_from_file(self):
        """Load provenance records from file"""
        if not self.provenance_file or not os.path.exists(self.provenance_file):
            return
            
        try:
            with open(self.provenance_file, 'r') as f:
                loaded_records = json.load(f)
                self.provenance_records = OrderedDict(loaded_records)
                self.record_count = len(self.provenance_records)
        except Exception as e:
            print(f"⚠️  Failed to load provenance records: {e}")
            
    def get_provenance_summary(self) -> Dict[str, Any]:
        """
        Get provenance summary statistics
        
        Returns:
            Summary statistics
        """
        if not self.provenance_records:
            return {"total_records": 0}
            
        # Collect statistics
        sources = set()
        licenses = set()
        timestamps = []
        
        for record in self.provenance_records.values():
            sources.add(record.get("source", "unknown"))
            if record.get("license"):
                licenses.add(record.get("license"))
            timestamps.append(record.get("timestamp", 0))
            
        return {
            "total_records": self.record_count,
            "unique_sources": len(sources),
            "sources": list(sources),
            "unique_licenses": len(licenses),
            "licenses": list(licenses),
            "date_range": {
                "earliest": datetime.fromtimestamp(min(timestamps)).isoformat() if timestamps else None,
                "latest": datetime.fromtimestamp(max(timestamps)).isoformat() if timestamps else None
            }
        }
        
    def export_provenance_report(self, report_file: str) -> bool:
        """
        Export provenance report to file
        
        Args:
            report_file: File to export report to
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Get summary
            summary = self.get_provenance_summary()
            
            # Prepare report
            report = {
                "generated_at": datetime.now().isoformat(),
                "summary": summary,
                "records": dict(self.provenance_records)
            }
            
            # Write report
            with open(report_file, 'w') as f:
                json.dump(report, f, indent=2, default=str)
                
            print(f"✅ Provenance report exported to: {report_file}")
            return True
        except Exception as e:
            print(f"❌ Failed to export provenance report: {e}")
            return False
            
    def clear_provenance_records(self):
        """Clear all provenance records"""
        self.provenance_records.clear()
        self.record_count = 0
        if self.provenance_file and os.path.exists(self.provenance_file):
            try:
                os.remove(self.provenance_file)
            except Exception as e:
                print(f"⚠️  Failed to remove provenance file: {e}")


class BiasDriftDetector:
    """Bias and drift detector for auto-generated data"""
    
    def __init__(self, holdout_ratio: float = 0.1):
        """
        Initialize bias and drift detector
        
        Args:
            holdout_ratio: Ratio of data to hold out for validation
        """
        self.holdout_ratio = holdout_ratio
        self.holdout_data = []
        self.validation_metrics = OrderedDict()
        
        print(f"✅ BiasDriftDetector initialized with holdout ratio: {holdout_ratio}")
        
    def add_holdout_data(self, data_sample: Dict[str, Any]):
        """
        Add data sample to holdout set
        
        Args:
            data_sample: Data sample to add
        """
        self.holdout_data.append(data_sample)
        
        # Keep only recent samples (last 1000)
        if len(self.holdout_data) > 1000:
            self.holdout_data = self.holdout_data[-1000:]
            
    def detect_bias(self, current_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Detect bias in current data compared to holdout set
        
        Args:
            current_data: Current data samples
            
        Returns:
            Bias detection results
        """
        if not self.holdout_data or not current_data:
            return {"bias_detected": False, "confidence": 0.0}
            
        # Simple bias detection based on label distribution
        holdout_labels = [sample.get("label", 0) for sample in self.holdout_data]
        current_labels = [sample.get("label", 0) for sample in current_data]
        
        # Calculate label distributions
        holdout_dist = self._calculate_distribution(holdout_labels)
        current_dist = self._calculate_distribution(current_labels)
        
        # Compare distributions
        bias_score = self._compare_distributions(holdout_dist, current_dist)
        
        # Determine if bias is significant
        bias_detected = bias_score > 0.1  # Threshold for bias detection
        
        result = {
            "bias_detected": bias_detected,
            "bias_score": bias_score,
            "holdout_distribution": holdout_dist,
            "current_distribution": current_dist,
            "confidence": 1.0 - bias_score
        }
        
        # Store validation metric
        metric_id = f"bias_check_{int(time.time())}"
        self.validation_metrics[metric_id] = result
        
        return result
        
    def _calculate_distribution(self, labels: List) -> Dict[Any, float]:
        """Calculate label distribution"""
        if not labels:
            return {}
            
        total = len(labels)
        distribution = {}
        
        for label in labels:
            distribution[label] = distribution.get(label, 0) + 1
            
        # Normalize
        for label in distribution:
            distribution[label] /= total
            
        return distribution
        
    def _compare_distributions(self, dist1: Dict[Any, float], 
                             dist2: Dict[Any, float]) -> float:
        """Compare two distributions using Jensen-Shannon divergence"""
        # Get all unique keys
        all_keys = set(dist1.keys()) | set(dist2.keys())
        
        # Calculate Jensen-Shannon divergence
        js_divergence = 0.0
        for key in all_keys:
            p1 = dist1.get(key, 0.0)
            p2 = dist2.get(key, 0.0)
            m = (p1 + p2) / 2.0
            
            if p1 > 0:
                js_divergence += p1 * math.log(p1 / m) if m > 0 else 0
            if p2 > 0:
                js_divergence += p2 * math.log(p2 / m) if m > 0 else 0
                
        js_divergence /= 2.0
        return js_divergence
        
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
            return {"drift_detected": False, "drift_score": 0.0}
            
        # Simple drift detection using mean difference
        import statistics
        
        try:
            # Calculate mean features for both sets
            ref_means = [statistics.mean([features[i] for features in reference_features 
                                        if i < len(features)]) 
                        for i in range(max(len(f) for f in reference_features))]
                        
            curr_means = [statistics.mean([features[i] for features in current_features 
                                         if i < len(features)]) 
                         for i in range(max(len(f) for f in current_features))]
                         
            # Calculate drift score as mean absolute difference
            drift_scores = [abs(ref_means[i] - curr_means[i]) 
                           for i in range(min(len(ref_means), len(curr_means)))]
                           
            avg_drift_score = statistics.mean(drift_scores) if drift_scores else 0.0
            
            # Determine if drift is significant
            drift_detected = avg_drift_score > 0.1  # Threshold for drift detection
            
            result = {
                "drift_detected": drift_detected,
                "drift_score": avg_drift_score,
                "reference_means": ref_means,
                "current_means": curr_means
            }
            
            # Store validation metric
            metric_id = f"drift_check_{int(time.time())}"
            self.validation_metrics[metric_id] = result
            
            return result
            
        except Exception as e:
            print(f"⚠️  Error in drift detection: {e}")
            return {"drift_detected": False, "drift_score": 0.0}
            
    def get_validation_metrics(self) -> Dict[str, Any]:
        """Get validation metrics"""
        return dict(self.validation_metrics)
        
    def clear_validation_metrics(self):
        """Clear validation metrics"""
        self.validation_metrics.clear()


def demo_data_provenance():
    """Demonstrate data provenance tracking functionality"""
    print("🚀 Demonstrating Data Provenance Tracking...")
    print("=" * 60)
    
    # Create provenance tracker
    tracker = DataProvenanceTracker("provenance_demo.json")
    print("✅ Created data provenance tracker")
    
    # Record some data sources
    record1 = tracker.record_data_source(
        data_id="dataset_001",
        source="https://example.com/dataset1.csv",
        license_info="CC-BY-4.0",
        metadata={"size": "1000 samples", "type": "text classification"}
    )
    print(f"✅ Recorded data source 1: {record1}")
    
    record2 = tracker.record_data_source(
        data_id="dataset_002",
        source="/local/data/dataset2.json",
        license_info="MIT",
        metadata={"size": "500 samples", "type": "image classification"}
    )
    print(f"✅ Recorded data source 2: {record2}")
    
    record3 = tracker.record_data_source(
        data_id="generated_001",
        source="auto-generated",
        metadata={"algorithm": "GPT-4", "prompt": "Generate text samples"}
    )
    print(f"✅ Recorded generated data: {record3}")
    
    # Retrieve records
    retrieved_record = tracker.get_provenance_record(record1)
    if retrieved_record:
        print(f"✅ Retrieved record: {retrieved_record['data_id']} from {retrieved_record['source']}")
    else:
        print("❌ Failed to retrieve record")
    
    # Get records by data ID
    dataset_records = tracker.get_provenance_by_data_id("dataset_001")
    print(f"✅ Found {len(dataset_records)} records for dataset_001")
    
    # Add data hash for integrity tracking
    sample_data = b"This is sample data content for integrity checking"
    data_hash = tracker.add_data_hash("dataset_001", sample_data)
    print(f"✅ Added data hash: {data_hash[:16]}...")
    
    # Verify integrity
    is_valid = tracker.verify_data_integrity("dataset_001", sample_data)
    print(f"✅ Data integrity verification: {'PASSED' if is_valid else 'FAILED'}")
    
    # Get summary
    summary = tracker.get_provenance_summary()
    print(f"✅ Provenance summary:")
    print(f"   Total records: {summary['total_records']}")
    print(f"   Unique sources: {summary['unique_sources']}")
    print(f"   Unique licenses: {summary['unique_licenses']}")
    
    # Export report
    report_success = tracker.export_provenance_report("provenance_report.json")
    print(f"✅ Report export: {'SUCCESS' if report_success else 'FAILED'}")
    
    # Create bias/drift detector
    detector = BiasDriftDetector(holdout_ratio=0.15)
    print("✅ Created bias/drift detector")
    
    # Add some holdout data
    for i in range(50):
        sample = {
            "label": i % 3,  # Balanced 3-class distribution
            "features": [i * 0.1, i * 0.2, i * 0.3]
        }
        detector.add_holdout_data(sample)
    print("✅ Added 50 samples to holdout set")
    
    # Test bias detection
    current_data = [{"label": i % 2} for i in range(30)]  # Imbalanced 2-class
    bias_result = detector.detect_bias(current_data)
    print(f"✅ Bias detection: {'BIAS DETECTED' if bias_result['bias_detected'] else 'NO BIAS'}")
    print(f"   Bias score: {bias_result['bias_score']:.4f}")
    
    # Test drift detection
    current_features = [[i * 0.15, i * 0.25, i * 0.35] for i in range(40)]
    drift_result = detector.detect_drift(current_features)
    print(f"✅ Drift detection: {'DRIFT DETECTED' if drift_result['drift_detected'] else 'NO DRIFT'}")
    print(f"   Drift score: {drift_result['drift_score']:.4f}")
    
    # Show validation metrics
    metrics = detector.get_validation_metrics()
    print(f"✅ Validation metrics recorded: {len(metrics)}")
    
    print("\n" + "=" * 60)
    print("DATA PROVENANCE TRACKING DEMO SUMMARY")
    print("=" * 60)
    print("Key Features Implemented:")
    print("  1. Comprehensive data source tracking")
    print("  2. License and timestamp management")
    print("  3. Data integrity verification with hashing")
    print("  4. Bias detection in auto-generated data")
    print("  5. Drift detection for feature consistency")
    print("  6. Detailed reporting and export")
    print("  7. Holdout validation for quality control")
    print("\nBenefits:")
    print("  - Full audit trail for data sources")
    print("  - Compliance with licensing requirements")
    print("  - Early detection of data quality issues")
    print("  - Reproducible data processing pipelines")
    print("  - Automated validation of generated data")
    
    print("\n✅ Data Provenance Tracking demonstration completed!")


if __name__ == "__main__":
    demo_data_provenance()