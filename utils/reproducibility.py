"""
Reproducibility Utilities for MAHIA-X
Implements seeds, model-weight hashes, and code snapshots for full reproducibility
"""

import json
import time
import os
import hashlib
import random
import subprocess
import sys
from typing import Dict, Any, Optional, List, Union
from datetime import datetime
import logging

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

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ReproducibilityManager:
    """Manager for ensuring reproducibility of experiments"""
    
    def __init__(self, experiment_dir: str = "experiments", experiment_name: str = "default"):
        """
        Initialize reproducibility manager
        
        Args:
            experiment_dir: Directory to store experiment metadata
            experiment_name: Name of the experiment
        """
        self.experiment_dir = experiment_dir
        self.experiment_name = experiment_name
        self.experiment_path = os.path.join(experiment_dir, experiment_name)
        
        # Create experiment directory
        os.makedirs(self.experiment_path, exist_ok=True)
        
        # Reproducibility metadata
        self.seed = None
        self.hashes = {}
        self.environment = {}
        self.code_snapshot = {}
        
        print(f"✅ ReproducibilityManager initialized for experiment: {experiment_name}")
        print(f"   Experiment path: {self.experiment_path}")
        
    def set_seeds(self, seed: int = 42) -> Dict[str, int]:
        """
        Set random seeds for reproducibility
        
        Args:
            seed: Random seed to use
            
        Returns:
            Dictionary of seeds set
        """
        self.seed = seed
        
        # Set Python random seed
        random.seed(seed)
        
        # Set NumPy seed
        if NUMPY_AVAILABLE:
            np.random.seed(seed)
            
        # Set PyTorch seeds
        if TORCH_AVAILABLE:
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            
        seeds_set = {
            "python": seed,
            "numpy": seed if NUMPY_AVAILABLE else None,
            "torch": seed if TORCH_AVAILABLE else None,
            "cuda": seed if TORCH_AVAILABLE else None
        }
        
        # Save seeds to file
        seeds_file = os.path.join(self.experiment_path, "seeds.json")
        with open(seeds_file, 'w') as f:
            json.dump(seeds_set, f, indent=2)
            
        print(f"✅ Random seeds set to {seed}")
        return seeds_set
        
    def calculate_model_hash(self, model_path: str) -> str:
        """
        Calculate SHA256 hash of model weights file
        
        Args:
            model_path: Path to model weights file
            
        Returns:
            SHA256 hash as hex string
        """
        if not os.path.exists(model_path):
            logger.warning(f"Model file not found: {model_path}")
            return ""
            
        try:
            hash_sha256 = hashlib.sha256()
            with open(model_path, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_sha256.update(chunk)
            model_hash = hash_sha256.hexdigest()
            
            # Store hash
            self.hashes[model_path] = model_hash
            
            print(f"✅ Model hash calculated for {model_path}: {model_hash[:16]}...")
            return model_hash
        except Exception as e:
            logger.error(f"Failed to calculate model hash: {e}")
            return ""
            
    def create_code_snapshot(self, source_dirs: Optional[List[str]] = None) -> Dict[str, str]:
        """
        Create snapshot of code files with hashes
        
        Args:
            source_dirs: List of directories to snapshot (default: ['.'])
            
        Returns:
            Dictionary mapping file paths to their hashes
        """
        if source_dirs is None:
            source_dirs = ['.']
            
        # Filter out non-existent directories
        source_dirs = [d for d in source_dirs if os.path.exists(d)]
        
        if not source_dirs:
            logger.warning("No valid source directories found for code snapshot")
            return {}
            
        print(f"📸 Creating code snapshot of directories: {source_dirs}")
        
        # Walk through directories and hash Python files
        code_hashes = {}
        
        for source_dir in source_dirs:
            for root, dirs, files in os.walk(source_dir):
                # Skip hidden directories and common build directories
                dirs[:] = [d for d in dirs if not d.startswith('.') and d not in ['__pycache__', 'build', 'dist']]
                
                for file in files:
                    if file.endswith(('.py', '.yaml', '.yml', '.json', '.md')):
                        file_path = os.path.join(root, file)
                        
                        try:
                            # Calculate file hash
                            file_hash = self._calculate_file_hash(file_path)
                            code_hashes[file_path] = file_hash
                        except Exception as e:
                            logger.warning(f"Failed to hash {file_path}: {e}")
                            
        # Save code snapshot
        self.code_snapshot = code_hashes
        snapshot_file = os.path.join(self.experiment_path, "code_snapshot.json")
        with open(snapshot_file, 'w') as f:
            json.dump({
                "timestamp": datetime.now().isoformat(),
                "files": code_hashes
            }, f, indent=2)
            
        print(f"✅ Code snapshot created with {len(code_hashes)} files")
        return code_hashes
        
    def _calculate_file_hash(self, file_path: str) -> str:
        """
        Calculate SHA256 hash of a file
        
        Args:
            file_path: Path to file
            
        Returns:
            SHA256 hash as hex string
        """
        hash_sha256 = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_sha256.update(chunk)
        return hash_sha256.hexdigest()
        
    def capture_environment(self) -> Dict[str, Any]:
        """
        Capture environment information for reproducibility
        
        Returns:
            Dictionary of environment information
        """
        environment_info = {
            "timestamp": datetime.now().isoformat(),
            "python_version": sys.version,
            "platform": sys.platform,
            "cwd": os.getcwd(),
            "environment_vars": dict(os.environ),
            "installed_packages": self._get_installed_packages()
        }
        
        # Save environment info
        self.environment = environment_info
        env_file = os.path.join(self.experiment_path, "environment.json")
        with open(env_file, 'w') as f:
            json.dump(environment_info, f, indent=2, default=str)
            
        print("✅ Environment information captured")
        return environment_info
        
    def _get_installed_packages(self) -> List[str]:
        """
        Get list of installed packages
        
        Returns:
            List of installed packages
        """
        try:
            result = subprocess.run([sys.executable, '-m', 'pip', 'list'], 
                                  capture_output=True, text=True, timeout=30)
            if result.returncode == 0:
                return result.stdout.split('\n')[2:]  # Skip header lines
        except Exception as e:
            logger.warning(f"Failed to get installed packages: {e}")
        return []
        
    def create_git_snapshot(self) -> Dict[str, str]:
        """
        Create snapshot of git repository state
        
        Returns:
            Dictionary of git information
        """
        git_info = {}
        
        try:
            # Get current commit hash
            result = subprocess.run(['git', 'rev-parse', 'HEAD'], 
                                  capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                git_info["commit_hash"] = result.stdout.strip()
                
            # Get current branch
            result = subprocess.run(['git', 'rev-parse', '--abbrev-ref', 'HEAD'], 
                                  capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                git_info["branch"] = result.stdout.strip()
                
            # Get git status
            result = subprocess.run(['git', 'status', '--porcelain'], 
                                  capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                git_info["modified_files"] = [line.strip() for line in result.stdout.split('\n') if line.strip()]
                
            # Get git diff for uncommitted changes
            result = subprocess.run(['git', 'diff'], 
                                  capture_output=True, text=True, timeout=30)
            if result.returncode == 0 and result.stdout:
                git_info["uncommitted_diff"] = result.stdout
                
        except Exception as e:
            logger.warning(f"Failed to capture git information: {e}")
            
        # Save git info
        if git_info:
            git_file = os.path.join(self.experiment_path, "git_snapshot.json")
            with open(git_file, 'w') as f:
                json.dump({
                    "timestamp": datetime.now().isoformat(),
                    "git_info": git_info
                }, f, indent=2)
                
            print("✅ Git repository snapshot created")
            
        return git_info
        
    def generate_reproducibility_report(self) -> str:
        """
        Generate comprehensive reproducibility report
        
        Returns:
            Formatted reproducibility report
        """
        report = f"""
==================== REPRODUCIBILITY REPORT ====================
Experiment: {self.experiment_name}
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
================================================================

1. SEEDS
--------
"""
        
        if self.seed is not None:
            report += f"Random Seed: {self.seed}\n"
        else:
            report += "No seeds set\n"
            
        report += "\n2. MODEL HASHES\n---------------\n"
        if self.hashes:
            for model_path, model_hash in self.hashes.items():
                report += f"{model_path}: {model_hash[:16]}...\n"
        else:
            report += "No model hashes calculated\n"
            
        report += "\n3. CODE SNAPSHOT\n----------------\n"
        if self.code_snapshot:
            report += f"Files snapshotted: {len(self.code_snapshot)}\n"
            # Show first 5 files as example
            for i, (file_path, file_hash) in enumerate(list(self.code_snapshot.items())[:5]):
                report += f"  {file_path}: {file_hash[:16]}...\n"
            if len(self.code_snapshot) > 5:
                report += f"  ... and {len(self.code_snapshot) - 5} more files\n"
        else:
            report += "No code snapshot created\n"
            
        report += "\n4. ENVIRONMENT\n--------------\n"
        if self.environment:
            report += f"Python Version: {self.environment.get('python_version', 'Unknown')[:50]}...\n"
            report += f"Platform: {self.environment.get('platform', 'Unknown')}\n"
        else:
            report += "No environment information captured\n"
            
        report += "\n5. GIT SNAPSHOT\n---------------\n"
        git_snapshot_path = os.path.join(self.experiment_path, "git_snapshot.json")
        if os.path.exists(git_snapshot_path):
            try:
                with open(git_snapshot_path, 'r') as f:
                    git_info = json.load(f)
                git_data = git_info.get("git_info", {})
                if "commit_hash" in git_data:
                    report += f"Commit: {git_data['commit_hash'][:16]}...\n"
                if "branch" in git_data:
                    report += f"Branch: {git_data['branch']}\n"
                if "modified_files" in git_data and git_data["modified_files"]:
                    report += f"Modified files: {len(git_data['modified_files'])}\n"
            except Exception as e:
                report += f"Failed to read git snapshot: {e}\n"
        else:
            report += "No git snapshot created\n"
            
        report += "\n================================================================\n"
        report += "To reproduce this experiment, ensure the same seeds, model weights,\n"
        report += "code versions, and environment are used.\n"
        report += "================================================================\n"
        
        # Save report to file
        report_file = os.path.join(self.experiment_path, "reproducibility_report.txt")
        with open(report_file, 'w') as f:
            f.write(report)
            
        return report
        
    def save_experiment_metadata(self):
        """
        Save all experiment metadata to a single file
        """
        metadata = {
            "experiment_name": self.experiment_name,
            "timestamp": datetime.now().isoformat(),
            "seed": self.seed,
            "hashes": self.hashes,
            "environment": self.environment,
            "code_snapshot": self.code_snapshot
        }
        
        metadata_file = os.path.join(self.experiment_path, "experiment_metadata.json")
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
            
        print(f"✅ Experiment metadata saved to {metadata_file}")


def demo_reproducibility():
    """Demonstrate reproducibility functionality"""
    print("🚀 Demonstrating Reproducibility Manager...")
    print("=" * 50)
    
    # Create reproducibility manager
    manager = ReproducibilityManager(
        experiment_dir="demo_experiments",
        experiment_name="reproducibility_demo"
    )
    print("✅ Created reproducibility manager")
    
    # Set seeds
    print("\n🎲 Setting random seeds...")
    seeds = manager.set_seeds(42)
    print(f"   Seeds set: {seeds}")
    
    # Create dummy model file for demo
    dummy_model_path = os.path.join("demo_experiments", "dummy_model.pth")
    with open(dummy_model_path, 'w') as f:
        f.write("Dummy model weights for demonstration purposes")
        
    # Calculate model hash
    print("\n🔍 Calculating model hash...")
    model_hash = manager.calculate_model_hash(dummy_model_path)
    print(f"   Model hash: {model_hash[:16]}...")
    
    # Create code snapshot
    print("\n📸 Creating code snapshot...")
    code_hashes = manager.create_code_snapshot(['.'])
    print(f"   Files hashed: {len(code_hashes)}")
    
    # Capture environment
    print("\n🖥️  Capturing environment...")
    env_info = manager.capture_environment()
    print(f"   Python version: {env_info['python_version'][:30]}...")
    
    # Create git snapshot
    print("\n🌿 Creating git snapshot...")
    git_info = manager.create_git_snapshot()
    if git_info:
        print(f"   Git info captured: {len(git_info)} fields")
    else:
        print("   No git repository found")
    
    # Generate reproducibility report
    print("\n📋 Generating reproducibility report...")
    report = manager.generate_reproducibility_report()
    print("✅ Reproducibility report generated")
    
    # Save metadata
    manager.save_experiment_metadata()
    
    # Show summary
    print("\n" + "=" * 60)
    print("REPRODUCIBILITY MANAGER DEMO SUMMARY")
    print("=" * 60)
    print("Key Features Implemented:")
    print("  1. Random seed management for Python, NumPy, PyTorch")
    print("  2. Model weight hashing for verification")
    print("  3. Code snapshot with file hashing")
    print("  4. Environment capture with package listing")
    print("  5. Git repository state snapshot")
    print("  6. Comprehensive reproducibility reporting")
    print("\nBenefits:")
    print("  - Full experiment reproducibility")
    print("  - Model weight verification")
    print("  - Code version tracking")
    print("  - Environment consistency")
    print("  - Research compliance and auditability")
    
    print("\n✅ Reproducibility manager demonstration completed!")


if __name__ == "__main__":
    demo_reproducibility()