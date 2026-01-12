"""
Hugging Face Dataset Integration for MAHIA
Integration with real GLUE / MMLU / BIG-Bench / LongBench / MMMU datasets
"""

# Standard library imports
import os
import json
from typing import Optional, Dict, Any, List, Tuple, Union

# Conditional imports for PyTorch
TORCH_AVAILABLE = False
try:
    import torch
    from torch.utils.data import Dataset, DataLoader
    TORCH_AVAILABLE = True
except ImportError:
    print("⚠️  PyTorch not available")

# Conditional imports for Hugging Face libraries
HUGGINGFACE_AVAILABLE = False
try:
    from transformers import AutoTokenizer, AutoModel
    from datasets import Dataset as HFDataset
    from datasets import load_dataset, DatasetDict
    HUGGINGFACE_AVAILABLE = True
    print("✅ Hugging Face libraries available")
except ImportError:
    print("⚠️  Hugging Face libraries not available")

# Conditional import for pandas
PANDAS_AVAILABLE = False
try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    print("⚠️  Pandas not available")

class GLUEDatasetLoader:
    """
    Loader for GLUE benchmark datasets
    """
    
    def __init__(self, 
                 cache_dir: str = "./hf_cache",
                 max_length: int = 512):
        """
        Initialize GLUE dataset loader
        
        Args:
            cache_dir: Directory to cache datasets
            max_length: Maximum sequence length
        """
        self.cache_dir = cache_dir
        self.max_length = max_length
        
        # Create cache directory
        os.makedirs(cache_dir, exist_ok=True)
        
        # GLUE tasks
        self.glue_tasks = [
            "cola", "sst2", "mrpc", "qqp", "stsb", 
            "mnli", "qnli", "rte", "wnli"
        ]
        
        print(f"✅ GLUEDatasetLoader initialized")
        
    def load_task(self, 
                  task_name: str,
                  split: str = "train",
                  tokenizer_name: str = "bert-base-uncased") -> Tuple[HFDataset, Any]:
        """
        Load specific GLUE task
        
        Args:
            task_name: Name of GLUE task
            split: Data split ("train", "validation", "test")
            tokenizer_name: Name of tokenizer to use
            
        Returns:
            Tuple of (dataset, tokenizer)
        """
        if not HUGGINGFACE_AVAILABLE:
            raise RuntimeError("Hugging Face libraries not available")
            
        if task_name not in self.glue_tasks:
            raise ValueError(f"Unknown GLUE task: {task_name}")
            
        print(f"🔄 Loading GLUE task: {task_name} ({split})")
        
        try:
            # Load dataset
            dataset = load_dataset("glue", task_name, split=split, cache_dir=self.cache_dir)
            
            # Load tokenizer
            tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, cache_dir=self.cache_dir)
            
            print(f"✅ Loaded {len(dataset)} samples from {task_name}")
            
            return dataset, tokenizer
            
        except Exception as e:
            print(f"❌ Failed to load GLUE task {task_name}: {e}")
            raise
            
    def preprocess_dataset(self, 
                          dataset: HFDataset,
                          tokenizer: Any,
                          task_name: str,
                          max_samples: Optional[int] = None) -> HFDataset:
        """
        Preprocess GLUE dataset for MAHIA training
        
        Args:
            dataset: Hugging Face dataset
            tokenizer: Tokenizer to use
            task_name: Name of GLUE task
            max_samples: Maximum number of samples to use
            
        Returns:
            Preprocessed dataset
        """
        def preprocess_function(examples):
            # Task-specific preprocessing
            if task_name in ["cola", "sst2"]:
                # Single sentence tasks
                return tokenizer(
                    examples["sentence"],
                    truncation=True,
                    padding="max_length",
                    max_length=self.max_length
                )
            elif task_name in ["mrpc", "qqp", "mnli", "qnli", "rte", "wnli"]:
                # Sentence pair tasks
                return tokenizer(
                    examples["sentence1"],
                    examples["sentence2"],
                    truncation=True,
                    padding="max_length",
                    max_length=self.max_length
                )
            elif task_name == "stsb":
                # Regression task
                return tokenizer(
                    examples["sentence1"],
                    examples["sentence2"],
                    truncation=True,
                    padding="max_length",
                    max_length=self.max_length
                )
            else:
                # Default preprocessing
                return tokenizer(
                    examples["sentence"],
                    truncation=True,
                    padding="max_length",
                    max_length=self.max_length
                )
                
        # Apply preprocessing
        processed_dataset = dataset.map(
            preprocess_function,
            batched=True,
            remove_columns=dataset.column_names
        )
        
        # Add labels
        if task_name == "stsb":
            # Regression labels
            processed_dataset = processed_dataset.rename_column("label", "labels")
        else:
            # Classification labels
            processed_dataset = processed_dataset.rename_column("label", "labels")
            
        # Limit samples if specified
        if max_samples and len(processed_dataset) > max_samples:
            processed_dataset = processed_dataset.select(range(max_samples))
            
        return processed_dataset

class MMLUDatasetLoader:
    """
    Loader for MMLU (Massive Multitask Language Understanding) dataset
    """
    
    def __init__(self, 
                 cache_dir: str = "./hf_cache",
                 max_length: int = 512):
        """
        Initialize MMLU dataset loader
        
        Args:
            cache_dir: Directory to cache datasets
            max_length: Maximum sequence length
        """
        self.cache_dir = cache_dir
        self.max_length = max_length
        
        # MMLU subjects
        self.mmlu_subjects = [
            "abstract_algebra", "anatomy", "astronomy", "business_ethics",
            "clinical_knowledge", "college_biology", "college_chemistry",
            "college_computer_science", "college_mathematics", "college_physics",
            "computer_security", "conceptual_physics", "econometrics",
            "electrical_engineering", "elementary_mathematics", "formal_logic",
            "global_facts", "high_school_biology", "high_school_chemistry",
            "high_school_computer_science", "high_school_european_history",
            "high_school_geography", "high_school_government_and_politics",
            "high_school_macroeconomics", "high_school_mathematics",
            "high_school_microeconomics", "high_school_physics",
            "high_school_psychology", "high_school_statistics", "high_school_us_history",
            "high_school_world_history", "human_aging", "human_sexuality",
            "international_law", "jurisprudence", "logical_fallacies",
            "machine_learning", "management", "marketing", "medical_genetics",
            "miscellaneous", "moral_disputes", "moral_scenarios", "nutrition",
            "philosophy", "prehistory", "professional_accounting",
            "professional_law", "professional_medicine", "professional_psychology",
            "public_relations", "security_studies", "sociology", "us_foreign_policy",
            "virology", "world_religions"
        ]
        
        print(f"✅ MMLUDatasetLoader initialized")
        
    def load_subject(self, 
                     subject: str,
                     split: str = "test",
                     tokenizer_name: str = "bert-base-uncased") -> Tuple[HFDataset, Any]:
        """
        Load specific MMLU subject
        
        Args:
            subject: Name of MMLU subject
            split: Data split ("train", "validation", "test")
            tokenizer_name: Name of tokenizer to use
            
        Returns:
            Tuple of (dataset, tokenizer)
        """
        if not HUGGINGFACE_AVAILABLE:
            raise RuntimeError("Hugging Face libraries not available")
            
        if subject not in self.mmlu_subjects:
            raise ValueError(f"Unknown MMLU subject: {subject}")
            
        print(f"🔄 Loading MMLU subject: {subject} ({split})")
        
        try:
            # Load dataset
            dataset = load_dataset("cais/mmlu", subject, split=split, cache_dir=self.cache_dir)
            
            # Load tokenizer
            tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, cache_dir=self.cache_dir)
            
            print(f"✅ Loaded {len(dataset)} samples from {subject}")
            
            return dataset, tokenizer
            
        except Exception as e:
            print(f"❌ Failed to load MMLU subject {subject}: {e}")
            raise
            
    def preprocess_dataset(self, 
                          dataset: HFDataset,
                          tokenizer: Any,
                          max_samples: Optional[int] = None) -> HFDataset:
        """
        Preprocess MMLU dataset for MAHIA training
        
        Args:
            dataset: Hugging Face dataset
            tokenizer: Tokenizer to use
            max_samples: Maximum number of samples to use
            
        Returns:
            Preprocessed dataset
        """
        def preprocess_function(examples):
            # Combine question and choices
            questions = examples["question"]
            choices = examples["choices"]
            
            # Format as multiple choice
            formatted_questions = []
            for q, c in zip(questions, choices):
                choice_text = "\n".join([f"{i}. {choice}" for i, choice in enumerate(c)])
                formatted_q = f"Question: {q}\nChoices:\n{choice_text}\nAnswer:"
                formatted_questions.append(formatted_q)
                
            return tokenizer(
                formatted_questions,
                truncation=True,
                padding="max_length",
                max_length=self.max_length
            )
            
        # Apply preprocessing
        processed_dataset = dataset.map(
            preprocess_function,
            batched=True,
            remove_columns=dataset.column_names
        )
        
        # Add labels
        processed_dataset = processed_dataset.rename_column("answer", "labels")
        
        # Limit samples if specified
        if max_samples and len(processed_dataset) > max_samples:
            processed_dataset = processed_dataset.select(range(max_samples))
            
        return processed_dataset

class BIGBenchDatasetLoader:
    """
    Loader for BIG-Bench dataset
    """
    
    def __init__(self, 
                 cache_dir: str = "./hf_cache",
                 max_length: int = 512):
        """
        Initialize BIG-Bench dataset loader
        
        Args:
            cache_dir: Directory to cache datasets
            max_length: Maximum sequence length
        """
        self.cache_dir = cache_dir
        self.max_length = max_length
        
        print(f"✅ BIGBenchDatasetLoader initialized")
        
    def load_task(self, 
                  task_name: str,
                  split: str = "validation",
                  tokenizer_name: str = "bert-base-uncased") -> Tuple[HFDataset, Any]:
        """
        Load specific BIG-Bench task
        
        Args:
            task_name: Name of BIG-Bench task
            split: Data split ("train", "validation", "test")
            tokenizer_name: Name of tokenizer to use
            
        Returns:
            Tuple of (dataset, tokenizer)
        """
        if not HUGGINGFACE_AVAILABLE:
            raise RuntimeError("Hugging Face libraries not available")
            
        print(f"🔄 Loading BIG-Bench task: {task_name} ({split})")
        
        try:
            # Load dataset
            dataset = load_dataset("bigbench", task_name, split=split, cache_dir=self.cache_dir)
            
            # Load tokenizer
            tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, cache_dir=self.cache_dir)
            
            print(f"✅ Loaded {len(dataset)} samples from {task_name}")
            
            return dataset, tokenizer
            
        except Exception as e:
            print(f"❌ Failed to load BIG-Bench task {task_name}: {e}")
            raise
            
    def preprocess_dataset(self, 
                          dataset: HFDataset,
                          tokenizer: Any,
                          max_samples: Optional[int] = None) -> HFDataset:
        """
        Preprocess BIG-Bench dataset for MAHIA training
        
        Args:
            dataset: Hugging Face dataset
            tokenizer: Tokenizer to use
            max_samples: Maximum number of samples to use
            
        Returns:
            Preprocessed dataset
        """
        def preprocess_function(examples):
            # Task-specific preprocessing
            # This is a simplified implementation - BIG-Bench has diverse formats
            inputs = examples.get("inputs", examples.get("question", [""] * len(examples["targets"])))
            targets = examples["targets"]
            
            # Format as text generation task
            formatted_inputs = [f"Input: {inp}\nOutput:" for inp in inputs]
            
            return tokenizer(
                formatted_inputs,
                truncation=True,
                padding="max_length",
                max_length=self.max_length
            )
            
        # Apply preprocessing
        processed_dataset = dataset.map(
            preprocess_function,
            batched=True,
            remove_columns=dataset.column_names
        )
        
        # Add labels (targets)
        # This is simplified - BIG-Bench targets can be complex
        processed_dataset = processed_dataset.rename_column("targets", "labels")
        
        # Limit samples if specified
        if max_samples and len(processed_dataset) > max_samples:
            processed_dataset = processed_dataset.select(range(max_samples))
            
        return processed_dataset

class LongBenchDatasetLoader:
    """
    Loader for LongBench dataset - benchmark for long context understanding
    """
    
    def __init__(self, 
                 cache_dir: str = "./hf_cache",
                 max_length: int = 4096):  # Longer context for LongBench
        """
        Initialize LongBench dataset loader
        
        Args:
            cache_dir: Directory to cache datasets
            max_length: Maximum sequence length (longer for LongBench)
        """
        self.cache_dir = cache_dir
        self.max_length = max_length
        
        # LongBench tasks
        self.longbench_tasks = [
            "narrativeqa", "qasper", "multifieldqa_en", "multifieldqa_zh",
            "hotpotqa", "2wikimqa", "musique", "dureader",
            "gov_report", "qmsum", "multi_news", "vcsum",
            "trec", "triviaqa", "samsum", "lsht",
            "passage_count", "passage_retrieval_en", "passage_retrieval_zh",
            "lcc", "repobench-p"
        ]
        
        print(f"✅ LongBenchDatasetLoader initialized")
        
    def load_task(self, 
                  task_name: str,
                  split: str = "test",
                  tokenizer_name: str = "bert-base-uncased") -> Tuple[HFDataset, Any]:
        """
        Load specific LongBench task
        
        Args:
            task_name: Name of LongBench task
            split: Data split ("train", "validation", "test")
            tokenizer_name: Name of tokenizer to use
            
        Returns:
            Tuple of (dataset, tokenizer)
        """
        if not HUGGINGFACE_AVAILABLE:
            raise RuntimeError("Hugging Face libraries not available")
            
        if task_name not in self.longbench_tasks:
            raise ValueError(f"Unknown LongBench task: {task_name}")
            
        print(f"🔄 Loading LongBench task: {task_name} ({split})")
        
        try:
            # Load dataset
            dataset = load_dataset("zai-org/LongBench", task_name, split=split, cache_dir=self.cache_dir)
            
            # Load tokenizer
            tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, cache_dir=self.cache_dir)
            
            print(f"✅ Loaded {len(dataset)} samples from {task_name}")
            
            return dataset, tokenizer
            
        except Exception as e:
            print(f"❌ Failed to load LongBench task {task_name}: {e}")
            raise
            
    def preprocess_dataset(self, 
                          dataset: HFDataset,
                          tokenizer: Any,
                          max_samples: Optional[int] = None) -> HFDataset:
        """
        Preprocess LongBench dataset for MAHIA training
        
        Args:
            dataset: Hugging Face dataset
            tokenizer: Tokenizer to use
            max_samples: Maximum number of samples to use
            
        Returns:
            Preprocessed dataset
        """
        def preprocess_function(examples):
            # LongBench has various formats, we'll handle the most common ones
            if "context" in examples and "input" in examples:
                # Context + input format
                texts = [f"Context: {ctx}\n\nQuestion: {inp}" for ctx, inp in zip(examples["context"], examples["input"])]
            elif "content" in examples:
                # Content-only format
                texts = examples["content"]
            elif "question" in examples and "passage" in examples:
                # Question + passage format
                texts = [f"Passage: {passage}\n\nQuestion: {question}" for passage, question in zip(examples["passage"], examples["question"])]
            else:
                # Default to joining all text fields
                texts = []
                for i in range(len(examples[list(examples.keys())[0]])):
                    text_parts = [str(examples[key][i]) for key in examples.keys() if key != "answers"]
                    texts.append("\n".join(text_parts))
            
            return tokenizer(
                texts,
                truncation=True,
                padding="max_length",
                max_length=self.max_length
            )
            
        # Apply preprocessing
        processed_dataset = dataset.map(
            preprocess_function,
            batched=True,
            remove_columns=dataset.column_names
        )
        
        # Add labels (answers)
        if "answers" in dataset.column_names:
            processed_dataset = processed_dataset.rename_column("answers", "labels")
        
        # Limit samples if specified
        if max_samples and len(processed_dataset) > max_samples:
            processed_dataset = processed_dataset.select(range(max_samples))
            
        return processed_dataset

class MMMUDatasetLoader:
    """
    Loader for MMMU (Massive Multi-discipline Multimodal Understanding) dataset
    """
    
    def __init__(self, 
                 cache_dir: str = "./hf_cache",
                 max_length: int = 512):
        """
        Initialize MMMU dataset loader
        
        Args:
            cache_dir: Directory to cache datasets
            max_length: Maximum sequence length
        """
        self.cache_dir = cache_dir
        self.max_length = max_length
        
        # MMMU subjects
        self.mmmu_subjects = [
            "art", "business", "computer_science", "economics",
            "engineering", "health", "history", "law",
            "math", "other", "philosophy", "physics",
            "psychology"
        ]
        
        print(f"✅ MMMUDatasetLoader initialized")
        
    def load_subject(self, 
                     subject: str,
                     split: str = "test",
                     tokenizer_name: str = "bert-base-uncased") -> Tuple[HFDataset, Any]:
        """
        Load specific MMMU subject
        
        Args:
            subject: Name of MMMU subject
            split: Data split ("train", "validation", "test")
            tokenizer_name: Name of tokenizer to use
            
        Returns:
            Tuple of (dataset, tokenizer)
        """
        if not HUGGINGFACE_AVAILABLE:
            raise RuntimeError("Hugging Face libraries not available")
            
        if subject not in self.mmmu_subjects:
            raise ValueError(f"Unknown MMMU subject: {subject}")
            
        print(f"🔄 Loading MMMU subject: {subject} ({split})")
        
        try:
            # Load dataset
            dataset = load_dataset("MMMU/MMMU", subject, split=split, cache_dir=self.cache_dir)
            
            # Load tokenizer
            tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, cache_dir=self.cache_dir)
            
            print(f"✅ Loaded {len(dataset)} samples from {subject}")
            
            return dataset, tokenizer
            
        except Exception as e:
            print(f"❌ Failed to load MMMU subject {subject}: {e}")
            raise
            
    def preprocess_dataset(self, 
                          dataset: HFDataset,
                          tokenizer: Any,
                          max_samples: Optional[int] = None) -> HFDataset:
        """
        Preprocess MMMU dataset for MAHIA training
        
        Args:
            dataset: Hugging Face dataset
            tokenizer: Tokenizer to use
            max_samples: Maximum number of samples to use
            
        Returns:
            Preprocessed dataset
        """
        def preprocess_function(examples):
            # MMMU is multimodal, but for text processing we focus on questions
            questions = examples["question"]
            choices = examples.get("options", [""] * len(questions))  # Some samples might not have options
            
            # Format as multiple choice when options are available
            formatted_questions = []
            for q, c in zip(questions, choices):
                if c and isinstance(c, list):
                    choice_text = "\n".join([f"{chr(65+i)}. {choice}" for i, choice in enumerate(c)])
                    formatted_q = f"Question: {q}\nChoices:\n{choice_text}\nAnswer:"
                else:
                    formatted_q = f"Question: {q}\nAnswer:"
                formatted_questions.append(formatted_q)
                
            return tokenizer(
                formatted_questions,
                truncation=True,
                padding="max_length",
                max_length=self.max_length
            )
            
        # Apply preprocessing
        processed_dataset = dataset.map(
            preprocess_function,
            batched=True,
            remove_columns=dataset.column_names
        )
        
        # Add labels
        if "answer" in dataset.column_names:
            processed_dataset = processed_dataset.rename_column("answer", "labels")
        elif "label" in dataset.column_names:
            processed_dataset = processed_dataset.rename_column("label", "labels")
        
        # Limit samples if specified
        if max_samples and len(processed_dataset) > max_samples:
            processed_dataset = processed_dataset.select(range(max_samples))
            
        return processed_dataset

class MAHIADataset(Dataset):
    """
    PyTorch Dataset wrapper for MAHIA training
    """
    
    def __init__(self, 
                 hf_dataset: HFDataset,
                 task_type: str = "classification"):
        """
        Initialize MAHIA dataset
        
        Args:
            hf_dataset: Hugging Face dataset
            task_type: Type of task ("classification", "regression", "generation")
        """
        self.hf_dataset = hf_dataset
        self.task_type = task_type
        
    def __len__(self) -> int:
        """Get dataset length"""
        return len(self.hf_dataset)
        
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get item from dataset
        
        Args:
            idx: Index
            
        Returns:
            Dictionary of tensors
        """
        item = self.hf_dataset[idx]
        
        # Convert to tensors
        tensor_item = {}
        for key, value in item.items():
            if isinstance(value, list):
                tensor_item[key] = torch.tensor(value, dtype=torch.long)
            else:
                tensor_item[key] = torch.tensor(value, dtype=torch.long)
                
        return tensor_item

class HuggingFaceBenchmarkSuite:
    """
    Complete benchmark suite integrating GLUE, MMLU, BIG-Bench, LongBench, and MMMU
    """
    
    def __init__(self,
                 cache_dir: str = "./hf_cache",
                 max_length: int = 512):
        """
        Initialize benchmark suite
        
        Args:
            cache_dir: Directory to cache datasets
            max_length: Maximum sequence length
        """
        self.cache_dir = cache_dir
        self.max_length = max_length
        
        # Dataset loaders
        self.glue_loader = GLUEDatasetLoader(cache_dir, max_length)
        self.mmlu_loader = MMLUDatasetLoader(cache_dir, max_length)
        self.bigbench_loader = BIGBenchDatasetLoader(cache_dir, max_length)
        self.longbench_loader = LongBenchDatasetLoader(cache_dir, max(4096, max_length))  # Longer context
        self.mmmu_loader = MMMUDatasetLoader(cache_dir, max_length)
        
        # Loaded datasets
        self.loaded_datasets = {}
        
        # Benchmark results storage
        self.benchmark_results = {}
        
        print(f"✅ HuggingFaceBenchmarkSuite initialized")
        
    def load_glue_tasks(self, 
                       tasks: List[str] = None,
                       split: str = "validation",
                       tokenizer_name: str = "bert-base-uncased",
                       max_samples_per_task: int = 1000) -> Dict[str, DataLoader]:
        """
        Load GLUE tasks
        
        Args:
            tasks: List of GLUE tasks to load (None for all)
            split: Data split to load
            tokenizer_name: Tokenizer to use
            max_samples_per_task: Maximum samples per task
            
        Returns:
            Dictionary mapping task names to DataLoaders
        """
        if tasks is None:
            tasks = self.glue_loader.glue_tasks
            
        dataloaders = {}
        
        for task in tasks:
            try:
                print(f"🔄 Loading GLUE task: {task}")
                
                # Load dataset
                dataset, tokenizer = self.glue_loader.load_task(task, split, tokenizer_name)
                
                # Preprocess dataset
                processed_dataset = self.glue_loader.preprocess_dataset(
                    dataset, tokenizer, task, max_samples_per_task
                )
                
                # Create PyTorch dataset
                pytorch_dataset = MAHIADataset(processed_dataset, "classification")
                
                # Create DataLoader
                dataloader = DataLoader(
                    pytorch_dataset,
                    batch_size=32,
                    shuffle=False,
                    num_workers=0  # Set to 0 to avoid multiprocessing issues
                )
                
                dataloaders[task] = dataloader
                self.loaded_datasets[f"glue_{task}"] = processed_dataset
                
                print(f"✅ Loaded GLUE task: {task} ({len(processed_dataset)} samples)")
                
            except Exception as e:
                print(f"⚠️  Failed to load GLUE task {task}: {e}")
                
        return dataloaders
        
    def load_mmlu_subjects(self, 
                          subjects: List[str] = None,
                          split: str = "test",
                          tokenizer_name: str = "bert-base-uncased",
                          max_samples_per_subject: int = 100) -> Dict[str, DataLoader]:
        """
        Load MMLU subjects
        
        Args:
            subjects: List of MMLU subjects to load (None for all)
            split: Data split to load
            tokenizer_name: Tokenizer to use
            max_samples_per_subject: Maximum samples per subject
            
        Returns:
            Dictionary mapping subject names to DataLoaders
        """
        if subjects is None:
            subjects = self.mmlu_loader.mmlu_subjects[:5]  # Load first 5 for demo
            
        dataloaders = {}
        
        for subject in subjects:
            try:
                print(f"🔄 Loading MMLU subject: {subject}")
                
                # Load dataset
                dataset, tokenizer = self.mmlu_loader.load_subject(subject, split, tokenizer_name)
                
                # Preprocess dataset
                processed_dataset = self.mmlu_loader.preprocess_dataset(
                    dataset, tokenizer, max_samples_per_subject
                )
                
                # Create PyTorch dataset
                pytorch_dataset = MAHIADataset(processed_dataset, "classification")
                
                # Create DataLoader
                dataloader = DataLoader(
                    pytorch_dataset,
                    batch_size=16,
                    shuffle=False,
                    num_workers=0  # Set to 0 to avoid multiprocessing issues
                )
                
                dataloaders[subject] = dataloader
                self.loaded_datasets[f"mmlu_{subject}"] = processed_dataset
                
                print(f"✅ Loaded MMLU subject: {subject} ({len(processed_dataset)} samples)")
                
            except Exception as e:
                print(f"⚠️  Failed to load MMLU subject {subject}: {e}")
                
        return dataloaders
        
    def load_bigbench_tasks(self, 
                           tasks: List[str] = ["simple_arithmetic_json_multiple_choice"],
                           split: str = "validation",
                           tokenizer_name: str = "bert-base-uncased",
                           max_samples_per_task: int = 50) -> Dict[str, DataLoader]:
        """
        Load BIG-Bench tasks
        
        Args:
            tasks: List of BIG-Bench tasks to load
            split: Data split to load
            tokenizer_name: Tokenizer to use
            max_samples_per_task: Maximum samples per task
            
        Returns:
            Dictionary mapping task names to DataLoaders
        """
        dataloaders = {}
        
        for task in tasks:
            try:
                print(f"🔄 Loading BIG-Bench task: {task}")
                
                # Load dataset
                dataset, tokenizer = self.bigbench_loader.load_task(task, split, tokenizer_name)
                
                # Preprocess dataset
                processed_dataset = self.bigbench_loader.preprocess_dataset(
                    dataset, tokenizer, max_samples_per_task
                )
                
                # Create PyTorch dataset
                pytorch_dataset = MAHIADataset(processed_dataset, "generation")
                
                # Create DataLoader
                dataloader = DataLoader(
                    pytorch_dataset,
                    batch_size=8,
                    shuffle=False,
                    num_workers=0  # Set to 0 to avoid multiprocessing issues
                )
                
                dataloaders[task] = dataloader
                self.loaded_datasets[f"bigbench_{task}"] = processed_dataset
                
                print(f"✅ Loaded BIG-Bench task: {task} ({len(processed_dataset)} samples)")
                
            except Exception as e:
                print(f"⚠️  Failed to load BIG-Bench task {task}: {e}")
                
        return dataloaders
        
    def load_longbench_tasks(self, 
                            tasks: List[str] = None,
                            split: str = "test",
                            tokenizer_name: str = "bert-base-uncased",
                            max_samples_per_task: int = 50) -> Dict[str, DataLoader]:
        """
        Load LongBench tasks for long context evaluation
        
        Args:
            tasks: List of LongBench tasks to load (None for all)
            split: Data split to load
            tokenizer_name: Tokenizer to use
            max_samples_per_task: Maximum samples per task
            
        Returns:
            Dictionary mapping task names to DataLoaders
        """
        if tasks is None:
            tasks = self.longbench_loader.longbench_tasks
            
        dataloaders = {}
        
        for task in tasks:
            try:
                print(f"🔄 Loading LongBench task: {task}")
                
                # Load dataset
                dataset, tokenizer = self.longbench_loader.load_task(task, split, tokenizer_name)
                
                # Preprocess dataset
                processed_dataset = self.longbench_loader.preprocess_dataset(
                    dataset, tokenizer, max_samples_per_task
                )
                
                # Create PyTorch dataset
                pytorch_dataset = MAHIADataset(processed_dataset, "generation")
                
                # Create DataLoader
                dataloader = DataLoader(
                    pytorch_dataset,
                    batch_size=4,  # Smaller batch size for long sequences
                    shuffle=False,
                    num_workers=0  # Set to 0 to avoid multiprocessing issues
                )
                
                dataloaders[task] = dataloader
                self.loaded_datasets[f"longbench_{task}"] = processed_dataset
                
                print(f"✅ Loaded LongBench task: {task} ({len(processed_dataset)} samples)")
                
            except Exception as e:
                print(f"⚠️  Failed to load LongBench task {task}: {e}")
                
        return dataloaders
        
    def load_mmmu_subjects(self, 
                          subjects: List[str] = None,
                          split: str = "test",
                          tokenizer_name: str = "bert-base-uncased",
                          max_samples_per_subject: int = 50) -> Dict[str, DataLoader]:
        """
        Load MMMU subjects for multimodal evaluation
        
        Args:
            subjects: List of MMMU subjects to load (None for all)
            split: Data split to load
            tokenizer_name: Tokenizer to use
            max_samples_per_subject: Maximum samples per subject
            
        Returns:
            Dictionary mapping subject names to DataLoaders
        """
        if subjects is None:
            subjects = self.mmmu_loader.mmmu_subjects
            
        dataloaders = {}
        
        for subject in subjects:
            try:
                print(f"🔄 Loading MMMU subject: {subject}")
                
                # Load dataset
                dataset, tokenizer = self.mmmu_loader.load_subject(subject, split, tokenizer_name)
                
                # Preprocess dataset
                processed_dataset = self.mmmu_loader.preprocess_dataset(
                    dataset, tokenizer, max_samples_per_subject
                )
                
                # Create PyTorch dataset
                pytorch_dataset = MAHIADataset(processed_dataset, "classification")
                
                # Create DataLoader
                dataloader = DataLoader(
                    pytorch_dataset,
                    batch_size=8,
                    shuffle=False,
                    num_workers=0  # Set to 0 to avoid multiprocessing issues
                )
                
                dataloaders[subject] = dataloader
                self.loaded_datasets[f"mmmu_{subject}"] = processed_dataset
                
                print(f"✅ Loaded MMMU subject: {subject} ({len(processed_dataset)} samples)")
                
            except Exception as e:
                print(f"⚠️  Failed to load MMMU subject {subject}: {e}")
                
        return dataloaders
        
    def get_dataset_info(self) -> Dict[str, int]:
        """
        Get information about loaded datasets
        
        Returns:
            Dictionary mapping dataset names to sample counts
        """
        info = {}
        for name, dataset in self.loaded_datasets.items():
            info[name] = len(dataset)
        return info
        
    def print_benchmark_info(self):
        """Print benchmark information"""
        info = self.get_dataset_info()
        
        print("\n" + "="*60)
        print("HUGGING FACE BENCHMARK SUITE INFORMATION")
        print("="*60)
        print(f"Loaded Datasets: {len(info)}")
        for name, count in info.items():
            print(f"  {name}: {count:,} samples")
        print("="*60)
        
    def add_benchmark_result(self, benchmark_name: str, result: Dict[str, Any]):
        """
        Add a benchmark result to the results storage
        
        Args:
            benchmark_name: Name of the benchmark
            result: Dictionary containing benchmark results
        """
        self.benchmark_results[benchmark_name] = result
        
    def generate_json_report(self, output_path: str = "benchmark_results.json"):
        """
        Generate a JSON report of all benchmark results
        
        Args:
            output_path: Path to save the JSON report
        """
        if not PANDAS_AVAILABLE:
            print("⚠️  Pandas not available, cannot generate JSON report")
            return
            
        # Prepare report data
        report_data = {
            "benchmark_suite": "HuggingFaceBenchmarkSuite",
            "datasets_loaded": self.get_dataset_info(),
            "results": self.benchmark_results,
            "generated_at": pd.Timestamp.now().isoformat()
        }
        
        # Save to JSON file
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(report_data, f, indent=2, ensure_ascii=False)
            
        print(f"✅ JSON report saved to {output_path}")
        
    def generate_csv_report(self, output_path: str = "benchmark_results.csv"):
        """
        Generate a CSV report of all benchmark results
        
        Args:
            output_path: Path to save the CSV report
        """
        if not PANDAS_AVAILABLE:
            print("⚠️  Pandas not available, cannot generate CSV report")
            return
            
        # Flatten results for CSV
        csv_data = []
        for benchmark_name, result in self.benchmark_results.items():
            row = {"benchmark": benchmark_name}
            row.update(result)
            csv_data.append(row)
            
        # Create DataFrame and save to CSV
        df = pd.DataFrame(csv_data)
        df.to_csv(output_path, index=False)
        
        print(f"✅ CSV report saved to {output_path}")
        
    def generate_comprehensive_report(self, 
                                    json_path: str = "benchmark_results.json",
                                    csv_path: str = "benchmark_results.csv"):
        """
        Generate comprehensive reports in both JSON and CSV formats
        
        Args:
            json_path: Path to save the JSON report
            csv_path: Path to save the CSV report
        """
        self.generate_json_report(json_path)
        self.generate_csv_report(csv_path)
        
        print("✅ Comprehensive benchmark reports generated")

# Example usage
def example_huggingface_integration():
    """Example of Hugging Face integration usage"""
    print("🔧 Setting up Hugging Face Integration example...")
    
    # Check if Hugging Face is available
    if not HUGGINGFACE_AVAILABLE:
        print("⚠️  Hugging Face libraries not available. Skipping example.")
        return
        
    # Create benchmark suite
    benchmark_suite = HuggingFaceBenchmarkSuite(
        cache_dir="./hf_cache_demo",
        max_length=256  # Shorter for demo
    )
    
    print("\n🚀 Loading GLUE tasks...")
    # Load a few GLUE tasks for demo
    glue_tasks = ["sst2", "qqp"]  # Simplified tasks for quick demo
    glue_loaders = benchmark_suite.load_glue_tasks(
        tasks=glue_tasks,
        split="validation",
        tokenizer_name="bert-base-uncased",
        max_samples_per_task=100
    )
    
    print("\n🚀 Loading MMLU subjects...")
    # Load a few MMLU subjects for demo
    mmlu_subjects = ["abstract_algebra", "anatomy"]
    mmlu_loaders = benchmark_suite.load_mmlu_subjects(
        subjects=mmlu_subjects,
        split="test",
        tokenizer_name="bert-base-uncased",
        max_samples_per_subject=50
    )
    
    print("\n🚀 Loading BIG-Bench tasks...")
    # Load a BIG-Bench task for demo
    bigbench_tasks = ["simple_arithmetic_json_multiple_choice"]
    bigbench_loaders = benchmark_suite.load_bigbench_tasks(
        tasks=bigbench_tasks,
        split="validation",
        tokenizer_name="bert-base-uncased",
        max_samples_per_task=25
    )
    
    print("\n🚀 Loading LongBench tasks...")
    # Load a LongBench task for demo
    longbench_tasks = ["narrativeqa"]
    longbench_loaders = benchmark_suite.load_longbench_tasks(
        tasks=longbench_tasks,
        split="test",
        tokenizer_name="bert-base-uncased",
        max_samples_per_task=10
    )
    
    print("\n🚀 Loading MMMU subjects...")
    # Load an MMMU subject for demo
    mmmu_subjects = ["math", "physics"]
    mmmu_loaders = benchmark_suite.load_mmmu_subjects(
        subjects=mmmu_subjects,
        split="test",
        tokenizer_name="bert-base-uncased",
        max_samples_per_subject=10
    )
    
    # Print benchmark information
    benchmark_suite.print_benchmark_info()
    
    # Demonstrate data loading
    print("\n🔍 Demonstrating data loading...")
    
    # Show sample from GLUE SST-2
    if "sst2" in glue_loaders:
        dataloader = glue_loaders["sst2"]
        print(f"\n📊 GLUE SST-2 DataLoader: {len(dataloader)} batches")
        
        # Get first batch
        for batch in dataloader:
            print(f"   Batch keys: {list(batch.keys())}")
            print(f"   Input shape: {batch['input_ids'].shape}")
            print(f"   Labels shape: {batch['labels'].shape}")
            break
            
    # Show sample from MMLU
    if mmlu_subjects and mmlu_subjects[0] in mmlu_loaders:
        dataloader = mmlu_loaders[mmlu_subjects[0]]
        print(f"\n📊 MMLU {mmlu_subjects[0]} DataLoader: {len(dataloader)} batches")
        
        # Get first batch
        for batch in dataloader:
            print(f"   Batch keys: {list(batch.keys())}")
            print(f"   Input shape: {batch['input_ids'].shape}")
            print(f"   Labels shape: {batch['labels'].shape}")
            break
            
    # Show sample from LongBench
    if longbench_tasks and longbench_tasks[0] in longbench_loaders:
        dataloader = longbench_loaders[longbench_tasks[0]]
        print(f"\n📊 LongBench {longbench_tasks[0]} DataLoader: {len(dataloader)} batches")
        
        # Get first batch
        for batch in dataloader:
            print(f"   Batch keys: {list(batch.keys())}")
            print(f"   Input shape: {batch['input_ids'].shape}")
            print(f"   Labels shape: {batch['labels'].shape}")
            break
            
    # Show sample from MMMU
    if mmmu_subjects and mmmu_subjects[0] in mmmu_loaders:
        dataloader = mmmu_loaders[mmmu_subjects[0]]
        print(f"\n📊 MMMU {mmmu_subjects[0]} DataLoader: {len(dataloader)} batches")
        
        # Get first batch
        for batch in dataloader:
            print(f"   Batch keys: {list(batch.keys())}")
            print(f"   Input shape: {batch['input_ids'].shape}")
            print(f"   Labels shape: {batch['labels'].shape}")
            break
            
    print("\n✅ Hugging Face Integration example completed!")

if __name__ == "__main__":
    example_huggingface_integration()