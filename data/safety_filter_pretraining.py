"""
Safety Filter Pretraining for MAHIA-X
Implements safety filter pretraining with safety-loss on toxicity/bias scores
"""

import math
import time
import json
import os
from typing import Dict, Any, Optional, List, Union, Tuple
from collections import OrderedDict, defaultdict
import re

# Conditional imports with fallbacks
TORCH_AVAILABLE = False

# Define fallback classes for when torch is not available
Dataset = object
DataLoader = None
Module = object

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import Dataset, DataLoader
    from torch.nn import Module
    TORCH_AVAILABLE = True
except ImportError:
    # Create minimal fallbacks
    class DummyModule:
        def __init__(self):
            pass
        def parameters(self):
            return []
    Module = DummyModule
    
    class DummyDataset:
        def __len__(self):
            return 0
    Dataset = DummyDataset
    
    pass

NUMPY_AVAILABLE = False
np = None

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    pass

# Conditional imports with fallbacks
NUMPY_AVAILABLE = False
np = None

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    pass

class SafetyFilterDataset(Dataset):
    """Dataset for safety filter training"""
    
    def __init__(self, texts: List[str], labels: List[int], 
                 toxicity_scores: Optional[List[float]] = None,
                 bias_scores: Optional[List[float]] = None):
        """
        Initialize safety filter dataset
        
        Args:
            texts: List of text samples
            labels: Safety labels (0 = safe, 1 = unsafe)
            toxicity_scores: Optional toxicity scores for each sample
            bias_scores: Optional bias scores for each sample
        """
        self.texts = texts
        self.labels = labels
        self.toxicity_scores = toxicity_scores or [0.0] * len(texts)
        self.bias_scores = bias_scores or [0.0] * len(texts)
        
    def __len__(self):
        return len(self.texts)
        
    def __getitem__(self, idx):
        return {
            "text": self.texts[idx],
            "label": self.labels[idx],
            "toxicity_score": self.toxicity_scores[idx],
            "bias_score": self.bias_scores[idx]
        }

class SafetyFilterModel(Module):
    """Simple safety filter model based on text features"""
    
    def __init__(self, vocab_size: int = 10000, embedding_dim: int = 128, 
                 hidden_dim: int = 256, num_layers: int = 2):
        """
        Initialize safety filter model
        
        Args:
            vocab_size: Size of vocabulary
            embedding_dim: Dimension of word embeddings
            hidden_dim: Dimension of hidden layers
            num_layers: Number of LSTM layers
        """
        try:
            super().__init__()
        except:
            pass  # Fallback for when torch is not available
        
        if TORCH_AVAILABLE:
            self.embedding = nn.Embedding(vocab_size, embedding_dim)
            self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers, 
                               batch_first=True, dropout=0.2)
            self.dropout = nn.Dropout(0.3)
            self.classifier = nn.Linear(hidden_dim, 2)  # Safe/Unsafe classification
            self.sigmoid = nn.Sigmoid()
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        """Initialize model weights"""
        if TORCH_AVAILABLE:
            for name, param in self.named_parameters():
                if 'weight' in name:
                    if len(param.shape) >= 2:
                        nn.init.xavier_uniform_(param)
                    else:
                        nn.init.uniform_(param, -0.1, 0.1)
                elif 'bias' in name:
                    nn.init.constant_(param, 0)
                
    def forward(self, input_ids):
        """
        Forward pass
        
        Args:
            input_ids: Input token IDs
            
        Returns:
            Logits for safety classification
        """
        if TORCH_AVAILABLE:
            # Embedding
            embedded = self.embedding(input_ids)
            
            # LSTM
            lstm_out, (hidden, _) = self.lstm(embedded)
            
            # Use last hidden state
            last_hidden = hidden[-1]
            
            # Classification
            dropped = self.dropout(last_hidden)
            logits = self.classifier(dropped)
        else:
            # Fallback for when torch is not available
            logits = torch.tensor([[0.5, 0.5]] * input_ids.size(0)) if torch else None
        
        return logits

class SafetyFilterPretrainer:
    """Safety filter pretrainer with safety-loss on toxicity/bias scores"""
    
    def __init__(self, model: SafetyFilterModel, 
                 bias_detection_toolkit: Optional[Any] = None,
                 device: str = "cpu"):
        """
        Initialize safety filter pretrainer
        
        Args:
            model: Safety filter model to train
            bias_detection_toolkit: Optional bias detection toolkit for bias scoring
            device: Device to use for training (cpu/cuda)
        """
        self.model = model
        self.bias_detection_toolkit = bias_detection_toolkit
        self.device = device
        
        # Move model to device
        if TORCH_AVAILABLE:
            if TORCH_AVAILABLE:
            self.model.to(self.device)
        
        # Training components
        self.optimizer = None
        self.scheduler = None
        self.training_history = OrderedDict()
        
        print("✅ SafetyFilterPretrainer initialized")
        print(f"   Device: {device}")
        print(f"   Model parameters: {sum(p.numel() for p in model.parameters()):,}")
        
    def prepare_optimizer(self, learning_rate: float = 1e-4, 
                         weight_decay: float = 1e-5):
        """
        Prepare optimizer for training
        
        Args:
            learning_rate: Learning rate for optimizer
            weight_decay: Weight decay for regularization
        """
        if TORCH_AVAILABLE:
            self.optimizer = torch.optim.AdamW(
                list(self.model.parameters()), 
                lr=learning_rate, 
                weight_decay=weight_decay
            )
        
        # Learning rate scheduler
        if TORCH_AVAILABLE and self.optimizer:
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=1000, eta_min=1e-6
            )
        
        print(f"✅ Optimizer prepared with LR={learning_rate}, WD={weight_decay}")
        
    def safety_loss(self, logits: torch.Tensor, labels: torch.Tensor,
                   toxicity_scores: torch.Tensor, bias_scores: torch.Tensor,
                   alpha: float = 0.7, beta: float = 0.2, gamma: float = 0.1) -> torch.Tensor:
        """
        Calculate safety loss combining classification loss with toxicity/bias scores
        
        Args:
            logits: Model logits
            labels: Ground truth labels
            toxicity_scores: Toxicity scores for each sample
            bias_scores: Bias scores for each sample
            alpha: Weight for classification loss
            beta: Weight for toxicity penalty
            gamma: Weight for bias penalty
            
        Returns:
            Combined safety loss
        """
        # Standard classification loss
        classification_loss = F.cross_entropy(logits, labels)
        
        # Get predicted probabilities
        probs = F.softmax(logits, dim=-1)
        unsafe_probs = probs[:, 1]  # Probability of being unsafe
        
        # Toxicity penalty - encourage low unsafe probability for low toxicity samples
        toxicity_penalty = torch.mean(unsafe_probs * toxicity_scores)
        
        # Bias penalty - encourage low unsafe probability for low bias samples
        bias_penalty = torch.mean(unsafe_probs * bias_scores)
        
        # Combined loss
        total_loss = (alpha * classification_loss + 
                     beta * toxicity_penalty + 
                     gamma * bias_penalty)
        
        return total_loss
        
    def train_epoch(self, dataloader: DataLoader, epoch: int) -> Dict[str, float]:
        """
        Train for one epoch
        
        Args:
            dataloader: Training data loader
            epoch: Current epoch number
            
        Returns:
            Training metrics
        """
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, batch in enumerate(dataloader):
            # Move data to device
            texts = batch["text"]
            labels = batch["label"].to(self.device)
            toxicity_scores = batch["toxicity_score"].to(self.device)
            bias_scores = batch["bias_score"].to(self.device)
            
            # Simple tokenization (in practice, use proper tokenizer)
            input_ids = self._simple_tokenize(texts).to(self.device)
            
            # Forward pass
            logits = self.model(input_ids)
            
            # Calculate loss
            loss = self.safety_loss(logits, labels, toxicity_scores, bias_scores)
            
            # Backward pass
            if self.optimizer:
                self.optimizer.zero_grad()
            loss.backward()
            if TORCH_AVAILABLE:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            if self.optimizer:
                self.optimizer.step()
            
            # Update metrics
            total_loss += loss.item()
            preds = torch.argmax(logits, dim=-1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            
            # Update learning rate
            if self.scheduler and TORCH_AVAILABLE:
                self.scheduler.step()
                
            # Print progress
            if batch_idx % 50 == 0:
                print(f"   Epoch {epoch}, Batch {batch_idx}: Loss={loss.item():.4f}")
                
        # Calculate epoch metrics
        avg_loss = total_loss / len(dataloader)
        accuracy = correct / total if total > 0 else 0.0
        
        metrics = {
            "epoch": epoch,
            "loss": avg_loss,
            "accuracy": accuracy,
            "learning_rate": self.optimizer.param_groups[0]['lr'] if self.optimizer else 0.0
        }
        
        return metrics
        
    def _simple_tokenize(self, texts: List[str], max_length: int = 128) -> torch.Tensor:
        """
        Simple tokenization for demonstration purposes
        In practice, use proper tokenizer like BERT tokenizer
        
        Args:
            texts: List of text samples
            max_length: Maximum sequence length
            
        Returns:
            Tokenized tensor
        """
        # Simple word-based tokenization
        tokenized = []
        for text in texts:
            # Convert to lowercase and split
            words = text.lower().split()
            # Convert words to indices (simple hash-based approach)
            indices = [hash(word) % 10000 for word in words[:max_length]]
            # Pad to max_length
            indices = indices + [0] * (max_length - len(indices))
            tokenized.append(indices)
            
        return torch.tensor(tokenized, dtype=torch.long)
        
    def train(self, train_dataset: SafetyFilterDataset, 
              val_dataset: Optional[SafetyFilterDataset] = None,
              epochs: int = 10, batch_size: int = 32, 
              learning_rate: float = 1e-4) -> Dict[str, Any]:
        """
        Train safety filter model
        
        Args:
            train_dataset: Training dataset
            val_dataset: Optional validation dataset
            epochs: Number of training epochs
            batch_size: Batch size
            learning_rate: Learning rate
            
        Returns:
            Training results
        """
        print(f"🚀 Starting safety filter pretraining for {epochs} epochs...")
        
        # Prepare optimizer
        self.prepare_optimizer(learning_rate)
        
        # Create data loaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size) if val_dataset else None
        
        # Training loop
        best_val_accuracy = 0.0
        training_start_time = time.time()
        
        for epoch in range(1, epochs + 1):
            # Train epoch
            train_metrics = self.train_epoch(train_loader, epoch)
            
            # Validation
            val_metrics = None
            if val_loader:
                val_metrics = self.evaluate(val_loader)
                print(f"   Epoch {epoch} - Train Loss: {train_metrics['loss']:.4f}, "
                      f"Train Acc: {train_metrics['accuracy']:.4f}, "
                      f"Val Acc: {val_metrics['accuracy']:.4f}")
                
                # Save best model
                if val_metrics['accuracy'] > best_val_accuracy:
                    best_val_accuracy = val_metrics['accuracy']
                    self.save_model("best_safety_filter_model.pth")
            else:
                print(f"   Epoch {epoch} - Loss: {train_metrics['loss']:.4f}, "
                      f"Accuracy: {train_metrics['accuracy']:.4f}")
                
            # Store metrics
            self.training_history[f"epoch_{epoch}"] = {
                "train": train_metrics,
                "validation": val_metrics
            }
            
        training_time = time.time() - training_start_time
        print(f"✅ Safety filter pretraining completed in {training_time:.2f}s")
        
        # Save final model
        self.save_model("final_safety_filter_model.pth")
        
        results = {
            "training_time": training_time,
            "best_validation_accuracy": best_val_accuracy,
            "final_training_metrics": train_metrics,
            "training_history": dict(self.training_history)
        }
        
        return results
        
    def evaluate(self, dataloader: DataLoader) -> Dict[str, float]:
        """
        Evaluate model on dataset
        
        Args:
            dataloader: Data loader for evaluation
            
        Returns:
            Evaluation metrics
        """
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch in dataloader:
                # Move data to device
                texts = batch["text"]
                labels = batch["label"].to(self.device)
                toxicity_scores = batch["toxicity_score"].to(self.device)
                bias_scores = batch["bias_score"].to(self.device)
                
                # Simple tokenization
                input_ids = self._simple_tokenize(texts).to(self.device)
                
                # Forward pass
                logits = self.model(input_ids)
                
                # Calculate loss
                loss = self.safety_loss(logits, labels, toxicity_scores, bias_scores)
                
                # Update metrics
                total_loss += loss.item()
                preds = torch.argmax(logits, dim=-1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
                
        avg_loss = total_loss / len(dataloader)
        accuracy = correct / total if total > 0 else 0.0
        
        return {
            "loss": avg_loss,
            "accuracy": accuracy
        }
        
    def predict_safety(self, texts: List[str]) -> List[Dict[str, Any]]:
        """
        Predict safety scores for texts
        
        Args:
            texts: List of texts to evaluate
            
        Returns:
            List of safety predictions
        """
        self.model.eval()
        predictions = []
        
        with torch.no_grad():
            # Simple tokenization
            input_ids = self._simple_tokenize(texts).to(self.device)
            
            # Forward pass
            logits = self.model(input_ids)
            probabilities = F.softmax(logits, dim=-1)
            
            # Extract unsafe probabilities
            unsafe_probs = probabilities[:, 1].cpu().numpy()
            
            for i, text in enumerate(texts):
                predictions.append({
                    "text": text,
                    "unsafe_probability": float(unsafe_probs[i]),
                    "is_safe": bool(unsafe_probs[i] < 0.5),
                    "safety_score": float(1.0 - unsafe_probs[i])
                })
                
        return predictions
        
    def save_model(self, filepath: str):
        """
        Save model to file
        
        Args:
            filepath: Path to save model
        """
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict() if self.optimizer else None,
        }, filepath)
        print(f"✅ Model saved to {filepath}")
        
    def load_model(self, filepath: str):
        """
        Load model from file
        
        Args:
            filepath: Path to load model from
        """
        if TORCH_AVAILABLE:
            checkpoint = torch.load(filepath, map_location=self.device)
        if TORCH_AVAILABLE:
            self.model.load_state_dict(checkpoint['model_state_dict'])
            if self.optimizer and checkpoint['optimizer_state_dict']:
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print(f"✅ Model loaded from {filepath}")


def create_sample_safety_data() -> Tuple[SafetyFilterDataset, SafetyFilterDataset]:
    """
    Create sample safety training data for demonstration
    
    Returns:
        Tuple of (train_dataset, val_dataset)
    """
    # Sample safe texts
    safe_texts = [
        "The weather is beautiful today.",
        "I enjoy reading books in my free time.",
        "Technology has improved our lives significantly.",
        "Education is important for personal development.",
        "Nature provides us with many benefits.",
        "Collaboration leads to better results.",
        "Healthy eating habits improve wellbeing.",
        "Exercise is beneficial for physical health.",
        "Music has a positive impact on mood.",
        "Traveling broadens our perspectives."
    ]
    
    # Sample unsafe texts
    unsafe_texts = [
        "This group of people are all lazy and stupid.",
        "I hate everyone who disagrees with me.",
        "Violence is the only solution to our problems.",
        "Certain races are inferior to others.",
        "Women should not be allowed to work.",
        "People with disabilities are burdens on society.",
        "Religious groups should be eliminated.",
        "Children should be afraid of their parents.",
        "The elderly are useless and should be ignored.",
        "Homosexuals are morally corrupt."
    ]
    
    # Create training data
    train_texts = safe_texts * 3 + unsafe_texts * 3  # 30 samples each
    train_labels = [0] * 30 + [1] * 30  # 0 = safe, 1 = unsafe
    train_toxicity = [0.1] * 30 + [0.8] * 30  # Low toxicity for safe, high for unsafe
    train_bias = [0.05] * 30 + [0.9] * 30  # Low bias for safe, high for unsafe
    
    # Create validation data
    val_texts = safe_texts + unsafe_texts  # 10 samples each
    val_labels = [0] * 10 + [1] * 10
    val_toxicity = [0.15] * 10 + [0.75] * 10
    val_bias = [0.1] * 10 + [0.85] * 10
    
    # Create datasets
    train_dataset = SafetyFilterDataset(
        train_texts, train_labels, train_toxicity, train_bias
    )
    
    val_dataset = SafetyFilterDataset(
        val_texts, val_labels, val_toxicity, val_bias
    )
    
    return train_dataset, val_dataset


def demo_safety_filter_pretraining():
    """Demonstrate safety filter pretraining functionality"""
    print("🚀 Demonstrating Safety Filter Pretraining...")
    print("=" * 50)
    
    # Check if CUDA is available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Create model
    model = SafetyFilterModel(vocab_size=10000, embedding_dim=128, hidden_dim=256)
    print("✅ Created safety filter model")
    
    # Create pretrainer
    pretrainer = SafetyFilterPretrainer(model, device=device)
    print("✅ Created safety filter pretrainer")
    
    # Create sample data
    print("🔄 Creating sample safety training data...")
    train_dataset, val_dataset = create_sample_safety_data()
    print(f"✅ Created datasets: {len(train_dataset)} training samples, {len(val_dataset)} validation samples")
    
    # Train model
    print("🏋️ Starting safety filter training...")
    results = pretrainer.train(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        epochs=5,
        batch_size=8,
        learning_rate=1e-3
    )
    
    print(f"✅ Training completed in {results['training_time']:.2f}s")
    print(f"Best validation accuracy: {results['best_validation_accuracy']:.4f}")
    
    # Test predictions
    print("\n🔍 Testing safety predictions...")
    test_texts = [
        "I love spending time with my family.",
        "All people from that country are criminals.",
        "The sunset is beautiful today.",
        "Women are not as intelligent as men."
    ]
    
    predictions = pretrainer.predict_safety(test_texts)
    for pred in predictions:
        print(f"   Text: '{pred['text']}'")
        print(f"   Safety Score: {pred['safety_score']:.3f}")
        print(f"   Is Safe: {pred['is_safe']}")
        print()
    
    # Save model
    pretrainer.save_model("demo_safety_filter_model.pth")
    
    print("=" * 50)
    print("SAFETY FILTER PRETRAINING DEMO SUMMARY")
    print("=" * 50)
    print("Key Features Implemented:")
    print("  1. Safety filter model with LSTM architecture")
    print("  2. Safety loss combining classification with toxicity/bias scores")
    print("  3. Training with sample safe/unsafe data")
    print("  4. Model evaluation and prediction capabilities")
    print("  5. Model saving/loading functionality")
    print("\nBenefits:")
    print("  - Proactive filtering of unsafe content")
    print("  - Integration with bias/toxicity scoring")
    print("  - Customizable safety thresholds")
    print("  - Extensible architecture for advanced features")
    
    print("\n✅ Safety filter pretraining demonstration completed!")


if __name__ == "__main__":
    demo_safety_filter_pretraining()