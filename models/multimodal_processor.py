"""
Multimodal Processor for MAHIA-X
Implements cross-modal interaction and contextual processing of text, image, and audio
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Any, Optional, List, Tuple, Union
from collections import OrderedDict
import time
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

class MultimodalEncoder(nn.Module):
    """Encoder for processing multiple modalities"""
    
    def __init__(self, text_dim: int = 768, image_dim: int = 2048, audio_dim: int = 1024, 
                 hidden_dim: int = 512, output_dim: int = 768):
        """
        Initialize multimodal encoder
        
        Args:
            text_dim: Text embedding dimension
            image_dim: Image feature dimension
            audio_dim: Audio feature dimension
            hidden_dim: Hidden layer dimension
            output_dim: Output dimension
        """
        super().__init__()
        
        self.text_dim = text_dim
        self.image_dim = image_dim
        self.audio_dim = audio_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        
        # Modality-specific encoders
        self.text_encoder = nn.Linear(text_dim, hidden_dim)
        self.image_encoder = nn.Linear(image_dim, hidden_dim)
        self.audio_encoder = nn.Linear(audio_dim, hidden_dim)
        
        # Cross-modal attention
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=8,
            dropout=0.1,
            batch_first=True
        )
        
        # Fusion layer
        self.fusion_layer = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
            nn.LayerNorm(output_dim)
        )
        
        # Modality weights for dynamic fusion
        self.modality_weights = nn.Parameter(torch.ones(3))
        
        self._init_weights()
        
    def _init_weights(self):
        """Initialize network weights"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0.0)
                
    def forward(self, text_features: Optional[torch.Tensor] = None,
                image_features: Optional[torch.Tensor] = None,
                audio_features: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass through multimodal encoder
        
        Args:
            text_features: Text embedding tensor
            image_features: Image feature tensor
            audio_features: Audio feature tensor
            
        Returns:
            Fused multimodal representation
        """
        if not TORCH_AVAILABLE:
            return torch.tensor([0.0]) if TORCH_AVAILABLE else torch.tensor([0.0])
            
        encoded_features = []
        
        # Process text features
        if text_features is not None:
            text_encoded = self.text_encoder(text_features)
            encoded_features.append(text_encoded)
        else:
            # Create dummy tensor if text not provided
            text_encoded = torch.zeros(text_features.size(0), self.hidden_dim) if text_features is not None else torch.zeros(1, self.hidden_dim)
            encoded_features.append(text_encoded)
            
        # Process image features
        if image_features is not None:
            image_encoded = self.image_encoder(image_features)
            encoded_features.append(image_encoded)
        else:
            # Create dummy tensor if image not provided
            batch_size = text_features.size(0) if text_features is not None else 1
            image_encoded = torch.zeros(batch_size, self.hidden_dim)
            encoded_features.append(image_encoded)
            
        # Process audio features
        if audio_features is not None:
            audio_encoded = self.audio_encoder(audio_features)
            encoded_features.append(audio_encoded)
        else:
            # Create dummy tensor if audio not provided
            batch_size = text_features.size(0) if text_features is not None else 1
            audio_encoded = torch.zeros(batch_size, self.hidden_dim)
            encoded_features.append(audio_encoded)
            
        # Apply cross-modal attention
        if len(encoded_features) > 1:
            # Stack features for attention
            stacked_features = torch.stack(encoded_features, dim=1)  # [batch, num_modalities, hidden_dim]
            
            # Apply self-attention across modalities
            attended_features, _ = self.cross_attention(
                stacked_features, stacked_features, stacked_features
            )
            
            # Weighted combination
            modality_weights = torch.softmax(self.modality_weights, dim=0)
            weighted_features = attended_features * modality_weights.unsqueeze(0).unsqueeze(-1)
            combined_features = torch.sum(weighted_features, dim=1)
        else:
            combined_features = encoded_features[0]
            
        # Fusion
        if len(encoded_features) == 3:
            concatenated = torch.cat(encoded_features, dim=-1)
            fused = self.fusion_layer(concatenated)
        else:
            fused = combined_features
            
        return fused


class ContextMemory:
    """Context memory for maintaining conversation history"""
    
    def __init__(self, max_context_length: int = 10, embedding_dim: int = 768):
        """
        Initialize context memory
        
        Args:
            max_context_length: Maximum number of conversation turns to remember
            embedding_dim: Dimension of context embeddings
        """
        self.max_context_length = max_context_length
        self.embedding_dim = embedding_dim
        self.context_history = OrderedDict()
        self.current_context = []
        
    def add_context(self, context_id: str, context_data: Dict[str, Any]):
        """
        Add context to memory
        
        Args:
            context_id: Unique identifier for context
            context_data: Context data dictionary
        """
        context_entry = {
            "context_id": context_id,
            "data": context_data,
            "timestamp": time.time(),
            "access_count": 0
        }
        
        self.context_history[context_id] = context_entry
        self.current_context.append(context_id)
        
        # Maintain maximum context length
        if len(self.current_context) > self.max_context_length:
            oldest_context = self.current_context.pop(0)
            if oldest_context in self.context_history:
                del self.context_history[oldest_context]
                
    def get_context_vector(self, context_ids: List[str]) -> Optional[torch.Tensor]:
        """
        Get context vector representation
        
        Args:
            context_ids: List of context IDs to include
            
        Returns:
            Context vector tensor or None
        """
        if not TORCH_AVAILABLE:
            return None
            
        context_vectors = []
        
        for context_id in context_ids:
            if context_id in self.context_history:
                context_data = self.context_history[context_id]["data"]
                # Extract embedding if available
                if "embedding" in context_data:
                    context_vectors.append(context_data["embedding"])
                self.context_history[context_id]["access_count"] += 1
                
        if context_vectors:
            # Average context vectors
            stacked = torch.stack(context_vectors)
            return torch.mean(stacked, dim=0)
        else:
            return None
            
    def get_relevant_context(self, query_embedding: torch.Tensor, 
                           top_k: int = 3) -> List[str]:
        """
        Get most relevant context based on query similarity
        
        Args:
            query_embedding: Query embedding tensor
            top_k: Number of top contexts to return
            
        Returns:
            List of relevant context IDs
        """
        if not TORCH_AVAILABLE or query_embedding is None:
            return self.current_context[-top_k:] if len(self.current_context) >= top_k else self.current_context
            
        similarities = []
        
        for context_id in self.current_context:
            if context_id in self.context_history:
                context_data = self.context_history[context_id]["data"]
                if "embedding" in context_data:
                    context_embedding = context_data["embedding"]
                    # Calculate cosine similarity
                    if context_embedding.dim() == query_embedding.dim():
                        similarity = torch.cosine_similarity(
                            query_embedding.unsqueeze(0),
                            context_embedding.unsqueeze(0)
                        ).item()
                        similarities.append((context_id, similarity))
                        
        # Sort by similarity and return top-k
        similarities.sort(key=lambda x: x[1], reverse=True)
        return [ctx_id for ctx_id, _ in similarities[:top_k]]


class MultimodalProcessor:
    """Main multimodal processor for cross-modal interaction"""
    
    def __init__(self, text_dim: int = 768, image_dim: int = 2048, 
                 audio_dim: int = 1024, hidden_dim: int = 512):
        """
        Initialize multimodal processor
        
        Args:
            text_dim: Text embedding dimension
            image_dim: Image feature dimension
            audio_dim: Audio feature dimension
            hidden_dim: Hidden layer dimension
        """
        self.text_dim = text_dim
        self.image_dim = image_dim
        self.audio_dim = audio_dim
        self.hidden_dim = hidden_dim
        
        # Initialize components
        self.multimodal_encoder = MultimodalEncoder(
            text_dim, image_dim, audio_dim, hidden_dim
        )
        self.context_memory = ContextMemory(max_context_length=10, embedding_dim=text_dim)
        
        # Processing statistics
        self.processing_stats = {
            "total_multimodal_inputs": 0,
            "text_inputs": 0,
            "image_inputs": 0,
            "audio_inputs": 0,
            "context_enhanced": 0
        }
        
        print(f"✅ MultimodalProcessor initialized")
        print(f"   Text dim: {text_dim}, Image dim: {image_dim}, Audio dim: {audio_dim}")
        
    def process_multimodal_input(self, 
                               input_id: str,
                               text_input: Optional[str] = None,
                               text_embedding: Optional[torch.Tensor] = None,
                               image_input: Optional[Any] = None,
                               image_features: Optional[torch.Tensor] = None,
                               audio_input: Optional[Any] = None,
                               audio_features: Optional[torch.Tensor] = None,
                               context_ids: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Process multimodal input with context awareness
        
        Args:
            input_id: Unique identifier for input
            text_input: Text input string
            text_embedding: Text embedding tensor
            image_input: Image input data
            image_features: Image feature tensor
            audio_input: Audio input data
            audio_features: Audio feature tensor
            context_ids: List of context IDs to consider
            
        Returns:
            Processing results
        """
        start_time = time.time()
        
        # Update statistics
        self.processing_stats["total_multimodal_inputs"] += 1
        if text_input or text_embedding is not None:
            self.processing_stats["text_inputs"] += 1
        if image_input or image_features is not None:
            self.processing_stats["image_inputs"] += 1
        if audio_input or audio_features is not None:
            self.processing_stats["audio_inputs"] += 1
            
        # Get context vector if context IDs provided
        context_vector = None
        if context_ids:
            context_vector = self.context_memory.get_context_vector(context_ids)
            if context_vector is not None:
                self.processing_stats["context_enhanced"] += 1
                
        # Process multimodal input
        fused_representation = self.multimodal_encoder(
            text_features=text_embedding,
            image_features=image_features,
            audio_features=audio_features
        )
        
        # Incorporate context if available
        if context_vector is not None and TORCH_AVAILABLE:
            # Simple concatenation and projection
            if fused_representation.dim() == context_vector.dim():
                combined = torch.cat([fused_representation, context_vector], dim=-1)
                # Project back to original dimension
                projector = nn.Linear(combined.size(-1), self.text_dim)
                if TORCH_AVAILABLE:
                    enhanced_representation = projector(combined)
                else:
                    enhanced_representation = fused_representation
            else:
                enhanced_representation = fused_representation
        else:
            enhanced_representation = fused_representation
            
        processing_time = time.time() - start_time
        
        # Store input in context memory
        context_data = {
            "input_id": input_id,
            "text_input": text_input,
            "embedding": enhanced_representation,
            "modalities": {
                "text": text_input is not None or text_embedding is not None,
                "image": image_input is not None or image_features is not None,
                "audio": audio_input is not None or audio_features is not None
            },
            "processing_time": processing_time
        }
        
        self.context_memory.add_context(input_id, context_data)
        
        return {
            "input_id": input_id,
            "representation": enhanced_representation,
            "modalities_processed": context_data["modalities"],
            "context_enhanced": context_vector is not None,
            "processing_time": processing_time,
            "timestamp": time.time()
        }
        
    def get_relevant_context(self, query_embedding: torch.Tensor, 
                           top_k: int = 3) -> List[Dict[str, Any]]:
        """
        Get relevant context for a query
        
        Args:
            query_embedding: Query embedding tensor
            top_k: Number of top contexts to return
            
        Returns:
            List of relevant context entries
        """
        relevant_context_ids = self.context_memory.get_relevant_context(
            query_embedding, top_k
        )
        
        relevant_contexts = []
        for context_id in relevant_context_ids:
            if context_id in self.context_memory.context_history:
                context_entry = self.context_memory.context_history[context_id]
                relevant_contexts.append({
                    "context_id": context_id,
                    "data": context_entry["data"],
                    "relevance_score": context_entry["access_count"],  # Simplified relevance
                    "timestamp": context_entry["timestamp"]
                })
                
        return relevant_contexts
        
    def get_processing_stats(self) -> Dict[str, Any]:
        """
        Get processing statistics
        
        Returns:
            Dictionary of statistics
        """
        return {
            "timestamp": time.time(),
            "stats": self.processing_stats,
            "context_memory_size": len(self.context_memory.context_history),
            "current_context_length": len(self.context_memory.current_context),
            "encoder_parameters": sum(p.numel() for p in self.multimodal_encoder.parameters()) if TORCH_AVAILABLE else 0
        }
        
    def export_multimodal_report(self, filepath: str) -> bool:
        """
        Export multimodal processing report to file
        
        Args:
            filepath: Path to export report
            
        Returns:
            True if successful, False otherwise
        """
        try:
            report = {
                "generated_at": datetime.now().isoformat(),
                "processing_stats": self.get_processing_stats(),
                "context_memory_summary": {
                    "total_contexts": len(self.context_memory.context_history),
                    "current_context_ids": list(self.context_memory.current_context)
                }
            }
            
            import json
            with open(filepath, 'w') as f:
                json.dump(report, f, indent=2, default=str)
                
            print(f"✅ Multimodal processing report exported to {filepath}")
            return True
        except Exception as e:
            print(f"❌ Failed to export multimodal report: {e}")
            return False


def demo_multimodal_processor():
    """Demonstrate multimodal processor functionality"""
    print("🚀 Demonstrating Multimodal Processor...")
    print("=" * 50)
    
    # Create multimodal processor
    processor = MultimodalProcessor(
        text_dim=768,
        image_dim=2048,
        audio_dim=1024,
        hidden_dim=512
    )
    print("✅ Created multimodal processor")
    
    # Simulate multimodal inputs
    print("\n🔄 Processing multimodal inputs...")
    
    if TORCH_AVAILABLE:
        # Create dummy embeddings
        text_embedding = torch.randn(1, 768)
        image_features = torch.randn(1, 2048)
        audio_features = torch.randn(1, 1024)
        
        # Process text-only input
        result1 = processor.process_multimodal_input(
            input_id="input_001",
            text_embedding=text_embedding,
            context_ids=[]
        )
        print(f"   Text-only input processed in {result1['processing_time']:.3f}s")
        
        # Process multimodal input
        result2 = processor.process_multimodal_input(
            input_id="input_002",
            text_embedding=text_embedding,
            image_features=image_features,
            audio_features=audio_features,
            context_ids=["input_001"]
        )
        print(f"   Multimodal input processed in {result2['processing_time']:.3f}s")
        print(f"   Context enhanced: {result2['context_enhanced']}")
        
        # Process image-only input
        result3 = processor.process_multimodal_input(
            input_id="input_003",
            image_features=image_features,
            context_ids=["input_001", "input_002"]
        )
        print(f"   Image-only input processed in {result3['processing_time']:.3f}s")
        
        # Get relevant context
        print("\n🔍 Retrieving relevant context...")
        relevant_context = processor.get_relevant_context(text_embedding, top_k=2)
        print(f"   Found {len(relevant_context)} relevant contexts")
        
    else:
        print("❌ PyTorch not available, skipping demonstration")
        
    # Show statistics
    print("\n📊 Processing Statistics:")
    stats = processor.get_processing_stats()
    print(f"   Total inputs processed: {stats['stats']['total_multimodal_inputs']}")
    print(f"   Text inputs: {stats['stats']['text_inputs']}")
    print(f"   Image inputs: {stats['stats']['image_inputs']}")
    print(f"   Audio inputs: {stats['stats']['audio_inputs']}")
    print(f"   Context enhanced: {stats['stats']['context_enhanced']}")
    
    # Export report
    report_success = processor.export_multimodal_report("multimodal_report.json")
    print(f"   Report export: {'SUCCESS' if report_success else 'FAILED'}")
    
    print("\n" + "=" * 50)
    print("MULTIMODAL PROCESSOR DEMO SUMMARY")
    print("=" * 50)
    print("Key Features Implemented:")
    print("  1. Cross-modal feature encoding")
    print("  2. Attention-based modality fusion")
    print("  3. Context-aware processing")
    print("  4. Dynamic context retrieval")
    print("  5. Comprehensive statistics tracking")
    print("\nBenefits:")
    print("  - Unified processing of text, image, and audio")
    print("  - Context-preserving conversation flow")
    print("  - Adaptive modality weighting")
    print("  - Efficient multimodal representation")
    
    print("\n✅ Multimodal processor demonstration completed!")


if __name__ == "__main__":
    demo_multimodal_processor()