"""
Multimodal Processor for MAHIA-X
Handles cross-modal interaction between text, image, and audio data
"""

import torch
import torch.nn as nn
from typing import Dict, Any, Optional, List, Union
from collections import OrderedDict
import time
from datetime import datetime

# Conditional imports
TORCH_AVAILABLE = False
torch = None
nn = None

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    # Create dummy classes for when torch is not available
    class DummyModule:
        pass
    
    class DummyNN:
        class Module:
            pass
        
        @staticmethod
        def Sequential(*args):
            return None
            
        @staticmethod
        def Linear(*args):
            return None
            
        @staticmethod
        def ReLU():
            return None
    
    nn = DummyNN()

class ModalEncoder(nn.Module):
    """Base encoder for different modalities"""
    
    def __init__(self, modality: str, input_dim: int, embed_dim: int):
        super().__init__()
        self.modality = modality
        self.input_dim = input_dim
        self.embed_dim = embed_dim
        
        # Simple projection layer
        self.projection = nn.Linear(input_dim, embed_dim) if TORCH_AVAILABLE else None
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode input to embedding space"""
        if self.projection is not None:
            return self.projection(x)
        return x

class CrossModalAttention(nn.Module):
    """Cross-modal attention mechanism"""
    
    def __init__(self, embed_dim: int, num_heads: int = 8):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        
        if TORCH_AVAILABLE:
            self.attention = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        else:
            self.attention = None
        
    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
        """Compute cross-modal attention"""
        if self.attention is not None:
            output, _ = self.attention(query, key, value)
            return output
        return query

class MultimodalFusionModule(nn.Module):
    """Fuses information from multiple modalities"""
    
    def __init__(self, embed_dim: int, modalities: List[str]):
        super().__init__()
        self.embed_dim = embed_dim
        self.modalities = modalities
        
        # Create encoders for each modality
        self.encoders = nn.ModuleDict({
            modality: ModalEncoder(modality, embed_dim, embed_dim) 
            for modality in modalities
        })
        
        # Cross-modal attention
        self.cross_attention = CrossModalAttention(embed_dim)
        
        # Fusion layer
        self.fusion_layer = nn.Linear(len(modalities) * embed_dim, embed_dim) if TORCH_AVAILABLE else None
        
    def forward(self, modal_inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Fuse modalities and return combined representation"""
        embeddings = []
        
        # Encode each modality
        for modality, input_tensor in modal_inputs.items():
            if modality in self.encoders:
                embedding = self.encoders[modality](input_tensor)
                embeddings.append(embedding)
                
        if not embeddings:
            # Return zero tensor if no embeddings
            return torch.zeros(1, self.embed_dim) if TORCH_AVAILABLE else None
            
        # Apply cross-modal attention
        if len(embeddings) > 1 and self.cross_attention is not None:
            # Use first modality as query
            query = embeddings[0]
            # Handle dimension matching for key/value
            if len(embeddings) > 1:
                # Concatenate others as key/value, ensuring dimension compatibility
                try:
                    key_value = torch.cat(embeddings[1:], dim=-1) if TORCH_AVAILABLE else None
                    if key_value is not None and query.shape[-1] == key_value.shape[-1]:
                        attended = self.cross_attention(query, key_value, key_value)
                        embeddings[0] = attended
                except Exception as e:
                    # Handle dimension mismatch gracefully
                    pass
                
        # Concatenate all embeddings
        if TORCH_AVAILABLE and embeddings:
            # Ensure all embeddings have the same dimensions before concatenating
            try:
                # Check if we need to reshape or pad embeddings
                target_dim = self.embed_dim
                
                # Process each embedding to ensure compatibility
                processed_embeddings = []
                for emb in embeddings:
                    if emb is not None:
                        # Handle dimension mismatches
                        if len(emb.shape) == 1:
                            # Reshape 1D tensor to 2D
                            emb = emb.unsqueeze(0)
                        if emb.shape[-1] != target_dim:
                            # If dimensions don't match, create a new tensor with correct dimensions
                            if emb.shape[-1] < target_dim:
                                # Pad with zeros
                                padding = torch.zeros((*emb.shape[:-1], target_dim - emb.shape[-1]))
                                emb = torch.cat([emb, padding], dim=-1)
                            else:
                                # Truncate to target dimension
                                emb = emb[..., :target_dim]
                        processed_embeddings.append(emb)
                
                if processed_embeddings:
                    # Now concatenate processed embeddings
                    combined = torch.cat(processed_embeddings, dim=-1)
                    # Apply fusion layer if available
                    if self.fusion_layer is not None and combined.shape[-1] == self.fusion_layer.in_features:
                        return self.fusion_layer(combined)
                    else:
                        # Return combined tensor with proper dimensions
                        return combined[..., :target_dim] if combined.shape[-1] > target_dim else combined
                else:
                    return torch.zeros(1, target_dim) if TORCH_AVAILABLE else None
            except Exception as e:
                # Fallback to simple approach
                return torch.zeros(1, self.embed_dim) if TORCH_AVAILABLE else None
        else:
            return embeddings[0] if embeddings else None

class MultimodalProcessor:
    """Main processor for multimodal interactions"""
    
    def __init__(self, embed_dim: int = 768):
        self.embed_dim = embed_dim
        self.modalities = ['text', 'image', 'audio']
        # Only create fusion module if torch is available
        self.fusion_module = MultimodalFusionModule(embed_dim, self.modalities) if TORCH_AVAILABLE else None
        self.context_history = OrderedDict()
        self.max_context_length = 10
        
    def process_multimodal_input(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Process multimodal input and return fused representation"""
        try:
            # Convert inputs to tensors
            tensor_inputs = {}
            for modality in self.modalities:
                if modality in inputs:
                    data = inputs[modality]
                    if isinstance(data, list) or isinstance(data, tuple):
                        # Convert list to tensor with proper dimensions
                        if TORCH_AVAILABLE:
                            # Ensure data matches embed_dim
                            if len(data) != self.embed_dim:
                                # Pad or truncate to match embed_dim
                                if len(data) < self.embed_dim:
                                    # Pad with zeros
                                    padded_data = list(data) + [0.0] * (self.embed_dim - len(data))
                                    tensor_inputs[modality] = torch.tensor(padded_data, dtype=torch.float32).unsqueeze(0)
                                else:
                                    # Truncate to embed_dim and ensure 2D tensor
                                    truncated_data = data[:self.embed_dim]
                                    tensor_inputs[modality] = torch.tensor(truncated_data, dtype=torch.float32).unsqueeze(0)
                            else:
                                # Data already matches embed_dim, ensure 2D tensor
                                tensor_inputs[modality] = torch.tensor(data, dtype=torch.float32).unsqueeze(0)
                        else:
                            tensor_inputs[modality] = data
                    else:
                        # Handle tensor inputs
                        if TORCH_AVAILABLE and torch.is_tensor(data):
                            # Ensure proper dimensions
                            if len(data.shape) == 1:
                                data = data.unsqueeze(0)
                            if data.shape[-1] != self.embed_dim:
                                if data.shape[-1] < self.embed_dim:
                                    # Pad with zeros
                                    padding = torch.zeros((data.shape[0], self.embed_dim - data.shape[-1]))
                                    data = torch.cat([data, padding], dim=-1)
                                else:
                                    # Truncate to embed_dim
                                    data = data[:, :self.embed_dim]
                            tensor_inputs[modality] = data
                        else:
                            tensor_inputs[modality] = data
                        
            # Apply fusion if available
            if self.fusion_module is not None and TORCH_AVAILABLE:
                fused_representation = self.fusion_module(tensor_inputs)
                result = {
                    'fused_representation': fused_representation,
                    'modalities_processed': list(tensor_inputs.keys()),
                    'timestamp': datetime.now().isoformat()
                }
            else:
                result = {
                    'fused_representation': None,
                    'modalities_processed': list(tensor_inputs.keys()),
                    'timestamp': datetime.now().isoformat()
                }
                
            # Store in context history
            context_id = str(hash(str(inputs)))
            self.context_history[context_id] = {
                'inputs': inputs,
                'result': result,
                'timestamp': datetime.now().isoformat()
            }
            
            # Maintain context history size
            if len(self.context_history) > self.max_context_length:
                # Remove oldest entry
                oldest_key = next(iter(self.context_history))
                del self.context_history[oldest_key]
                
            return result
            
        except Exception as e:
            print(f"Error processing multimodal input: {e}")
            return {
                'error': str(e),
                'modalities_processed': [],
                'timestamp': datetime.now().isoformat()
            }
            
    def get_context_aware_response(self, current_input: Dict[str, Any], context_id: str = None) -> Dict[str, Any]:
        """Generate context-aware response using conversation history"""
        try:
            # Get relevant context
            context_data = []
            if context_id and context_id in self.context_history:
                context_data.append(self.context_history[context_id])
            elif self.context_history:
                # Use most recent context
                latest_key = list(self.context_history.keys())[-1]
                context_data.append(self.context_history[latest_key])
                
            # Combine current input with context
            combined_input = current_input.copy()
            for ctx in context_data:
                ctx_inputs = ctx.get('inputs', {})
                combined_input.update(ctx_inputs)
                
            # Process combined input
            result = self.process_multimodal_input(combined_input)
            result['context_used'] = len(context_data) > 0
            
            return result
            
        except Exception as e:
            print(f"Error generating context-aware response: {e}")
            return {
                'error': str(e),
                'context_used': False,
                'timestamp': datetime.now().isoformat()
            }

# Example usage
if __name__ == "__main__":
    # Initialize multimodal processor
    processor = MultimodalProcessor()
    
    # Example multimodal input
    inputs = {
        'text': [0.1, 0.2, 0.3, 0.4],  # Simplified text embedding
        'image': [0.5, 0.6, 0.7, 0.8],  # Simplified image features
        'audio': [0.9, 1.0, 1.1, 1.2]   # Simplified audio features
    }
    
    # Process input
    result = processor.process_multimodal_input(inputs)
    print("Multimodal Processing Result:", result)