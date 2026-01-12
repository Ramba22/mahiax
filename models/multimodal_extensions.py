"""
Multimodal Extensions for MAHIA-X: Audio/Speech/CLIP Pretraining.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import sys
import os
from typing import Dict, Any, Tuple, Optional, List
import math

# Add the project root to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

try:
    from modell_V5_MAHIA_HyenaMoE import MAHIA_V5, SparseMoETopK
    MULTIMODAL_AVAILABLE = True
except ImportError:
    MULTIMODAL_AVAILABLE = False
    print("⚠️  MAHIA-X modules not available for multimodal extensions")


class AudioFeatureExtractor(nn.Module):
    """Audio feature extractor for speech processing"""
    
    def __init__(self, input_channels: int = 1, hidden_dim: int = 64, 
                 output_dim: int = 128, kernel_sizes: List[int] = [3, 5, 7]):
        super().__init__()
        self.input_channels = input_channels
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        
        # Multi-scale convolutional layers
        self.conv_layers = nn.ModuleList([
            nn.Conv1d(input_channels, hidden_dim, kernel_size=ks, padding=ks//2)
            for ks in kernel_sizes
        ])
        
        # Batch normalization
        self.batch_norm = nn.BatchNorm1d(hidden_dim * len(kernel_sizes))
        
        # Temporal pooling
        self.temporal_pool = nn.AdaptiveAvgPool1d(output_size=32)
        
        # Feature projection
        self.projection = nn.Sequential(
            nn.Linear(hidden_dim * len(kernel_sizes) * 32, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim * 2, output_dim)
        )
        
    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        """Forward pass
        Args:
            audio: Audio tensor of shape (B, C, T) where T is time steps
        Returns:
            Extracted features of shape (B, D)
        """
        B, C, T = audio.shape
        
        # Apply multi-scale convolutions
        conv_outputs = []
        for conv in self.conv_layers:
            conv_out = conv(audio)  # (B, H, T)
            conv_outputs.append(conv_out)
            
        # Concatenate multi-scale features
        multi_scale = torch.cat(conv_outputs, dim=1)  # (B, H*len, T)
        
        # Batch normalization
        multi_scale = self.batch_norm(multi_scale)  # (B, H*len, T)
        
        # Temporal pooling
        pooled = self.temporal_pool(multi_scale)  # (B, H*len, 32)
        
        # Flatten and project
        flattened = pooled.view(B, -1)  # (B, H*len*32)
        features = self.projection(flattened)  # (B, D)
        
        return features


class SpeechEncoder(nn.Module):
    """Speech encoder for processing audio features"""
    
    def __init__(self, input_dim: int = 128, hidden_dim: int = 256, 
                 output_dim: int = 256, num_layers: int = 2):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        
        # LSTM for temporal modeling
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=num_layers, 
                           batch_first=True, bidirectional=True, dropout=0.1)
        
        # Attention mechanism
        self.attention = nn.MultiheadAttention(hidden_dim * 2, num_heads=8, 
                                             batch_first=True, dropout=0.1)
        
        # Feature projection
        self.projection = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, output_dim)
        )
        
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Forward pass
        Args:
            features: Audio features of shape (B, D) or (B, T, D)
        Returns:
            Encoded features of shape (B, D)
        """
        B = features.shape[0]
        
        # Handle different input shapes
        if features.dim() == 2:
            # Add time dimension
            features = features.unsqueeze(1)  # (B, 1, D)
            
        # LSTM encoding
        lstm_out, _ = self.lstm(features)  # (B, T, H*2)
        
        # Attention mechanism
        attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)  # (B, T, H*2)
        
        # Global average pooling
        pooled = torch.mean(attn_out, dim=1)  # (B, H*2)
        
        # Projection
        encoded = self.projection(pooled)  # (B, D)
        
        return encoded


class CLIPEmbeddingProjector(nn.Module):
    """Project CLIP embeddings to MAHIA-X feature space"""
    
    def __init__(self, clip_dim: int = 512, mahia_dim: int = 64, 
                 hidden_dim: int = 256):
        super().__init__()
        self.clip_dim = clip_dim
        self.mahia_dim = mahia_dim
        
        # Projection network
        self.projector = nn.Sequential(
            nn.Linear(clip_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, mahia_dim)
        )
        
    def forward(self, clip_embeddings: torch.Tensor) -> torch.Tensor:
        """Forward pass
        Args:
            clip_embeddings: CLIP embeddings of shape (B, clip_dim)
        Returns:
            Projected features of shape (B, mahia_dim)
        """
        return self.projector(clip_embeddings)


class MultimodalMAHIA(nn.Module):
    """Multimodal MAHIA-X with audio/speech/CLIP support"""
    
    def __init__(self, vocab_size: int = 10000, text_seq_len: int = 64, 
                 tab_dim: int = 50, embed_dim: int = 64, fused_dim: int = 128,
                 moe_experts: int = 8, moe_topk: int = 2,
                 audio_input_channels: int = 1, audio_output_dim: int = 128,
                 clip_dim: int = 512):
        super().__init__()
        
        # Text processing (from original MAHIA)
        self.text_encoder = nn.Embedding(vocab_size, embed_dim)
        self.text_pos_emb = nn.Parameter(torch.randn(1, text_seq_len, embed_dim))
        self.text_layer = nn.TransformerEncoderLayer(embed_dim, nhead=8, dim_feedforward=embed_dim*2, dropout=0.1, batch_first=True)
        self.text_transformer = nn.TransformerEncoder(self.text_layer, num_layers=2)
        
        # Tabular processing (from original MAHIA)
        self.tab_encoder = nn.Sequential(
            nn.Linear(tab_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim)
        )
        
        # Audio processing
        self.audio_extractor = AudioFeatureExtractor(
            input_channels=audio_input_channels,
            hidden_dim=64,
            output_dim=audio_output_dim
        )
        self.speech_encoder = SpeechEncoder(
            input_dim=audio_output_dim,
            hidden_dim=128,
            output_dim=embed_dim
        )
        
        # CLIP embedding projector
        self.clip_projector = CLIPEmbeddingProjector(
            clip_dim=clip_dim,
            mahia_dim=embed_dim
        )
        
        # Multimodal fusion
        self.multimodal_fusion = nn.Sequential(
            nn.Linear(embed_dim * 3, fused_dim),  # Text + Tabular + Audio/CLIP
            nn.LayerNorm(fused_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(fused_dim, fused_dim)
        )
        
        # MoE processing (from original MAHIA)
        if MULTIMODAL_AVAILABLE:
            self.moe = SparseMoETopK(dim=fused_dim, num_experts=moe_experts, top_k=moe_topk)
        else:
            # Fallback implementation
            self.moe = nn.Linear(fused_dim, fused_dim)
        
        # Final fusion and classification
        self.final_fusion = nn.Sequential(
            nn.Linear(fused_dim, fused_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(fused_dim // 2, 2)  # Binary classification
        )
        
        # Modality weights for dynamic fusion
        self.modality_weights = nn.Parameter(torch.ones(3))  # Text, Tabular, Audio/CLIP
        
    def forward(self, text_tokens: Optional[torch.LongTensor] = None,
                tab_feats: Optional[torch.Tensor] = None,
                audio: Optional[torch.Tensor] = None,
                clip_embeddings: Optional[torch.Tensor] = None,
                return_aux: bool = True) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Forward pass with multimodal support
        Args:
            text_tokens: Text tokens of shape (B, L)
            tab_feats: Tabular features of shape (B, D_tab)
            audio: Audio data of shape (B, C, T)
            clip_embeddings: CLIP embeddings of shape (B, clip_dim)
            return_aux: Whether to return auxiliary loss
        Returns:
            Tuple of (logits, aux_loss)
        """
        B = None
        text_encoded = None
        tab_encoded = None
        audio_encoded = None
        
        # Process text if provided
        if text_tokens is not None:
            B = text_tokens.shape[0]
            text_emb = self.text_encoder(text_tokens)  # (B, L, D)
            text_emb = text_emb + self.text_pos_emb[:, :text_emb.size(1), :]
            text_encoded = self.text_transformer(text_emb)  # (B, L, D)
            text_encoded = text_encoded.mean(dim=1)  # (B, D)
            
        # Process tabular if provided
        if tab_feats is not None:
            if B is None:
                B = tab_feats.shape[0]
            tab_encoded = self.tab_encoder(tab_feats)  # (B, D)
            
        # Process audio if provided
        if audio is not None:
            if B is None:
                B = audio.shape[0]
            audio_features = self.audio_extractor(audio)  # (B, D_audio)
            audio_encoded = self.speech_encoder(audio_features)  # (B, D)
            
        # Process CLIP embeddings if provided
        if clip_embeddings is not None:
            if B is None:
                B = clip_embeddings.shape[0]
            if audio_encoded is None:  # Only process CLIP if audio not provided
                audio_encoded = self.clip_projector(clip_embeddings)  # (B, D)
                
        # Ensure we have batch size
        if B is None:
            B = 1
            
        # Use zero tensors for missing modalities
        if text_encoded is None:
            text_encoded = torch.zeros(B, self.text_encoder.embedding_dim, 
                                     device=next(self.parameters()).device)
        if tab_encoded is None:
            tab_encoded = torch.zeros(B, self.tab_encoder[0].in_features, 
                                    device=next(self.parameters()).device)
            tab_encoded = self.tab_encoder(tab_encoded)
        if audio_encoded is None:
            audio_encoded = torch.zeros(B, self.text_encoder.embedding_dim, 
                                      device=next(self.parameters()).device)
            
        # Apply modality weights
        weights = F.softmax(self.modality_weights, dim=0)
        text_weight, tab_weight, audio_weight = weights
        
        # Weighted fusion
        fused = self.multimodal_fusion(
            torch.cat([
                text_encoded * text_weight,
                tab_encoded * tab_weight,
                audio_encoded * audio_weight
            ], dim=-1)
        )  # (B, fused_dim)
        
        # Apply MoE
        fused_expanded = fused.unsqueeze(1)  # (B, 1, fused_dim)
        moe_out, aux_loss = self.moe(fused_expanded, return_aux=return_aux)
        moe_pooled = moe_out.squeeze(1)  # (B, fused_dim)
        
        # Final classification
        logits = self.final_fusion(moe_pooled)  # (B, 2)
        
        return logits, aux_loss


class MultimodalPretrainer:
    """Pretrainer for multimodal MAHIA-X"""
    
    def __init__(self, model: MultimodalMAHIA):
        self.model = model
        
        # Pretraining objectives
        self.mlm_head = nn.Linear(model.text_encoder.embedding_dim, model.text_encoder.num_embeddings)
        self.audio_reconstruction = nn.Linear(model.speech_encoder.output_dim, 128)  # Simplified
        self.contrastive_temperature = nn.Parameter(torch.ones(1))
        
    def masked_language_modeling_loss(self, text_tokens: torch.Tensor, 
                                    mask_positions: torch.Tensor) -> torch.Tensor:
        """Compute masked language modeling loss
        Args:
            text_tokens: Text tokens with some masked
            mask_positions: Positions of masked tokens
        Returns:
            MLM loss
        """
        # Forward pass
        logits, _ = self.model(text_tokens=text_tokens)
        
        # Get predictions for masked positions
        # This is simplified - in practice, you'd need to extract the right representations
        predictions = self.mlm_head(logits)  # (B, V)
        
        # Compute loss (simplified)
        targets = text_tokens[mask_positions]  # (B, num_masked)
        loss = F.cross_entropy(predictions, targets)
        
        return loss
    
    def contrastive_loss(self, features_a: torch.Tensor, 
                        features_b: torch.Tensor) -> torch.Tensor:
        """Compute contrastive loss between two modalities
        Args:
            features_a: Features from modality A
            features_b: Features from modality B
        Returns:
            Contrastive loss
        """
        # Normalize features
        features_a = F.normalize(features_a, dim=-1)
        features_b = F.normalize(features_b, dim=-1)
        
        # Compute similarity matrix
        similarity = torch.matmul(features_a, features_b.T) / self.contrastive_temperature
        
        # Labels for positive pairs
        labels = torch.arange(similarity.shape[0], device=similarity.device)
        
        # Compute loss
        loss_a = F.cross_entropy(similarity, labels)
        loss_b = F.cross_entropy(similarity.T, labels)
        
        return (loss_a + loss_b) / 2
    
    def audio_reconstruction_loss(self, audio: torch.Tensor) -> torch.Tensor:
        """Compute audio reconstruction loss
        Args:
            audio: Original audio
        Returns:
            Reconstruction loss
        """
        # Forward pass
        logits, _ = self.model(audio=audio)
        
        # Reconstruction (simplified)
        reconstructed = self.audio_reconstruction(logits)
        target = audio.mean(dim=-1)  # Simplified target
        
        loss = F.mse_loss(reconstructed, target)
        return loss


def demo_multimodal_extensions():
    """Demonstrate multimodal extensions"""
    if not MULTIMODAL_AVAILABLE:
        print("❌ MAHIA-X modules not available for multimodal extensions")
        # Continue with demo using mock components
        
    print("🚀 Demonstrating Multimodal Extensions (Audio/Speech/CLIP)...")
    print("=" * 60)
    
    # Create multimodal MAHIA model
    multimodal_model = MultimodalMAHIA(
        vocab_size=1000,
        text_seq_len=32,
        tab_dim=20,
        embed_dim=64,
        fused_dim=128,
        moe_experts=4,
        moe_topk=2,
        audio_input_channels=1,
        audio_output_dim=128,
        clip_dim=512
    )
    print("✅ Initialized Multimodal MAHIA Model")
    print(f"   Model parameters: {sum(p.numel() for p in multimodal_model.parameters()):,}")
    
    # Test text-only input
    batch_size = 2
    text_tokens = torch.randint(0, 1000, (batch_size, 32))
    logits, aux_loss = multimodal_model(text_tokens=text_tokens)
    print(f"✅ Text-only forward pass:")
    print(f"   Text tokens: {text_tokens.shape}")
    print(f"   Logits: {logits.shape}")
    if aux_loss is not None:
        print(f"   Aux loss: {aux_loss.item():.6f}")
    
    # Test tabular-only input
    tab_feats = torch.randn(batch_size, 20)
    logits, aux_loss = multimodal_model(tab_feats=tab_feats)
    print(f"✅ Tabular-only forward pass:")
    print(f"   Tabular features: {tab_feats.shape}")
    print(f"   Logits: {logits.shape}")
    
    # Test audio input
    audio = torch.randn(batch_size, 1, 16000)  # 16kHz audio
    logits, aux_loss = multimodal_model(audio=audio)
    print(f"✅ Audio forward pass:")
    print(f"   Audio: {audio.shape}")
    print(f"   Logits: {logits.shape}")
    
    # Test CLIP embeddings
    clip_embeddings = torch.randn(batch_size, 512)
    logits, aux_loss = multimodal_model(clip_embeddings=clip_embeddings)
    print(f"✅ CLIP embeddings forward pass:")
    print(f"   CLIP embeddings: {clip_embeddings.shape}")
    print(f"   Logits: {logits.shape}")
    
    # Test multimodal input
    logits, aux_loss = multimodal_model(
        text_tokens=text_tokens,
        tab_feats=tab_feats,
        audio=audio
    )
    print(f"✅ Multimodal forward pass:")
    print(f"   Combined logits: {logits.shape}")
    
    # Test modality weights
    modality_weights = F.softmax(multimodal_model.modality_weights, dim=0)
    print(f"✅ Modality weights:")
    print(f"   Text weight: {modality_weights[0].item():.3f}")
    print(f"   Tabular weight: {modality_weights[1].item():.3f}")
    print(f"   Audio/CLIP weight: {modality_weights[2].item():.3f}")
    
    # Create pretrainer
    pretrainer = MultimodalPretrainer(multimodal_model)
    print("✅ Initialized Multimodal Pretrainer")
    
    # Test MLM loss (simplified)
    mask_positions = torch.randint(0, 32, (batch_size, 5))  # 5 masked positions
    mlm_loss = pretrainer.masked_language_modeling_loss(text_tokens, mask_positions)
    print(f"✅ MLM loss: {mlm_loss.item():.6f}")
    
    # Test contrastive loss
    features_a = torch.randn(batch_size, 64)
    features_b = torch.randn(batch_size, 64)
    contrastive_loss = pretrainer.contrastive_loss(features_a, features_b)
    print(f"✅ Contrastive loss: {contrastive_loss.item():.6f}")
    
    # Test audio reconstruction loss
    audio_recon_loss = pretrainer.audio_reconstruction_loss(audio)
    print(f"✅ Audio reconstruction loss: {audio_recon_loss.item():.6f}")
    
    # Create individual components
    audio_extractor = AudioFeatureExtractor(input_channels=1, output_dim=128)
    speech_encoder = SpeechEncoder(input_dim=128, output_dim=64)
    clip_projector = CLIPEmbeddingProjector(clip_dim=512, mahia_dim=64)
    
    print(f"✅ Individual components created:")
    print(f"   Audio extractor parameters: {sum(p.numel() for p in audio_extractor.parameters()):,}")
    print(f"   Speech encoder parameters: {sum(p.numel() for p in speech_encoder.parameters()):,}")
    print(f"   CLIP projector parameters: {sum(p.numel() for p in clip_projector.parameters()):,}")
    
    # Test audio processing pipeline
    audio_features = audio_extractor(audio)
    audio_encoded = speech_encoder(audio_features)
    print(f"✅ Audio processing pipeline:")
    print(f"   Raw audio: {audio.shape}")
    print(f"   Extracted features: {audio_features.shape}")
    print(f"   Encoded features: {audio_encoded.shape}")
    
    # Test CLIP projection
    clip_projected = clip_projector(clip_embeddings)
    print(f"✅ CLIP projection:")
    print(f"   Original embeddings: {clip_embeddings.shape}")
    print(f"   Projected features: {clip_projected.shape}")
    
    print("\n" + "=" * 60)
    print("MULTIMODAL EXTENSIONS DEMO SUMMARY")
    print("=" * 60)
    print("Key Features Implemented:")
    print("  1. Audio feature extraction with multi-scale convolutions")
    print("  2. Speech encoding with LSTM and attention")
    print("  3. CLIP embedding projection to MAHIA space")
    print("  4. Multimodal fusion with dynamic weighting")
    print("  5. Pretraining objectives (MLM, contrastive, reconstruction)")
    print("\nBenefits:")
    print("  - Rich multimodal representations")
    print("  - Flexible input combinations")
    print("  - Pretraining capabilities")
    print("  - Dynamic modality importance")
    
    print("\n✅ Multimodal Extensions demonstration completed!")


def main():
    """Main demonstration function"""
    demo_multimodal_extensions()


if __name__ == '__main__':
    main()