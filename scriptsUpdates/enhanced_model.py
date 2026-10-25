# Enhanced model.py implementing the complete architecture

import torch
import torch.nn as nn
from transformers import Wav2Vec2Model, BertModel
import enhanced_config as config

class CoAttentionModule(nn.Module):
    """Implements the Co-attention block from the architecture diagram."""
    
    def __init__(self, dim):
        super().__init__()
        self.query_proj = nn.Linear(dim, dim)
        self.key_proj = nn.Linear(dim, dim)
        self.value_proj = nn.Linear(dim, dim)
        self.softmax = nn.Softmax(dim=-1)
        self.scale = dim ** -0.5

    def forward(self, query_features, key_value_features, key_padding_mask=None):
        q = self.query_proj(query_features)
        k = self.key_proj(key_value_features)
        v = self.value_proj(key_value_features)

        # Scaled dot-product; unscaled 768-dim scores saturate the softmax
        attention_scores = torch.bmm(q, k.transpose(1, 2)) * self.scale
        if key_padding_mask is not None:
            # Keep queries from attending to padded key positions
            attention_scores = attention_scores.masked_fill(
                key_padding_mask.unsqueeze(1), torch.finfo(attention_scores.dtype).min
            )
        attention_weights = self.softmax(attention_scores)
        output = torch.bmm(attention_weights, v)
        
        return output

class EnhancedFFGModel(nn.Module):
    """
    Enhanced Feature Fusion model implementing the complete architecture
    with prosodic and paralinguistic feature extraction.
    """
    
    def __init__(self):
        super().__init__()
        
        # 1. Pre-trained Feature Extractors
        self.acoustic_model = Wav2Vec2Model.from_pretrained(config.ACOUSTIC_MODEL_NAME)
        self.linguistic_model = BertModel.from_pretrained(config.LINGUISTIC_MODEL_NAME)
        
        # Freeze feature extractors initially
        for param in self.acoustic_model.parameters():
            param.requires_grad = False
        for param in self.linguistic_model.parameters():
            param.requires_grad = False
        
        # 2. Prosodic Feature Vector Network
        self.prosodic_extractor = nn.Sequential(
            nn.Linear(6, 64),  # 6 prosodic features -> 64 dim
            nn.ReLU(),
            nn.Dropout(config.DROPOUT),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Dropout(config.DROPOUT),
            nn.Linear(128, config.ACOUSTIC_DIM)  # Map to acoustic dimension
        )
        
        # 3. Paralinguistic Feature Vector Network  
        self.paralinguistic_extractor = nn.Sequential(
            nn.Linear(4, 32),  # 4 paralinguistic features -> 32 dim
            nn.ReLU(),
            nn.Dropout(config.DROPOUT),
            nn.Linear(32, 64),
            nn.ReLU(), 
            nn.Dropout(config.DROPOUT),
            nn.Linear(64, config.LINGUISTIC_DIM)  # Map to linguistic dimension
        )
        
        # 4. Feature Fusion Module
        self.acoustic_transformer = nn.TransformerEncoderLayer(
            d_model=config.ACOUSTIC_DIM, nhead=8, batch_first=True
        )
        self.linguistic_transformer = nn.TransformerEncoderLayer(
            d_model=config.LINGUISTIC_DIM, nhead=8, batch_first=True
        )
        
        # Multiple Co-attention Modules
        self.co_attention_acoustic = nn.ModuleList([
            CoAttentionModule(config.ACOUSTIC_DIM) for _ in range(config.NUM_ATTENTION_LAYERS)
        ])
        self.co_attention_linguistic = nn.ModuleList([
            CoAttentionModule(config.LINGUISTIC_DIM) for _ in range(config.NUM_ATTENTION_LAYERS)
        ])
        
        # LayerNorms for stability
        self.norm_a = nn.LayerNorm(config.ACOUSTIC_DIM)
        self.norm_l = nn.LayerNorm(config.LINGUISTIC_DIM)
        
        # 5. Feature combination layers
        self.prosodic_acoustic_fusion = nn.Linear(config.ACOUSTIC_DIM * 2, config.ACOUSTIC_DIM)
        self.paralinguistic_text_fusion = nn.Linear(config.LINGUISTIC_DIM * 2, config.LINGUISTIC_DIM)
        
        # 6. Final Classifier
        self.pooling = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Sequential(
            nn.Linear(config.ACOUSTIC_DIM + config.LINGUISTIC_DIM, 512),
            nn.ReLU(),
            nn.Dropout(config.DROPOUT),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(config.DROPOUT),
            nn.Linear(256, 2)  # Binary classification: Control vs Dementia
        )
        self.backbone_finetuned = False
        
    def forward(self, acoustic_input, linguistic_input_ids, linguistic_attention_mask, 
                prosodic_features, paralinguistic_features):
        
        # Extract base features from pre-trained models
        acoustic_features = self.acoustic_model(acoustic_input).last_hidden_state
        linguistic_features = self.linguistic_model(
            input_ids=linguistic_input_ids,
            attention_mask=linguistic_attention_mask
        ).last_hidden_state
        
        # Extract enhanced features
        prosodic_vector = self.prosodic_extractor(prosodic_features)  # (B, 768)
        paralinguistic_vector = self.paralinguistic_extractor(paralinguistic_features)  # (B, 768)
        
        # Add feature dimensions for broadcasting
        prosodic_expanded = prosodic_vector.unsqueeze(1).expand_as(acoustic_features)
        paralinguistic_expanded = paralinguistic_vector.unsqueeze(1).expand_as(linguistic_features)
        
        # Fuse prosodic with acoustic features
        acoustic_enhanced = torch.cat([acoustic_features, prosodic_expanded], dim=-1)
        acoustic_fused = self.prosodic_acoustic_fusion(acoustic_enhanced)
        
        # Fuse paralinguistic with text features  
        linguistic_enhanced = torch.cat([linguistic_features, paralinguistic_expanded], dim=-1)
        linguistic_fused = self.paralinguistic_text_fusion(linguistic_enhanced)
        
        # True where the BERT input is padding
        text_padding_mask = linguistic_attention_mask == 0

        # Initial transformer processing
        Fa = self.acoustic_transformer(acoustic_fused)
        Ft = self.linguistic_transformer(linguistic_fused, src_key_padding_mask=text_padding_mask)

        # Multiple Co-attention deep interaction
        current_Fa = Fa
        current_Ft = Ft

        for i in range(config.NUM_ATTENTION_LAYERS):
            # Co-attention between modalities
            Fat = self.co_attention_acoustic[i](current_Ft, current_Fa)  # Text queries Audio
            Fta = self.co_attention_linguistic[i](current_Fa, current_Ft,
                                                  key_padding_mask=text_padding_mask)  # Audio queries Text

            # Residual connections with normalization
            current_Fa = self.norm_a(current_Fa + Fta)
            current_Ft = self.norm_l(current_Ft + Fat)

        # Global pooling (text is averaged over real tokens only)
        pooled_acoustic = self.pooling(current_Fa.transpose(1, 2)).squeeze(2)
        text_mask = linguistic_attention_mask.unsqueeze(-1).to(current_Ft.dtype)
        pooled_linguistic = (current_Ft * text_mask).sum(dim=1) / text_mask.sum(dim=1).clamp_min(1.0)
        
        # Final classification
        concatenated_features = torch.cat((pooled_acoustic, pooled_linguistic), dim=1)
        logits = self.classifier(concatenated_features)
        
        return logits

    def enable_backbone_finetuning(self, acoustic_layers, linguistic_layers):
        """Unfreeze the last k transformer blocks in each backbone for fine-tuning."""
        if self.backbone_finetuned:
            return
        if acoustic_layers > 0 and hasattr(self.acoustic_model, "encoder"):
            target_layers = self.acoustic_model.encoder.layers[-acoustic_layers:]
            for layer in target_layers:
                for param in layer.parameters():
                    param.requires_grad = True
        if linguistic_layers > 0 and hasattr(self.linguistic_model, "encoder"):
            target_layers = self.linguistic_model.encoder.layer[-linguistic_layers:]
            for layer in target_layers:
                for param in layer.parameters():
                    param.requires_grad = True
        self.backbone_finetuned = True