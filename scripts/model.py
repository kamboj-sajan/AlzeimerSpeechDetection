# scripts/model.py

import torch
import torch.nn as nn
from transformers import Wav2Vec2Model, BertModel
import config

class CoAttentionModule(nn.Module):
    """Implements the Co-attention block from Figure 2(c)."""
    def __init__(self, dim):
        super().__init__()
        self.query_proj = nn.Linear(dim, dim)
        self.key_proj = nn.Linear(dim, dim)
        self.value_proj = nn.Linear(dim, dim)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, query_features, key_value_features):
        # query_features: (Batch, Seq_Q, Dim)
        # key_value_features: (Batch, Seq_KV, Dim)
        
        q = self.query_proj(query_features)
        k = self.key_proj(key_value_features)
        v = self.value_proj(key_value_features)
        
        # Attention scores
        attention_scores = torch.bmm(q, k.transpose(1, 2)) # (B, Seq_Q, Seq_KV)
        
        # Softmax to get weights
        attention_weights = self.softmax(attention_scores)
        
        # Apply weights to values
        output = torch.bmm(attention_weights, v) # (B, Seq_Q, Dim)
        return output

class FFGModel(nn.Module):
    """
    The main Feature Fusion model based on the provided architecture.
    """
    def __init__(self):
        super().__init__()
        # 1. Feature Extractors
        self.acoustic_model = Wav2Vec2Model.from_pretrained(config.ACOUSTIC_MODEL_NAME)
        self.linguistic_model = BertModel.from_pretrained(config.LINGUISTIC_MODEL_NAME)
        
        # Freeze feature extractors initially to save memory and speed up training
        for param in self.acoustic_model.parameters():
            param.requires_grad = False
        for param in self.linguistic_model.parameters():
            param.requires_grad = False
            
        # 2. Feature Fusion Module (as in the diagram)
        # Transformers to align features
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

        # 3. Final Classifier
        self.pooling = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Sequential(
            nn.Linear(config.ACOUSTIC_DIM + config.LINGUISTIC_DIM, 256),
            nn.ReLU(),
            nn.Dropout(config.DROPOUT),
            nn.Linear(256, 2) # Binary classification: Control vs Dementia
        )

    def forward(self, acoustic_input, linguistic_input_ids, linguistic_attention_mask):
        # Get base features
        # Acoustic: (B, Seq_A, 768)
        acoustic_features = self.acoustic_model(acoustic_input).last_hidden_state
        # Linguistic: (B, Seq_L, 768)
        linguistic_features = self.linguistic_model(
            input_ids=linguistic_input_ids,
            attention_mask=linguistic_attention_mask
        ).last_hidden_state

        # Initial alignment
        Fa = self.acoustic_transformer(acoustic_features)
        Ft = self.linguistic_transformer(linguistic_features)

        # Multiple Co-attention deep interaction
        current_Fa = Fa
        current_Ft = Ft
        for i in range(config.NUM_ATTENTION_LAYERS):
            # As per paper: output of one co-attention becomes query for the next
            # while key/value are from the other modality
            Fat = self.co_attention_acoustic[i](current_Ft, current_Fa) # Text queries Audio
            Fta = self.co_attention_linguistic[i](current_Fa, current_Ft) # Audio queries Text

            # Add & Norm (residual connection)
            current_Fa = self.norm_a(current_Fa + Fta)
            current_Ft = self.norm_l(current_Ft + Fat)

        # Pooling to get a fixed-size vector for each modality
        pooled_acoustic = self.pooling(current_Fa.transpose(1, 2)).squeeze(2)
        pooled_linguistic = self.pooling(current_Ft.transpose(1, 2)).squeeze(2)

        # Concatenate and classify
        concatenated_features = torch.cat((pooled_acoustic, pooled_linguistic), dim=1)
        logits = self.classifier(concatenated_features)
        
        return logits
