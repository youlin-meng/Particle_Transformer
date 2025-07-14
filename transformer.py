import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_units, dropout_rate, return_to_input_dim=True):
        super().__init__()
        self.layers = nn.ModuleList()
        self.return_to_input_dim = return_to_input_dim
        prev_dim = input_dim
        
        for units in hidden_units:
            self.layers.append(nn.Linear(prev_dim, units))
            self.layers.append(nn.Dropout(dropout_rate))
            prev_dim = units
        
        if return_to_input_dim:
            self.layers.append(nn.Linear(prev_dim, input_dim))
    
    def forward(self, x):
        for i, layer in enumerate(self.layers):
            if isinstance(layer, nn.Linear):
                if i == len(self.layers) - 1 and self.return_to_input_dim:
                    x = layer(x)
                else:
                    x = F.gelu(layer(x))
            else:
                x = layer(x)
        return x

class TransformerEncoderMHSA(nn.Module):
    def __init__(self, d_model, num_heads, mlp_units, dropout_rate, with_cls_token=False):
        super().__init__()
        self.with_cls_token = with_cls_token
        self.self_attention = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            dropout=dropout_rate,
            batch_first=True
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.mlp = MLP(d_model, mlp_units, dropout_rate, True)
        self.dropout = nn.Dropout(dropout_rate)
    
    def forward(self, x, key_padding_mask=None):
        # Self-attention with residual
        x1 = self.norm1(x)
        attn_output, attn_weights = self.self_attention(
            query=x1, key=x1, value=x1,
            key_padding_mask=key_padding_mask,
            need_weights=self.with_cls_token
        )
        x = x + self.dropout(attn_output)
        
        # MLP with residual
        x2 = self.norm2(x)
        mlp_output = self.mlp(x2)
        x = x + self.dropout(mlp_output)
        
        if self.with_cls_token:
            return x, attn_weights
        return x

class PartClassifier(nn.Module):
    def __init__(self, max_particles=13, n_channels=6, embed_dim=64, num_heads=8,
                 num_transformers=3, mlp_units=[128, 64], mlp_head_units=[128, 64],
                 num_classes=1, dropout_rate=0.3):
        super().__init__()
        
        # Input embedding
        self.input_projection = nn.Linear(n_channels, embed_dim)
        self.pos_embedding = nn.Parameter(torch.randn(1, max_particles, embed_dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        
        # Transformer layers
        self.transformers = nn.ModuleList()
        for i in range(num_transformers):
            is_final = (i == num_transformers - 1)
            self.transformers.append(
                TransformerEncoderMHSA(
                    embed_dim, num_heads, mlp_head_units, 
                    dropout_rate, with_cls_token=is_final
                )
            )
        
        # Classifier head
        self.classifier = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, mlp_units[0]),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(mlp_units[0], num_classes)
        )
    
    def create_src_mask(self, x):
        padding_mask = (x == -99.0).all(dim=-1)  # True for padding positions
        src_mask = ~padding_mask  # True for real particles
        return src_mask
    
    def forward(self, x, src_mask=None):
        batch_size = x.size(0)
        
        if src_mask is None:
            src_mask = self.create_src_mask(x)
        else:
            computed_mask = self.create_src_mask(x)
            if not torch.equal(src_mask, computed_mask):
                src_mask = computed_mask
        
        # Convert to boolean
        src_mask = src_mask.bool()
        
        # Create key_padding_mask for attention (True = ignore position)
        key_padding_mask = ~src_mask  # True for padding, False for real particles
  
        x_processed = x.clone()
        x_processed[~src_mask] = 0.0 
        x_embedded = self.input_projection(x_processed) + self.pos_embedding
        
        # Add CLS token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat((cls_tokens, x_embedded), dim=1)
        
        cls_mask = torch.zeros(batch_size, 1, dtype=torch.bool, device=x.device)
        extended_key_padding_mask = torch.cat([cls_mask, key_padding_mask], dim=1)
        
        # Process through transformers
        attn_weights = None
        for i, transformer in enumerate(self.transformers):
            if i == len(self.transformers) - 1:  # Final layer returns attention
                x, attn_weights = transformer(x, extended_key_padding_mask)
            else:
                x = transformer(x, extended_key_padding_mask)
        
        # Classification using CLS token
        cls_output = x[:, 0]  # Extract CLS token
        logits = self.classifier(cls_output)
        
        # Process attention weights - extract attention TO cls FROM particles
        if attn_weights is not None:
            particle_attention = attn_weights[:, 0, 1:]  # [batch, particles]
            particle_attention = particle_attention * src_mask.float()
            
            real_particle_sums = src_mask.float().sum(dim=1, keepdim=True)
            particle_attention = particle_attention / (real_particle_sums + 1e-8)

        else:
            particle_attention = torch.zeros(batch_size, x.size(1)-1, device=x.device)
        
        return logits, particle_attention

def create_part_classifier(max_particles=13, n_channels=6, embed_dim=64, num_heads=8,
                          num_transformers=3, mlp_head_units=[128, 64], mlp_units=[128, 64],
                          num_classes=1, dropout_rate=0.3):
    
    return PartClassifier(
        max_particles=max_particles,
        n_channels=n_channels,
        embed_dim=embed_dim,
        num_heads=num_heads,
        num_transformers=num_transformers,
        mlp_head_units=mlp_head_units,
        mlp_units=mlp_units,
        num_classes=num_classes,
        dropout_rate=dropout_rate
    )