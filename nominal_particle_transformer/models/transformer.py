import torch
import torch.nn as nn
import torch.nn.functional as F

class ParticleTransformer(nn.Module):
    def __init__(self, max_particles=23, n_channels=6, embed_dim=64, num_heads=8,
                 num_transformers=6, mlp_units=[256, 64], mlp_head_units=[256, 128],
                 num_classes=1, dropout_rate=0.3):
        super().__init__()
        
        self.input_projection = nn.Linear(n_channels, embed_dim)
        self.pad_embedding = nn.Parameter(torch.randn(1, 1, embed_dim, dtype=torch.float32) * 0.02)
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        self.input_norm = nn.LayerNorm(embed_dim)
        
        self.transformers = nn.ModuleList([
            TransformerEncoderMHSA(
                embed_dim, num_heads, mlp_head_units, 
                dropout_rate, with_cls_token=(i == num_transformers - 1)
            )
            for i in range(num_transformers)
        ])
        
        self.classifier = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, mlp_units[0]),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(mlp_units[0], num_classes)
        )

    def create_src_mask(self, x):
        return ~(x == -99.0).all(dim=-1)
    
    def forward(self, x, src_mask=None):
        batch_size = x.size(0)
        
        if src_mask is None:
            src_mask = self.create_src_mask(x)
        src_mask = src_mask.bool()
        key_padding_mask = ~src_mask
        
        x_embedded = self.input_projection(x)
        pad_embeddings = self.pad_embedding.expand_as(x_embedded).to(x_embedded.dtype)
        x_embedded[~src_mask] = pad_embeddings[~src_mask]
        x_embedded = self.input_norm(x_embedded)
        
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat((cls_tokens, x_embedded), dim=1)
        
        cls_mask = torch.zeros(batch_size, 1, dtype=torch.bool, device=x.device)
        extended_key_padding_mask = torch.cat([cls_mask, key_padding_mask], dim=1)
        
        attn_weights = None
        for i, transformer in enumerate(self.transformers):
                if i == len(self.transformers) - 1:
                    x, attn_weights = transformer(x, extended_key_padding_mask)
                else:
                    x = transformer(x, extended_key_padding_mask)
        
        cls_output = x[:, 0]
        logits = self.classifier(cls_output)
        
        if attn_weights is not None:
            full_attention = attn_weights  # [batch, heads, seq_len, seq_len]
            cls_attention = attn_weights[:, 0, 1:]  # [batch, heads, particles]
            
        return logits, {"full": full_attention, "cls": cls_attention}

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
        x1 = self.norm1(x)
        attn_output, attn_weights = self.self_attention(
            query=x1, key=x1, value=x1,
            key_padding_mask=key_padding_mask,
            need_weights=self.with_cls_token
        )
        x = x + self.dropout(attn_output)
        
        x2 = self.norm2(x)
        mlp_output = self.mlp(x2)
        x = x + self.dropout(mlp_output)
        
        if self.with_cls_token:
            return x, attn_weights
        return x

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