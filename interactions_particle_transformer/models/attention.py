import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_units, activation='gelu', dropout_rate=0.1, 
                 use_layer_norm=False, output_dim=None):
        super().__init__()
        self.layers = nn.ModuleList()
        prev_dim = input_dim
        
        for units in hidden_units:
            self.layers.append(nn.Linear(prev_dim, units))
            if use_layer_norm:
                self.layers.append(nn.LayerNorm(units))
            if activation == 'gelu':
                self.layers.append(nn.GELU())
            else:
                self.layers.append(nn.ReLU())
            self.layers.append(nn.Dropout(dropout_rate))
            prev_dim = units
            
        if output_dim is not None:
            self.layers.append(nn.Linear(prev_dim, output_dim))
    
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

class ParticleAttentionBlock(nn.Module):
    def __init__(self, embed_dim=128, num_heads=8, mlp_ratio=4, 
                 dropout_rate=0.1, pairwise_dim=4, interaction_mode='sum'):
        super().__init__()
        self.interaction_mode = interaction_mode
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        # Self-attention layers
        self.norm1 = nn.LayerNorm(embed_dim)
        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.attn_drop = nn.Dropout(dropout_rate)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.proj_drop = nn.Dropout(dropout_rate)
        
        # MLP
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = MLP(embed_dim, [embed_dim * mlp_ratio, embed_dim], 
                      activation='gelu', dropout_rate=dropout_rate)
        
        # Pairwise interaction processing
        if interaction_mode == 'concat':
            self.pairwise_mlp = MLP(pairwise_dim + 2*embed_dim, [256, 128, num_heads])
        else:
            self.pairwise_mlp = MLP(pairwise_dim, [64, 64, num_heads])
    
    def forward(self, x, u, mask=None):
        batch_size, n_particles = x.shape[0], x.shape[1]
        
        # Process pairwise interactions
        if self.interaction_mode == 'concat':
            x1 = x.unsqueeze(2).expand(-1, -1, n_particles, -1)
            x2 = x.unsqueeze(1).expand(-1, n_particles, -1, -1)
            u_concat = torch.cat([x1, x2, u], dim=-1)
            u_flat = u_concat.reshape(batch_size * n_particles * n_particles, -1)
            pairwise_bias = self.pairwise_mlp(u_flat)
            pairwise_bias = pairwise_bias.view(batch_size, n_particles, n_particles, self.num_heads)
            pairwise_bias = pairwise_bias.permute(0, 3, 1, 2)
        else:
            u_flat = u.reshape(batch_size * n_particles * n_particles, -1)
            pairwise_bias = self.pairwise_mlp(u_flat)
            pairwise_bias = pairwise_bias.view(batch_size, n_particles, n_particles, self.num_heads)
            pairwise_bias = pairwise_bias.permute(0, 3, 1, 2)
        
        # Self-attention with residual
        x_norm = self.norm1(x)
        qkv = self.qkv(x_norm).reshape(batch_size, n_particles, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        attn = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.head_dim))
        attn = attn + pairwise_bias
        
        if mask is not None:
            attn = attn.masked_fill(mask.unsqueeze(1).unsqueeze(2), float('-inf'))
            
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)
        
        x_attn = (attn @ v).transpose(1, 2).reshape(batch_size, n_particles, self.embed_dim)
        x_attn = self.proj(x_attn)
        x_attn = self.proj_drop(x_attn)
        x = x + x_attn
        
        # MLP with residual
        x = x + self.mlp(self.norm2(x))
        return x

class ClassAttentionBlock(nn.Module):
    def __init__(self, embed_dim=128, num_heads=8, mlp_ratio=4, dropout_rate=0.0):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        self.norm1 = nn.LayerNorm(embed_dim)
        self.q = nn.Linear(embed_dim, embed_dim)
        self.kv = nn.Linear(embed_dim, embed_dim * 2)
        self.attn_drop = nn.Dropout(dropout_rate)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.proj_drop = nn.Dropout(dropout_rate)
        
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = MLP(embed_dim, [embed_dim * mlp_ratio, embed_dim], 
                      activation='gelu', dropout_rate=dropout_rate)
    
    def forward(self, cls_token, x, mask=None):
        cls_norm = self.norm1(cls_token)
        x_norm = self.norm1(x)
        
        q = self.q(cls_norm).reshape(1, cls_token.size(0), self.num_heads, self.head_dim)
        q = q.permute(1, 2, 0, 3)
        
        kv = self.kv(x_norm).reshape(x.size(0), x.size(1), 2, self.num_heads, self.head_dim)
        kv = kv.permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]
        
        attn = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.head_dim))
        
        if mask is not None:
            attn = attn.masked_fill(mask.unsqueeze(1).unsqueeze(2), float('-inf'))
            
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)
        
        cls_attn = (attn @ v).transpose(1, 2).reshape(cls_token.size(0), 1, self.embed_dim)
        cls_attn = self.proj(cls_attn)
        cls_attn = self.proj_drop(cls_attn)
        cls_token = cls_token + cls_attn
        
        cls_token = cls_token + self.mlp(self.norm2(cls_token))
        return cls_token