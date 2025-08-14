import torch
import torch.nn as nn
import torch.nn.functional as F
from models.attention import MLP, ParticleAttentionBlock, ClassAttentionBlock

class ParticleTransformer(nn.Module):
    def __init__(self, max_particles=23, n_channels=6, embed_dim=128, num_heads=8,
                 num_particle_blocks=8, num_class_blocks=2, mlp_ratio=4,
                 dropout_rate=0.1, int_mode='concat', pairwise_dim=4):
        super().__init__()
        
        self.particle_embed = MLP(n_channels, [256, 512, embed_dim], 
                                activation='gelu', dropout_rate=dropout_rate,
                                use_layer_norm=True)
        
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        
        self.particle_blocks = nn.ModuleList([
            ParticleAttentionBlock(embed_dim, num_heads, mlp_ratio, dropout_rate, pairwise_dim, interaction_mode=int_mode)
            for _ in range(num_particle_blocks)
        ])
        
        self.class_blocks = nn.ModuleList([
            ClassAttentionBlock(embed_dim, num_heads, mlp_ratio, dropout_rate=0.0)
            for _ in range(num_class_blocks)
        ])
        
        self.norm = nn.LayerNorm(embed_dim)
        self.head = MLP(embed_dim, [embed_dim, 1], 
                       activation='gelu', dropout_rate=dropout_rate)
    
    def create_src_mask(self, x):
        padding_mask = (x == -99.0).all(dim=-1)
        src_mask = ~padding_mask
        return src_mask
    
    def forward(self, x, u, src_mask=None):
        batch_size = x.size(0)
        
        if src_mask is None:
            src_mask = self.create_src_mask(x)
        src_mask = src_mask.bool()
        key_padding_mask = ~src_mask
        
        x_embedded = self.particle_embed(x)
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        
        for block in self.particle_blocks:
            x_embedded = block(x_embedded, u, key_padding_mask)
        
        for block in self.class_blocks:
            cls_tokens = block(cls_tokens, x_embedded, key_padding_mask)
        
        cls_output = self.norm(cls_tokens.squeeze(1))
        logits = self.head(cls_output)
        
        return logits, None
