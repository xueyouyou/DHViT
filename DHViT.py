"""
Created on Tue Mar 30 20:46:21 2021

@author: xuegeeker
@blog: https://github.com/xuegeeker
@email: xuegeeker@163.com
"""

import torch
import numpy as np
from torch import nn, einsum
from einops import rearrange, repeat
from einops.layers.torch import Rearrange

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class CrossAttention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.to_k = nn.Linear(dim, inner_dim , bias=False)
        self.to_v = nn.Linear(dim, inner_dim , bias = False)
        self.to_q = nn.Linear(dim, inner_dim, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x_qkv):
        b, n, _, h = *x_qkv.shape, self.heads

        k = self.to_k(x_qkv)
        k = rearrange(k, 'b n (h d) -> b h n d', h = h)

        v = self.to_v(x_qkv)
        v = rearrange(v, 'b n (h d) -> b h n d', h = h)

        q = self.to_q(x_qkv[:, 0].unsqueeze(1))
        q = rearrange(q, 'b n (h d) -> b h n d', h = h)

        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale

        attn = dots.softmax(dim=-1)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out =  self.to_out(out)
        return out


class MultiScaleTransformerEncoder(nn.Module):

    def __init__(self, small_dim = 128, small_dim_head = 64, large_dim = 128, large_dim_head = 64, 
                 cross_attn_depth = 1, cross_attn_heads = 8, dropout = 0.):
        super().__init__()

        self.cross_attn_layers = nn.ModuleList([])
        for _ in range(cross_attn_depth):
            self.cross_attn_layers.append(nn.ModuleList([
                nn.Linear(small_dim, large_dim),
                nn.Linear(large_dim, small_dim),
                PreNorm(large_dim, CrossAttention(large_dim, heads = cross_attn_heads, dim_head = large_dim_head, dropout = dropout)),
                nn.Linear(large_dim, small_dim),
                nn.Linear(small_dim, large_dim),
                PreNorm(small_dim, CrossAttention(small_dim, heads = cross_attn_heads, dim_head = small_dim_head, dropout = dropout)),
            ]))

    def forward(self, xs, xl):

        for f_sl, g_ls, cross_attn_s, f_ls, g_sl, cross_attn_l in self.cross_attn_layers:
            small_class = xs[:, 0]
            x_small = xs[:, 1:]
            large_class = xl[:, 0]
            x_large = xl[:, 1:]

            cal_q = f_ls(large_class.unsqueeze(1))
            cal_qkv = torch.cat((cal_q, x_small), dim=1)
            cal_out = cal_q + cross_attn_l(cal_qkv)
            cal_out = g_sl(cal_out)
            xl = torch.cat((cal_out, x_large), dim=1)

            cal_q = f_sl(small_class.unsqueeze(1))
            cal_qkv = torch.cat((cal_q, x_large), dim=1)
            cal_out = cal_q + cross_attn_s(cal_qkv)
            cal_out = g_ls(cal_out)
            xs = torch.cat((cal_out, x_small), dim=1)

        return xs, xl


class DHViT(nn.Module):
    
    def __init__(self, spec_model, spat_model, lidar_model, dim, num_classes,multi_scale_enc_depth = 1,pool = 'cls', small_dim = 128,
                 large_dim = 128, num_classes1=128, num_classes2=128):

        super().__init__()
        self.spec_model = spec_model
        self.spat_model = spat_model
        self.lidar_model = lidar_model
        
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'
        
        self.multi_scale_transformers_stage1 = nn.ModuleList([])
        for _ in range(multi_scale_enc_depth):
            self.multi_scale_transformers_stage1.append(MultiScaleTransformerEncoder())
            
        self.multi_scale_transformers_stage2 = nn.ModuleList([])
        for _ in range(multi_scale_enc_depth):
            self.multi_scale_transformers_stage2.append(MultiScaleTransformerEncoder())

        self.pool = pool
        self.to_latent = nn.Identity()

        self.mlp_head_small = nn.Sequential(
            nn.LayerNorm(small_dim),
            nn.Linear(small_dim, num_classes1)
        )

        self.mlp_head_large = nn.Sequential(
            nn.LayerNorm(large_dim),
            nn.Linear(large_dim, num_classes2)
        )
        
        dim_new =  dim
        
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim_new),
            nn.Linear(dim_new, dim_new),
            nn.GELU(),
            nn.Linear(dim_new, num_classes),
        )

    def forward(self, x1, x2, x3):
        x1 = self.spec_model(x1) 
        x2 = self.spat_model(x2) 
        x3 = self.lidar_model(x3) 
        
        for multi_scale_transformer in self.multi_scale_transformers_stage1:
            x1, x2 = multi_scale_transformer(x1, x2)
        
        for multi_scale_transformer in self.multi_scale_transformers_stage1:
            x2, x3 = multi_scale_transformer(x2, x3)
            
        for multi_scale_transformer in self.multi_scale_transformers_stage1:
            x3, x1 = multi_scale_transformer(x3, x1)
            
            
        for multi_scale_transformer in self.multi_scale_transformers_stage2:
            x1, x3 = multi_scale_transformer(x1, x3)
        
        for multi_scale_transformer in self.multi_scale_transformers_stage2:
            x2, x1 = multi_scale_transformer(x2, x1)
            
        for multi_scale_transformer in self.multi_scale_transformers_stage2:
            x3, x2 = multi_scale_transformer(x3, x2)
            
        
        x1 = x1.mean(dim = 1) if self.pool == 'mean' else x1[:, 0]
        x2 = x2.mean(dim = 1) if self.pool == 'mean' else x2[:, 0]
        x3 = x3.mean(dim = 1) if self.pool == 'mean' else x3[:, 0]

        x = x1*x2*x3

        return self.mlp_head(x)

