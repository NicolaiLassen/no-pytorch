'''
    @author: Shuhao Cao
    https://github.com/scaomath/galerkin-transformer/blob/main/libs/layers.py
    @author: Zijie Li
    https://arxiv.org/pdf/2205.13671.pdf
'''

import torch
import torch.nn as nn
from torch.nn.init import xavier_normal_
from .fno import FNO1d, FNO2d, FNO3d
from .attention import FourierAttention, GalerkinAttention
from .pos import RoPE, Grid
from .functional import default

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class FeedForward(nn.Module):
    def __init__(self, 
                 dim=256,
                 hidden_dim: int = 1024,
                 out_dim=None,
                 batch_norm=False,
                 activation='relu',
                 dropout=0.1):
        super(FeedForward, self).__init__()
        out_dim = default(out_dim, dim)

        self.fc_in = nn.Linear(dim, hidden_dim)

        if activation == 'silu':
            self.activation = nn.SiLU()
        elif activation == 'gelu':
            self.activation = nn.GELU()
        else:
            self.activation = nn.ReLU()

        self.batch_norm = batch_norm
        if self.batch_norm:
            self.bn = nn.BatchNorm1d(hidden_dim)
            
        self.fc_out = nn.Linear(hidden_dim, out_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.activation(self.fc_in(x))
        x = self.dropout(x)
        
        if self.batch_norm:
            x = x.permute((0, 2, 1))
            x = self.bn(x)
            x = x.permute((0, 2, 1))
        
        x = self.fc_out(x)
        return x
    

class CrossAttention(nn.Module):
    def __init__(self):
        super(CrossAttention, self).__init__()

    def forward(self, x):
        return x

class FourierTransformer(nn.Module):
    def __init__(self,
                dim=64,
                dim_head=64,
                heads=4,
                mlp_dim=128,
                depth=4,
                dropout=0.1,
                rel_pos=Grid,
                attn_init=xavier_normal_,
                diagonal_weight=0.01
        ):
        super(FourierTransformer, self).__init__()
        
        self.attention_layers = []
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, 
                       FourierAttention(
                                dim=dim,
                                heads=heads,
                                dim_head=dim_head,
                                rel_pos=rel_pos(dim_head),
                                init=attn_init,
                                diagonal_weight=diagonal_weight
                            )
                        ),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout))
            ]))
        
    def forward(self, x: torch.Tensor, mask=None):
        for att, ff in self.attention_layers:
            x = att(x, mask=mask) + x
            x = ff(x) + x
        return x

class GalerkinTransformer(nn.Module):
    def __init__(self,
                dim=64,
                dim_head=64,
                heads=4,
                mlp_dim=128,
                depth=4,
                dropout=0.1,
                rel_pos=Grid,
                attn_init=xavier_normal_,
                diagonal_weight=0.01
        ):
        super(GalerkinTransformer, self).__init__()
        
        self.attention_layers = []
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, 
                       GalerkinAttention(
                                dim=dim,
                                heads=heads,
                                dim_head=dim_head,
                                rel_pos=rel_pos(dim_head),
                                init=attn_init,
                                diagonal_weight=diagonal_weight
                            )
                        ),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout))
            ]))
    
    def forward(self, x: torch.Tensor, mask=None):
        for att, ff in self.attention_layers:
            x = att(x, mask=mask) + x
            x = ff(x) + x
        return x