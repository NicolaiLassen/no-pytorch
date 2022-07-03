'''
    @author: Shuhao Cao
    https://github.com/scaomath/galerkin-transformer/blob/main/libs/layers.py
    @author: Zijie Li
    https://arxiv.org/pdf/2205.13671.pdf
'''

import torch
import torch.nn as nn
from .fno import FNO1d, FNO2d
from .attention import FourierAttention, GalerkinAttention, RoPE
import copy

class FourierTransformer1d(nn.Module):
    def __init__(self,
                in_channels,
                out_channels,
                dim_head=64,
                heads=4,
                attention_depth=4,
                spectral_depth=1,
                spectral_dim=20,
                head_pos=RoPE,
                attn_norm=False,
                xavier_init=0.01,
                diagonal_weight=0.01,
        ):
        super(FourierTransformer1d, self).__init__()
        
        self.project = nn.Linear(in_channels, spectral_dim)
        
        attention_layer = FourierAttention(
            in_channels=spectral_dim,
            out_channels=out_channels,
            heads=heads,
            dim_head=dim_head,
            head_pos=head_pos(dim_head),
            xavier_init=xavier_init,
            diagonal_weight=diagonal_weight
        )
        self.attention_layers = nn.ModuleList(
            [copy.deepcopy(attention_layer) for _ in range(attention_depth)])    
    
        self.regressor = FNO1d(spectral_depth=spectral_depth)

    def forward(self, x: torch.Tensor, mask=None):
        
        x = self.project(x)
        
        for attention in self.attention_layers:
            x = attention(x, mask=mask)
        
        self.regressor(x)
        
        return x
    
    def get_grid(self, shape, device):
        return
    
class GalerkinTransformer1d(nn.Module):
    def __init__(self,
                 head_pos=RoPE,
                 dim_head=64
        ):
        super(GalerkinTransformer1d, self).__init__()
        
        GalerkinAttention()
        
    def forward(self, x: torch.Tensor):
        return
    
    def get_grid(self, shape, device):
        return
   
class FourierTransformer2d(nn.Module):
    def __init__(self,
                 head_pos=RoPE,
                 dim_head=64
        ):
        super(FourierTransformer2d, self).__init__()
        
        pos = head_pos(dim_head)
        FourierAttention()
        
    def forward(self, x: torch.Tensor):
        return
    
    def get_grid(self, shape, device):
        return
    
class GalerkinTransformer2d(nn.Module):
    def __init__(self,
                 
        ):
        super(GalerkinTransformer2d, self).__init__()
        
        GalerkinAttention()
        
    def forward(self, x: torch.Tensor):
        return
    
    def get_grid(self, shape, device):
        return