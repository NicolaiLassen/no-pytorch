import torch
import torch.nn as nn
from .attention import FourierAttention, GalerkinAttention
from rotary_embedding_torch import RotaryEmbedding

import vit_pytorch.vit

class RoPE(nn.Module):
    def __init__(self,
                dim,
                custom_freqs = None,
                freqs_for = 'lang',
                theta = 10000,
                max_freq = 10,
                num_freqs = 1,
                learned_freq = False
        ):
        super(RoPE, self).__init__()
        self.rotary_emb = RotaryEmbedding(dim,
                                          custom_freqs,
                                          freqs_for,
                                          theta,
                                          max_freq,
                                          num_freqs,
                                          learned_freq
                                          )
    def forward(self, q: torch.Tensor, k: torch.Tensor):
        q = self.rotary_emb.rotate_queries_or_keys(q)
        k = self.rotary_emb.rotate_queries_or_keys(k)
        return q, k

class FourierTransformer1d(nn.Module):
    def __init__(self,
                 head_pos=RoPE,
                 dim_head=64
        ):
        super(FourierTransformer1d, self).__init__()
        
        pos = head_pos(dim_head)
        FourierAttention()
        
    def forward(self, x: torch.Tensor):
        
        
        return
    
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