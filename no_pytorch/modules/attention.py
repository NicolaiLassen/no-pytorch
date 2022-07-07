'''
    @author: Shuhao Cao
    https://github.com/scaomath/galerkin-transformer/blob/main/libs/layers.py
    @author: Zijie Li
    https://arxiv.org/pdf/2205.13671.pdf
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import xavier_normal_
from einops import rearrange
import copy

class FourierAttention(nn.Module):
    def __init__(self,
                 dim,
                 rel_pos=None,
                 heads=8,
                 dim_head=64,
                 dropout=0.1,
                 init=xavier_normal_,
                 diagonal_weight=1e-2,
                 symmetric_init=False,
                 norm=False,
                 norm_type='layer',
                 eps=1e-5,):
        super(FourierAttention, self).__init__()
        
        inner_dim = dim_head *  heads    
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)
        
        self.head_pos = rel_pos
        
        self.diagonal_weight = diagonal_weight
        self.symmetric_init = symmetric_init
        
        self.add_norm = norm
        self.norm_type = norm_type
        
        if norm:
            self._get_norm(eps=eps)

        self.attn_weight = None
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value, pos=None, mask=None, weight=None):
        if mask is not None:
            mask = mask.unsqueeze(1)

        bsz = query.size(0)
        
        if weight is not None:
            query, key = weight*query, weight*key

        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)
        
        if self.head_pos is not None:
           q, k = self.head_pos(q, k)
            
        return

class GalerkinAttention(nn.Module):
    def __init__(self,
                 dim,
                 rel_pos=None,
                 dot_pos=None,
                 pos_dim: int = 1,
                 heads=8,
                 dim_head=64,
                 dropout=0.1,
                 init=xavier_normal_,
                 diagonal_weight=1e-2,
                 symmetric_init=False,
                 norm=True,
                 eps=1e-5,
                 return_att=False  
                 ):
        super(GalerkinAttention, self).__init__()
        
        self.d_k = dim // heads
        self.heads = heads
        inner_dim = dim_head *  heads    
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)
        
        self.head_pos = rel_pos
        self.dot_pos = dot_pos
        
        self.pos_dim = pos_dim
        self.norm = norm
        
        self.return_att = return_att
        self.dropout = nn.Dropout(dropout)
        
        self.norm_K = self._get_layernorm(self.d_k, self.heads, eps=eps)
        self.norm_V = self._get_layernorm(self.d_k, self.heads, eps=eps)
        
        # TODO active ETC

    def forward(self, x: torch.Tensor, pos: torch.Tensor=None, mask: torch.Tensor=None, weight: torch.Tensor=None):
        b = x.size(0)
        qkv = self.to_qkv(x).chunk(3, dim = -1)        
        q, k, v = \
            map(lambda t:  rearrange(t, 'b n (h k d) -> b (n d) h k', h = self.heads, k=self.d_k).transpose(1, 2), qkv)
        
        if weight is not None:
            q, k = weight*q, weight*k
    
        if self.head_pos is not None:
           q, k = self.head_pos(q, k, pos, self.heads)
           
        if self.norm:
            k = torch.stack(
                        [norm(x) for norm, x in
                        zip(self.norm_K, (k[:, i, ...] for i in range(self.heads)))], dim=1)
            v = torch.stack(
                        [norm(x) for norm, x in
                        zip(self.norm_V, (v[:, i, ...] for i in range(self.heads)))], dim=1)   
        
        seq_len = q.size(-2)
        dots = torch.matmul(k.transpose(-2, -1), v)

        if self.dot_pos is not None:
            raise "NOT IMPLEMENTED"

        if mask is not None:
            # TODO
            raise RuntimeError("linear attention does not support casual mask.")

        p_attn = dots / seq_len

        p_attn = self.dropout(p_attn)

        out = torch.matmul(q, p_attn)
        
        out_dim = self.heads * self.d_k if pos is None else self.heads * (self.d_k + self.pos_dim)
         
        out = x.transpose(1, 2).contiguous().view(b, -1, out_dim)
        
        if self.return_att:
            raise out, p_attn
        return  out      
    
    @staticmethod
    def _get_layernorm(normalized_dim, n_head, **kwargs):
        return nn.ModuleList(
            [copy.deepcopy(nn.LayerNorm(normalized_dim, **kwargs)) for _ in range(n_head)])