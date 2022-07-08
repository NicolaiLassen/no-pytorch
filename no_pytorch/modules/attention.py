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

class CrossAttention(nn.Module):
    def __init__(self):
        super(CrossAttention, self).__init__()

    def forward(self, x):
        raise "NOT IMPLEMENTED"
        return x

class FourierAttention(nn.Module):
    def __init__(self,
                 dim,
                 qkv_pos=None,
                 dot_pos=None,
                 pos_dim: int = 1,
                 heads=8,
                 dim_head=64,
                 fc_dropout=0.01,
                 attn_dropout=0.01,
                 xavier_init=1e-4,
                 symmetric_init=False,
                 norm=True,
                 eps=1e-5,
                 return_att=False  
                 ):
        super(FourierAttention, self).__init__()
        
        self.heads = heads
        self.dim_head = dim_head
        inner_dim = dim_head *  heads        
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)
        
        self.symmetric_init = symmetric_init
        self.xavier_init = xavier_init
        self.qkv_pos = qkv_pos
        self.dot_pos = dot_pos
        
        self.pos_dim = pos_dim
        self.norm = norm
        
        self.return_att = return_att
        self.attn_dropout = nn.Dropout(attn_dropout)
        
        self.norm_K = self._get_layernorm(dim_head, self.heads, eps=eps)
        self.norm_Q = self._get_layernorm(dim_head, self.heads, eps=eps)
        
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(fc_dropout)
        )
        
        self._init_parameters()

    def forward(self, x: torch.Tensor, pos: torch.Tensor=None, mask: torch.Tensor=None, weight: torch.Tensor=None):
        qkv= self.to_qkv(x).chunk(3, dim = -1)
        
        if self.qkv_pos is not None:
           assert self.dim_head == self.qkv_pos.dim,\
               f'Size of head dim and pos dim much match, got {self.dim_head}, {self.qkv_pos.dim}'
           qkv = self.qkv_pos(*qkv, pos, self.heads)
           
        q, k, v = \
            map(lambda t:  
                rearrange(t, 'b n (h k d) -> b (n d) h k', h=self.heads, k=self.dim_head)
                .transpose(1, 2)
            , qkv)
        
        if weight is not None:
            q, k = weight*q, weight*k

        if self.norm:
            k = torch.stack(
                        [norm(x) for norm, x in
                        zip(self.norm_K, (k[:, i, ...] for i in range(self.heads)))], dim=1)
            q = torch.stack(
                        [norm(x) for norm, x in
                        zip(self.norm_Q, (q[:, i, ...] for i in range(self.heads)))], dim=1)   
        
        seq_len = q.size(-2)
        dots = torch.matmul(q, k.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.dim_head))

        if self.dot_pos is not None:
            raise "NOT IMPLEMENTED"

        if mask is not None:
                scores = scores.masked_fill(mask == 0, 0)

        p_attn = dots / seq_len

        p_attn = self.attn_dropout(p_attn)

        out = torch.matmul(p_attn, v)
        
        out = rearrange(out.transpose(1, 2).contiguous(), 'b d h k -> b d (h k)')
        
        out = self.to_out(out)
        
        if self.return_att:
            return out, p_attn
        return  out          
    
    def _init_parameters(self):
        for param in self.to_qkv.parameters():
            xavier_normal_(param, gain=self.xavier_init)
            if self.diagonal_weight > 0.0:
                param.data += self.diagonal_weight * \
                    torch.diag(torch.ones(
                        param.size(-1), dtype=torch.float))
            if self.symmetric_init:
                param.data += param.data.T
    
    @staticmethod
    def _get_layernorm(normalized_dim, n_head, **kwargs):
        return nn.ModuleList(
            [copy.deepcopy(nn.LayerNorm(normalized_dim, **kwargs)) for _ in range(n_head)])

class GalerkinAttention(nn.Module):
    def __init__(self,
                 dim,
                 qkv_pos=None,
                 dot_pos=None,
                 pos_dim: int = 1,
                 heads=8,
                 dim_head=64,
                 fc_dropout=0.01,
                 attn_dropout=0.01,
                 xavier_init=1e-4,
                 symmetric_init=False,
                 norm=True,
                 eps=1e-5,
                 return_att=False  
                 ):
        super(GalerkinAttention, self).__init__()
        
        self.heads = heads
        self.dim_head = dim_head
        inner_dim = dim_head *  heads        
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)
        
        self.symmetric_init = symmetric_init
        self.xavier_init = xavier_init
        self.qkv_pos = qkv_pos
        self.dot_pos = dot_pos
        
        self.pos_dim = pos_dim
        self.norm = norm
        
        self.return_att = return_att
        self.attn_dropout = nn.Dropout(attn_dropout)
        
        self.norm_K = self._get_layernorm(self.dim_head, self.heads, eps=eps)
        self.norm_V = self._get_layernorm(self.dim_head, self.heads, eps=eps)
        
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(fc_dropout)
        )
        
        self._init_parameters()

    def forward(self, x: torch.Tensor, pos: torch.Tensor=None, mask: torch.Tensor=None, weight: torch.Tensor=None):
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        
        if self.qkv_pos is not None:
            assert self.dim_head == self.qkv_pos.dim,\
                f'Size of head dim and pos dim much match, got {self.dim_head}, {self.qkv_pos.dim}'
            qkv = self.qkv_pos(*qkv, pos, self.heads)
        
        q, k, v = \
            map(lambda t:  
                rearrange(t, 'b n (h k d) -> b (n d) h k', h=self.heads, k=self.dim_head)
                .transpose(1, 2)
            , qkv)
        
        if weight is not None:
            q, k = weight*q, weight*k
    
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

        p_attn = self.attn_dropout(p_attn)

        out = torch.matmul(q, p_attn)
        
        out = rearrange(out.transpose(1, 2).contiguous(), 'b d h k -> b d (h k)')
        
        out = self.to_out(out)
        
        if self.return_att:
            return out, p_attn
        return  out      
    
    def _init_parameters(self):
        for param in self.to_qkv.parameters():
            xavier_normal_(param, gain=self.xavier_init)
            if self.symmetric_init:
                param.data += param.data.T
     
    @staticmethod
    def _get_layernorm(normalized_dim, n_head, **kwargs):
        return nn.ModuleList(
            [copy.deepcopy(nn.LayerNorm(normalized_dim, **kwargs)) for _ in range(n_head)])