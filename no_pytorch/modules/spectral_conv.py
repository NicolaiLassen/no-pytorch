'''
    @author: Zongyi Li
    1D Fourier layer. It does FFT, linear transform, and Inverse FFT.
    https://github.com/zongyi-li/fourier_neural_operator/blob/74b1572d4e02f215728b4aa5bf46374ed7daba06/fourier_1d.py
    
    2D Fourier layer. It does FFT, linear transform, and Inverse FFT.
    https://github.com/zongyi-li/fourier_neural_operator/blob/74b1572d4e02f215728b4aa5bf46374ed7daba06/fourier_2d.py
'''

import torch
import torch.nn as nn
import torch.fft as fft
from torch.nn.init import xavier_normal_
from .functional import pair, triplet

class SpectralConv1d(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels, 
                 modes=12,
                 dropout=0.01,
                 norm='ortho',
                 init=xavier_normal_,
                 activation=nn.SiLU,
                 return_freq=False  
        ):
        super(SpectralConv1d, self).__init__()
    
        self.modes_x = modes
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        self.norm = norm
        self.return_freq = return_freq
        
        self.activation = activation()
        self.shortcut = nn.Conv1d(in_channels, out_channels, 1)
        self.dropout =  nn.Dropout(dropout)

        scale = (1 / (in_channels * out_channels))
        self.fourier_weight = nn.Parameter(torch.rand(in_channels, out_channels, self.modes_x, dtype=torch.cfloat))
        
        for param in self.fourier_weight:
            init(param, gain=scale * torch.sqrt(torch.tensor(in_channels+out_channels)))
        
    @staticmethod
    def compl_mul1d(input, weights):
        # (batch, in_channel, x ), (in_channel, out_channel, x) -> (batch, out_channel, x)
        return torch.einsum("bix,iox->box", input, weights)

    def forward(self, x: torch.Tensor):
        b, _, w = x.shape
        
        assert self.modes_x <= w // 2 + 1, "Modes should be smaller than w // 2 + 1"
        
        res = self.shortcut(x)
        x = self.dropout(x)
        
        # compute fourier coeffcients up to factor of e^(- constant)
        x_ft = fft.rfft(x, norm=self.norm, dim=-1)
        
        # multiply relevant fourier modes
        out_ft = torch.zeros(b, self.out_channels, self.modes_x, dtype=torch.cfloat, device=x.device)
        out_ft[:, :, :self.modes_x] = self.compl_mul1d(x_ft[:, :, :self.modes_x], self.fourier_weight)
        
        # return to physical space
        x = fft.irfft(out_ft, n=w, norm=self.norm, dim=-1)
        
        x = self.activation(x + res)
              
        if self.return_freq:
            return x, out_ft
        
        return x
        
class SpectralConv2d(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels, 
                 modes=12,
                 dropout=0.01,
                 norm='ortho',
                 init=xavier_normal_,
                 activation=nn.SiLU,
                 return_freq=False
        ):
        super(SpectralConv2d, self).__init__()
        
        self.modes_x, self.modes_y = pair(modes)
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        self.norm = norm
        self.return_freq = return_freq
    
        self.activation = activation()
        self.shortcut = nn.Conv2d(in_channels, out_channels, 1)
        self.dropout =  nn.Dropout(dropout)

        scale = (1 / (in_channels * out_channels))
        self.fourier_weight = nn.ParameterList([nn.Parameter(
            torch.rand(in_channels, out_channels, self.modes_x, self.modes_y, dtype=torch.cfloat)) 
                                                for _ in range(2)])
        
        for param in self.fourier_weight:
            init(param, gain=scale * torch.sqrt(torch.tensor(in_channels+out_channels)))
        
    @staticmethod
    def compl_mul2d(input, weights):
        # (batch, in_channel, x,y ), (in_channel, out_channel, x,y) -> (batch, out_channel, x,y)
        return torch.einsum("bixy,ioxy->boxy", input, weights)

    def forward(self, x: torch.Tensor):
        b, _, w, h = x.shape
        
        assert self.modes_x <= w, "Modes x should be smaller than w"
        assert self.modes_y <= h // 2 + 1, "Modes y should be smaller than h // 2 + 1"
        
        res = self.shortcut(x)
        x = self.dropout(x)
        
        # multiply relevant fourier modes
        x_ft = fft.rfft2(x, s=(w, h), norm=self.norm, dim=(-2, -1))

        # multiply relevant fourier modes
        out_ft = torch.zeros(b, self.out_channels,  self.modes_x, self.modes_y, dtype=torch.cfloat, device=x.device)
        out_ft[:, :, :self.modes_x, :self.modes_y] = \
            self.compl_mul2d(x_ft[:, :, :self.modes_x, :self.modes_y], self.fourier_weight[0])
        out_ft[:, :, -self.modes_x:, :self.modes_y] = \
            self.compl_mul2d(x_ft[:, :, -self.modes_x:, :self.modes_y], self.fourier_weight[1])
            
        # return to physical space
        x = fft.irfft2(out_ft, s=(w, h), norm=self.norm, dim=(-2, -1))
        
        x = self.activation(x + res)      
        
        if self.return_freq:
            return x, out_ft
        
        return x

class SpectralConv3d(nn.Module):
    def __init__(self,
                in_channels,
                 out_channels, 
                 modes=12,
                 dropout=0.01,
                 norm='ortho',
                 init=xavier_normal_,
                 activation=nn.SiLU,
                 return_freq=False
                 ):
        super(SpectralConv3d, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        
        self.modes_x, self.modes_y, self.modes_z = triplet(modes)
        
        self.norm = norm
        self.return_freq = return_freq
    
        self.activation = activation()
        self.shortcut = nn.Conv3d(in_channels, out_channels, 1)
        self.dropout =  nn.Dropout(dropout)

        scale = (1 / (in_channels * out_channels))
        self.fourier_weight = nn.ParameterList([nn.Parameter(
            torch.rand(in_channels, out_channels, self.modes_x, self.modes_y, self.modes_z, dtype=torch.cfloat)) 
                                                for _ in range(4)])
        
        for param in self.fourier_weight:
            init(param, gain=scale * torch.sqrt(torch.tensor(in_channels+out_channels)))
        
    # Complex multiplication
    def compl_mul3d(self, input, weights):
        # (batch, in_channel, x,y,t ), (in_channel, out_channel, x,y,t) -> (batch, out_channel, x,y,t)
        return torch.einsum("bixyz,ioxyz->boxyz", input, weights)

    def forward(self, x):
        b, _, w, h, d = x.shape
        
        assert self.modes_x <= w, "Modes x should be smaller than w"
        assert self.modes_y <= h, "Modes y should be smaller than h"
        assert self.modes_z <= d // 2 + 1, "Modes z should be smaller than h // 2 + 1"
        
        res = self.shortcut(x)
        x = self.dropout(x)
        
        # compute Fourier coeffcients up to factor of e^(- something constant)
        x_ft = torch.fft.rfftn(x, dim=[-3,-2,-1])

        # Multiply relevant Fourier modes
        out_ft = torch.zeros(b, self.out_channels, self.modes_x, self.modes_y, self.modes_z, dtype=torch.cfloat, device=x.device)
        
        out_ft[:, :, :self.modes_x, :self.modes_y, :self.modes_z] = \
            self.compl_mul3d(x_ft[:, :, :self.modes_x, :self.modes_y, :self.modes_z], self.fourier_weight[0])
        out_ft[:, :, -self.modes_x:, :self.modes_y, :self.modes_z] = \
            self.compl_mul3d(x_ft[:, :, -self.modes_x:, :self.modes_y, :self.modes_z], self.fourier_weight[1])
        out_ft[:, :, :self.modes_x, -self.modes_y:, :self.modes_z] = \
            self.compl_mul3d(x_ft[:, :, :self.modes_x, -self.modes_y:, :self.modes_z], self.fourier_weight[2])
        out_ft[:, :, -self.modes_x:, -self.modes_y:, :self.modes_z] = \
            self.compl_mul3d(x_ft[:, :, -self.modes_x:, -self.modes_y:, :self.modes_z], self.fourier_weight[3])

        # return to physical space
        x = torch.fft.irfftn(out_ft, s=(w, h, d))
        
        x = self.activation(x + res)      
        
        if self.return_freq:
            return x, out_ft
        
        return x

if __name__ == '__main__':
    x = torch.rand((1, 10, 64))
    spec_1d = SpectralConv1d(10, 10)