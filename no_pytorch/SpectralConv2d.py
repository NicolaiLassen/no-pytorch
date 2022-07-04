'''
    @author: Zongyi Li
    2D Fourier layer. It does FFT, linear transform, and Inverse FFT.
    https://github.com/zongyi-li/fourier_neural_operator/blob/74b1572d4e02f215728b4aa5bf46374ed7daba06/fourier_2d_time.py
'''

from einops import rearrange
import torch
import torch.nn as nn
import torch.fft as fft
from torch.nn.init import xavier_normal_

def pair(t):
    return t if isinstance(t, tuple) else (t, t)

class SpectralConv2d(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels, 
                 modes=8,
                 dropout=0.1,
                 norm='ortho',
                 activation=nn.SiLU,
                 return_freq=False,         
        ):
        super(SpectralConv2d, self).__init__()
        
        modes_x, modes_y = pair(modes)
        self.modes_x = modes_x
        self.modes_y = modes_y
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        self.return_freq = return_freq
        self.norm = norm

        self.activation = activation()
        self.shortcut = nn.Linear(in_channels, out_channels)
        self.dropout =  nn.Dropout(dropout)

        scale = (1 / (in_channels * out_channels))
        self.fourier_weight = nn.ParameterList([nn.Parameter(
            torch.rand(in_channels, out_channels, self.modes_x, self.modes_y, dtype=torch.cfloat)) 
                                                for _ in range(2)])
        
        for param in self.fourier_weight:
            xavier_normal_(param, gain=scale * torch.sqrt(torch.tensor(in_channels+out_channels)))
        
    @staticmethod
    def compl_mul2d(input, weights):
        # (batch, in_channel, x,y ), (in_channel, out_channel, x,y) -> (batch, out_channel, x,y)
        return torch.einsum("bixy,ioxy->boxy", input, weights)

    def forward(self, x):
        b, _, w, h = x.shape
        
        x = rearrange(x, "b c w h -> b w h c")
        
        res = self.shortcut(x)
        x = self.dropout(x)
        
        x = rearrange(x, "b w h c -> b c w h")
        
        # multiply relevant fourier modes
        x_ft = fft.rfft2(x, s=(w, h), norm=self.norm)

        # multiply relevant fourier modes
        out_ft = torch.zeros(b, self.out_channels,  w, h//2 + 1, dtype=torch.cfloat, device=x.device)
        
        out_ft[:, :, :self.modes_x, :self.modes_y] = \
            self.compl_mul2d(x_ft[:, :, :self.modes_x, :self.modes_y], self.fourier_weight[0])
            
        out_ft[:, :, -self.modes_x:, :self.modes_y] = \
            self.compl_mul2d(x_ft[:, :, -self.modes_x:, :self.modes_y], self.fourier_weight[1])
            
        # return to physical space
        x = fft.irfft2(out_ft, s=(w, h), norm=self.norm)
        
        x = rearrange(x, "b c w h -> b w h c")
    
        x = self.activation(x + res)      
        
        x = rearrange(x, "b w h c -> b c w h")  
        
        if self.return_freq:
            return x, out_ft
        else:
            return x    
             
if __name__ == '__main__':
    x = torch.rand((1, 10, 64, 64))
    spec_2d = SpectralConv2d(10, 10)
    print(spec_2d(x).shape)