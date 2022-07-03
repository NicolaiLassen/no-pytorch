'''
    @author: Zongyi Li
    1D Fourier layer. It does FFT, linear transform, and Inverse FFT.
    https://github.com/zongyi-li/fourier_neural_operator/blob/74b1572d4e02f215728b4aa5bf46374ed7daba06/fourier_1d.py
    
    2D Fourier layer. It does FFT, linear transform, and Inverse FFT.
    https://github.com/zongyi-li/fourier_neural_operator/blob/74b1572d4e02f215728b4aa5bf46374ed7daba06/fourier_2d.py
'''

import torch
from einops import rearrange
import torch.nn as nn
from .spectral_conv import SpectralConv1d, SpectralConv2d

class FNO1d(nn.Module):
    def __init__(self,
                 in_channels=1,
                 out_channels=1,
                 freq_dim=20,
                 fourier_modes=12,
                 n_spectral_layers=8,
                 dim_feedforward=128,
                 activation=nn.SiLU
        ):
        super(FNO1d, self).__init__()

        self.project = nn.Linear(in_channels + 1, freq_dim)
        
        self.layers = nn.ModuleList([])
        for _ in range(n_spectral_layers):
            self.layers.append(SpectralConv1d(freq_dim, freq_dim, fourier_modes))

        self.fc_out = nn.Sequential(
            nn.Linear(freq_dim, dim_feedforward),
            activation(),
            nn.Linear(dim_feedforward, out_channels)
        )

    def forward(self, x: torch.Tensor):
        
        if x.ndim == 3:
            x = rearrange(x, "b c w -> b w c")
        elif x.ndim == 4:
            x = rearrange(x, "b t c w -> b w (t c)")          
             
        grid = self.get_grid(x.shape, x.device)
        
        # field and bound grid
        x = torch.cat((x, grid), dim=-1)
    
        x = self.project(x)
        
        x = rearrange(x, "b w c -> b c w")
                
        for spectral in self.layers:
            x = spectral(x)
        
        x = rearrange(x, "b c w -> b w c")
        x = self.fc_out(x)
        
        if x.ndim == 3:
            x = rearrange(x, "b w c -> b c w")
        elif x.ndim == 4:
            x = rearrange(x, "b w c -> b 1 c w")     
        
        return x
    
    def get_grid(self, shape, device):
        batchsize, size_x = shape[0], shape[1]
        gridx = torch.linspace(0, 1, size_x, dtype=torch.float)
        gridx = gridx.reshape(1, size_x, 1).repeat([batchsize, 1, 1])
        return gridx.to(device)

class FNO2d(nn.Module):
    def __init__(self,
                 in_channels=1,
                 out_channels=1,
                 freq_dim=20,
                 fourier_modes=12,
                 n_spectral_layers=8,
                 dim_feedforward=128,
                 activation=nn.SiLU
        ):
        super(FNO2d, self).__init__()

        self.project = nn.Linear(in_channels + 2, freq_dim)
        
        self.layers = nn.ModuleList([])
        for _ in range(n_spectral_layers):
            self.layers.append(SpectralConv2d(freq_dim, freq_dim, fourier_modes))

        self.fc_out = nn.Sequential(
            nn.Linear(freq_dim, dim_feedforward),
            activation(),
            nn.Linear(dim_feedforward, out_channels)
        )

    def forward(self, x: torch.Tensor):
        
        if x.ndim == 4:
            x = rearrange(x, "b c w h -> b w h c")
        elif x.ndim == 5:
            x = rearrange(x, "b t c w h -> b w h (t c)")          
             
        grid = self.get_grid(x.shape, x.device)
        
        # field and bound grid
        x = torch.cat((x, grid), dim=-1)
    
        x = self.project(x)
        
        x = rearrange(x, "b w h c -> b c w h")
                
        for spectral in self.layers:
            x = spectral(x)
        
        x = rearrange(x, "b c w h -> b w h c")
        x = self.fc_out(x)
        
        if x.ndim == 4:
            x = rearrange(x, "b w h c -> b c w h")
        elif x.ndim == 5:
            x = rearrange(x, "b w h c -> b 1 c w h")     
        
        return x

    def get_grid(self, shape, device):
        batchsize, size_x, size_y = shape[0], shape[1], shape[2]
        gridx = torch.linspace(0, 1, size_x, dtype=torch.float)
        gridx = gridx.reshape(1, size_x, 1, 1).repeat([batchsize, 1, size_y, 1])
        
        gridy = torch.linspace(0, 1, size_y, dtype=torch.float)
        gridy = gridy.reshape(1, 1, size_y, 1).repeat([batchsize, size_x, 1, 1])
        return torch.cat((gridx, gridy), dim=-1).to(device)