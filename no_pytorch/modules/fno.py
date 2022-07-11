'''
    @author: Zongyi Li
    Fourier layer. It does FFT, linear transform, and Inverse FFT.
    https://github.com/zongyi-li/fourier_neural_operator/blob/74b1572d4e02f215728b4aa5bf46374ed7daba06/fourier_1d.py
    https://github.com/zongyi-li/fourier_neural_operator/blob/74b1572d4e02f215728b4aa5bf46374ed7daba06/fourier_2d.py
    https://github.com/zongyi-li/fourier_neural_operator/blob/74b1572d4e02f215728b4aa5bf46374ed7daba06/fourier_3d.py
'''

import torch
from einops import rearrange
import torch.nn as nn
from .spectral_conv import SpectralConv1d, SpectralConv2d, SpectralConv3d
from .functional import get_grid_1d, get_grid_2d, get_grid_3d

class FNO1d(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 freq_dim,
                 fourier_modes=12,
                 depth=8,
                 dim_ff=128,
                 spectral_activation=nn.SiLU,
                 activation_ff=nn.SiLU,
                 grid=True,
                 last_activation=True
        ):
        super(FNO1d, self).__init__()
        
        _grid_size =  1 if grid else 0
        self.grid = grid
        self.project = nn.Linear(in_channels + _grid_size, freq_dim, 1)
        
        self.spectral_layers = nn.ModuleList([])
        for _ in range(depth):
            self.spectral_layers.append(
                SpectralConv1d(
                    in_channels=freq_dim,
                    out_channels=freq_dim,
                    modes=fourier_modes,
                    activation=spectral_activation
                ))

        if not last_activation:
            self.spectral_layers[-1].activation = nn.Identity()

        self.fc_out = nn.Sequential(
            nn.Linear(freq_dim, dim_ff),
            activation_ff(),
            nn.Linear(dim_ff, out_channels)
        )

    def forward(self, x: torch.Tensor):
        
        if self.grid:
            grid = get_grid_1d(x.shape, x.device)
            x = torch.cat((x, grid), dim=-1)
        
        x = self.project(x)
        
        for spectral_layer in self.spectral_layers:
            x = spectral_layer(x)
        
        x = self.fc_out(x)
        
        return x

class FNO2d(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 freq_dim,
                 fourier_modes=12,
                 depth=8,
                 dim_ff=128,
                 spectral_activation=nn.SiLU,
                 activation_ff=nn.SiLU,
                 grid=True,
                 last_activation=True
        ):
        super(FNO2d, self).__init__()

        _grid_size = 2 if grid else 0
        self.grid = grid
        self.project = nn.Linear(in_channels + _grid_size, freq_dim, 1)
        
        self.spectral_layers = nn.ModuleList([])
        for _ in range(depth):
            self.spectral_layers.append(
                SpectralConv2d(
                    in_channels=freq_dim,
                    out_channels=freq_dim,
                    modes=fourier_modes,
                    activation=spectral_activation
                ))

        if not last_activation:
            self.spectral_layers[-1].activation = nn.Identity()

        self.fc_out = nn.Sequential(
            nn.Linear(freq_dim, dim_ff),
            activation_ff(),
            nn.Linear(dim_ff, out_channels)
        )

    def forward(self, x: torch.Tensor, grid=None):
        
        if self.grid:
            grid = get_grid_2d(x.shape, x.device) if grid is None else grid
            x = torch.cat((x, grid), dim=-1)
        
        x = self.project(x)
        
        for spectral_layer in self.spectral_layers:
            x = spectral_layer(x)
            
        x = self.fc_out(x)
        
        return x
    
class FNO3d(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 freq_dim,
                 fourier_modes=12,
                 depth=8,
                 dim_ff=128,
                 spectral_activation=nn.SiLU,
                 activation_ff=nn.SiLU,
                 grid=True,
                 last_activation=True
        ):
        super(FNO3d, self).__init__()

        _grid_size = 3 if grid else 0
        self.grid = grid
        self.project = nn.Linear(in_channels + _grid_size, freq_dim, 1)
        
        self.spectral_layers = nn.ModuleList([])
        for _ in range(depth):
            self.spectral_layers.append(
                SpectralConv3d(
                    in_channels=freq_dim,
                    out_channels=freq_dim,
                    modes=fourier_modes,
                    activation=spectral_activation
                ))

        if not last_activation:
            self.spectral_layers[-1].activation = nn.Identity()

        self.fc_out = nn.Sequential(
            nn.Linear(freq_dim, dim_ff),
            activation_ff(),
            nn.Linear(dim_ff, out_channels)
        )

    def forward(self, x: torch.Tensor, grid=None):
        
        if self.grid:
            grid = get_grid_3d(x.shape, x.device) if grid is None else grid
            x = torch.cat((x, grid), dim=-1)
        
        x = self.project(x)
        
        for spectral_layer in self.spectral_layers:
            x = spectral_layer(x)
         
        x = self.fc_out(x)
        
        return x