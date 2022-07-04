'''
    @author: Zongyi Li
    Fourier layer. It does FFT, linear transform, and Inverse FFT.
    https://github.com/zongyi-li/fourier_neural_operator/blob/74b1572d4e02f215728b4aa5bf46374ed7daba06/fourier_1d.py
    https://github.com/zongyi-li/fourier_neural_operator/blob/74b1572d4e02f215728b4aa5bf46374ed7daba06/fourier_2d.py
'''

import torch
from einops import rearrange
import torch.nn as nn
from .spectral_conv import SpectralConv1d, SpectralConv2d, SpectralConv3d
from .functional import get_1d_grid, get_2d_grid, get_3d_grid

class FNO1d(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 freq_dim,
                 depth_spectral_layers=8,
                 fourier_modes=12,
                 dim_ff=128,
                 spectral_activation=nn.SiLU,
                 activation_ff=nn.SiLU,
                 grid=True,
                 last_activation=True
        ):
        super(FNO1d, self).__init__()
        
        _grid_size = 1
        self.grid = grid
        if grid:
            self.project = nn.Conv1d(in_channels + _grid_size, freq_dim, 1)
        
        self.spectral_layers = nn.ModuleList([])
        for _ in range(depth_spectral_layers):
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
        
        # field and bound grid
        if self.grid:
            grid = get_1d_grid(x.shape, x.device)
            x = torch.cat((x, grid), dim=1)
            x = self.project(x)
        
        for spectral_layer in self.spectral_layers:
            x = spectral_layer(x)
        
        x = rearrange(x, "b c w -> b w c")
        
        x = self.fc_out(x)
        
        x = rearrange(x, "b w c -> b c w")
        
        return x

class FNO2d(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 freq_dim,
                 depth_spectral_layers=8,
                 fourier_modes=12,
                 dim_ff=128,
                 spectral_activation=nn.SiLU,
                 activation_ff=nn.SiLU,
                 grid=True,
                 last_activation=True
        ):
        super(FNO2d, self).__init__()

        _grid_size = 2
        self.grid = grid
        if grid:
            self.project = nn.Conv2d(in_channels + _grid_size, freq_dim, 1)
        
        self.spectral_layers = nn.ModuleList([])
        for _ in range(depth_spectral_layers):
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
        
        # field and bound grid
        if self.grid:
            grid = get_2d_grid(x.shape, x.device) if grid is None else grid
            x = torch.cat((x, grid), dim=1)
            x = self.project(x)
        
        for spectral_layer in self.spectral_layers:
            x = spectral_layer(x)
        
        x = rearrange(x, "b c w h -> b w h c")
        
        x = self.fc_out(x)
        
        x = rearrange(x, "b w h c -> b c w h")
        
        return x
    
class FNO3d(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 freq_dim,
                 depth_spectral_layers=8,
                 fourier_modes=12,
                 dim_ff=128,
                 spectral_activation=nn.SiLU,
                 activation_ff=nn.SiLU,
                 grid=True,
                 last_activation=True
        ):
        super(FNO3d, self).__init__()

        _grid_size = 3
        self.grid = grid
        if grid:
            self.project = nn.Conv3d(in_channels + _grid_size, freq_dim, 1)
        
        self.spectral_layers = nn.ModuleList([])
        for _ in range(depth_spectral_layers):
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
        
        # field and bound grid
        if self.grid:
            grid = get_3d_grid(x.shape, x.device) if grid is None else grid
            x = torch.cat((x, grid), dim=1)
            x = self.project(x)
        
        for spectral_layer in self.spectral_layers:
            x = spectral_layer(x)
        
        x = rearrange(x, "b c w h d -> b w h d c")
        
        x = self.fc_out(x)
        
        x = rearrange(x, "b w h d c -> b c w h d")
        
        return x