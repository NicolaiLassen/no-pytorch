'''
    @author: Zongyi Li
    Fourier layer. It does FFT, linear transform, and Inverse FFT.
    https://github.com/zongyi-li/fourier_neural_operator/blob/74b1572d4e02f215728b4aa5bf46374ed7daba06/fourier_1d.py
    https://github.com/zongyi-li/fourier_neural_operator/blob/74b1572d4e02f215728b4aa5bf46374ed7daba06/fourier_2d.py
'''

import torch
from einops import rearrange
import torch.nn as nn
from .spectral_conv import SpectralConv1d, SpectralConv2d
from .functional import get_1d_grid, get_2d_grid

class FNO1d(nn.Module):
    def __init__(self,
                 in_channels=1,
                 out_channels=1,
                 freq_dim=20,
                 fourier_modes=12,
                 n_spectral_layers=8,
                 spectral_activation=nn.SiLU,
                 dim_ff=128,
                 activation_ff=nn.SiLU,
                 grid=True,
                 last_activation=True
        ):
        super(FNO1d, self).__init__()
        
        self.grid = grid
        if grid:
            self.project = nn.Linear(in_channels + 1, freq_dim)
        
        self.spectral_layers = nn.ModuleList([])
        for _ in range(n_spectral_layers):
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
        
        if x.ndim == 3:
            x = rearrange(x, "b c w -> b w c")
        elif x.ndim == 4:
            x = rearrange(x, "b t c w -> b w (t c)")          
        
        # field and bound grid
        if self.grid:
            grid = get_1d_grid(x.shape, x.device)
            x = torch.cat((x, grid), dim=-1)
            x = self.project(x)
        
        x = rearrange(x, "b w c -> b c w")
        
        for spectral_layer in self.spectral_layers:
            x = spectral_layer(x)
        
        x = rearrange(x, "b c w -> b w c")
        x = self.fc_out(x)
        
        if x.ndim == 3:
            x = rearrange(x, "b w c -> b c w")
        elif x.ndim == 4:
            x = rearrange(x, "b w c -> b 1 c w")     
        
        return x

class FNO2d(nn.Module):
    def __init__(self,
                 in_channels=1,
                 out_channels=1,
                 freq_dim=20,
                 fourier_modes=12,
                 n_spectral_layers=8,
                 spectral_activation=nn.SiLU,
                 dim_ff=128,
                 activation_ff=nn.SiLU,
                 grid=True,
                 last_activation=True
        ):
        super(FNO2d, self).__init__()

        self.grid = grid
        if grid:
            self.project = nn.Linear(in_channels + 2, freq_dim)
        
        self.spectral_layers = nn.ModuleList([])
        for _ in range(n_spectral_layers):
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
        
        if x.ndim == 4:
            x = rearrange(x, "b c w h -> b w h c")
        elif x.ndim == 5:
            x = rearrange(x, "b t c w h -> b w h (t c)")          
             
        # field and bound grid
        if self.grid:
            grid = get_2d_grid(x.shape, x.device) if grid is None else grid
            x = torch.cat((x, grid), dim=-1)
            x = self.project(x)
        
        x = rearrange(x, "b w h c -> b c w h")

        for spectral_layer in self.spectral_layers:
            x = spectral_layer(x)
        
        x = rearrange(x, "b c w h -> b w h c")
        x = self.fc_out(x)
        
        if x.ndim == 4:
            x = rearrange(x, "b w h c -> b c w h")
        elif x.ndim == 5:
            x = rearrange(x, "b w h c -> b 1 c w h")     
        
        return x
    
class FNO3d(nn.Module):
    
    def __init__(self):
        super(FNO3d, self).__init__()

    def forward(self, x):

        return x