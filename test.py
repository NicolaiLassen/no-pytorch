""" Testing the pack """

import torch
from no_pytorch import FNO1d, FNO2d, FNO3d

x = torch.rand(1, 10, 8, 8, 8)
model = FNO3d(in_channels=10, out_channels=1, fourier_modes=4, depth_spectral_layers=16, freq_dim=20)
x = model(x)
print(x.shape)
