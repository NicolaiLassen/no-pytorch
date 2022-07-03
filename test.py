""" Testing the pack """

import torch
from no_pytorch import FNO1d, FNO2d

x = torch.rand(1, 10, 64)
model = FNO1d(in_channels=10, out_channels=100, depth_spectral_layers=16, freq_dim=20)

print(model(x).shape)

import vit_pytorch.vit