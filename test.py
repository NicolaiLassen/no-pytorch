""" Testing the pack """

import torch
from no_pytorch import FNO1d, FNO2d

x = torch.rand(1, 10, 64, 64)
model = FNO2d(in_channels=10, out_channels=1, n_spectral_layers=16)

print(model(x).shape)