""" Testing the pack """

import torch
from no_pytorch import FNO1d

x = torch.rand(1, 10, 8)

model = FNO1d(in_channels=10, out_channels=1, fourier_modes=4, depth=8, freq_dim=20)

model(x)  # (1, 1, 64)

print(model(x).shape)