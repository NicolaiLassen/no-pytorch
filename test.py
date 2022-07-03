""" Testing the pack """

import torch
from fno_pytorch import FNO1d

x = torch.rand(1, 10, 64)
model = FNO1d(in_channels=10)

print(model(x).shape)