""" Testing the pack """

import torch
from no_pytorch import FNO2d
import vit_pytorch.vit
import torch.nn as nn
from einops import rearrange

from no_pytorch.modules.attention import GalerkinAttention

x = torch.rand(1, 10, 64)

model = GalerkinAttention(dim=64)

print(model(x).shape)
