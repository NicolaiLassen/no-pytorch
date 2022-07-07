""" Testing the pack """

import torch
from no_pytorch import FNO2d, RoPE
import torch.nn as nn
from einops import rearrange

from no_pytorch import GalerkinTransformer, FourierTransformer

x = torch.rand(1, 10, 64)

model = GalerkinTransformer(dim=64, qkv_pos=RoPE(256), dim_head=256, depth=12)

print(model(x).shape)
