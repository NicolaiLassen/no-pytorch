""" Testing the pack """

import torch
import torch.nn as nn
from einops import rearrange
from no_pytorch import GalerkinTransformer, FourierTransformer, FNO2d, RoPE

x = torch.rand(1, 10, 64, 64).cuda()

class Net(nn.Module):
    """Some Information about Net"""
    def __init__(self):
        super(Net, self).__init__()
        
        self.proj = nn.Conv2d(10, 64, 1)
        self.petrov_galerkin = GalerkinTransformer(dim=64, qkv_pos=RoPE(64), dim_head=64, depth=1)
        self.fourier = FNO2d(in_channels=64, out_channels=1, freq_dim=20, depth=1)
        
    def forward(self, x):
        w, h = x.shape[2:]
        x = self.proj(x)
        x = rearrange(x, 'b c w h -> b (w h) c')
        x = self.petrov_galerkin(x)
        x = rearrange(x, 'b (w h) c -> b c w h', w=w, h=h)
        x = self.fourier(x)
        return x

model = Net().cuda()
# in [x_0, x_1 ... x_10] out x_10+1
x_hat = model(x) # (1, 1, 64, 64)

print(x_hat.shape)