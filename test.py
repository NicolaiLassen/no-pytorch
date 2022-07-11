""" Testing the pack """

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from no_pytorch import GalerkinTransformer, FNO2d, RoPE, WeightedL2Loss2d, SpectralConv2d
import h5py
import os
from einops import repeat
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from torch.utils.data import DataLoader, Dataset
import h5py
import numpy as np

plt.figure(figsize=(16,9),dpi=140)
def plot_prediction(y_pred, y_target, epoch) -> None:
    y_pred = y_pred.detach().cpu().numpy()
    y_target = y_target.detach().cpu().numpy()
    
    timeline = np.arange(y_pred.shape[0]) * 4e-12 * 1e9
    
    plt.plot(timeline, np.mean(y_target[:,0].reshape(y_target.shape[0],-1), axis=1), 'r')
    plt.plot(timeline, np.mean(y_target[:,1].reshape(y_target.shape[0],-1), axis=1), 'g')
    plt.plot(timeline, np.mean(y_target[:,2].reshape(y_target.shape[0],-1), axis=1), 'b')

    plt.plot(timeline, np.mean(y_pred[:,0].reshape(y_pred.shape[0],-1), axis=1), 'rx')
    plt.plot(timeline, np.mean(y_pred[:,1].reshape(y_pred.shape[0],-1), axis=1), 'gx')
    plt.plot(timeline, np.mean(y_pred[:,2].reshape(y_pred.shape[0],-1), axis=1), 'bx')
    
    legend_elements = [
        Line2D([0], [0], color='red', lw=4, label='Mx MagTense'),
        Line2D([0], [0], color='green', lw=4, label='My MagTense'),
        Line2D([0], [0], color='blue', lw=4, label='Mz MagTense'),
        Line2D([0], [0], marker='x', color='red', label='Mx Model'),
        Line2D([0], [0], marker='x', color='green', label='My Model'),
        Line2D([0], [0], marker='x', color='blue', label='Mz Model'),
    ]
    
    plt.legend(handles=legend_elements)
    plt.setp(plt.gca().get_legend().get_texts(), fontsize='16')
    plt.ylabel('$M_i [-]$', fontsize=32)
    plt.xlabel('$Time [ns]$', fontsize=32)
    plt.xticks(fontsize=24)
    plt.yticks(fontsize=24)
    plt.grid()
    plt.title('Galerkin Transformer', fontsize=48)
    plt.savefig("./images/galerkin_mag_problem_4_2_e{}.png".format(epoch))
    plt.clf()

class Net(nn.Module):
    """Some Information about Net"""
    def __init__(self):
        super(Net, self).__init__()
        
        # self.petrov_galerkin = GalerkinTransformer(dim=32, qkv_pos=RoPE(16), dim_head=16, depth=1, heads=4)
        self.fourier = FNO2d(in_channels=32, out_channels=3, freq_dim=20, fourier_modes=8, depth=3, grid=False)
        
    def forward(self, x: torch.Tensor, f: torch.Tensor):
        _, _, _, w, h = x.shape

        f = repeat(f, "b c -> b c w h", w=w, h=h)
        f = F.normalize(f)
        
        x = rearrange(x, 'b t c w h -> b (t c) w h')
        
        x = torch.cat([f, x], dim=1).permute(0, 2, 3, 1)
        
        """         x = rearrange(x, 'b w h c -> b (w h) c')
        
        x = self.petrov_galerkin(x)
        
        x = rearrange(x, 'b (w h) c -> b w h c', w=w, h=h)     """
        x = self.fourier(x)

        x = x.permute(0, 3, 1, 2).unsqueeze(1)
    
        return x

model = Net()
model = Net().cuda()

class MagDatasetLite(Dataset):
    def __init__(self,
                 batch_size=8,
                 block_size=20,
                 stride=20,
                 data_path=None,
                 n_data=-1):

        self.block_size = block_size
        self.batch_size = batch_size
        self.n_data = n_data
        self.stride = stride
        self.data_path = data_path        

        if self.data_path is not None:
            self._initialize()

    def __len__(self):
        return self.n_samples

    def _initialize(self):
        seq, field = self.read_h5_dataset(self.data_path, self.block_size, self.batch_size, self.stride, self.n_data)
        self.f = field
        self.a = seq[:, :10]  # (N, n, n, T_0:T_1)
        self.u = seq[:, 10:20] # (N, n, n, T_1:T_1+T_STEP)
        self.n_samples = len(self.a)

    @staticmethod
    def read_h5_dataset(
    file_path: str,
    block_size: int,
    batch_size: int = 32,
    stride: int = 5,
    n_data: int = -1,
    ) -> torch.Tensor:

        assert os.path.isfile(
            file_path), "Training HDF5 file {} not found".format(file_path)

        seq = []
        fields = []
        with h5py.File(file_path, "r") as f:

            n_seq = 0
            for key in f.keys():
                data_series = np.array(f[key]['sequence'])
                data_fields = np.array(f[key]['field'][:2])

                # Truncate in block of block_size
                for i in range(0,  data_series.shape[0] - block_size + 1, stride):
                    seq.append(np.expand_dims(data_series[i: i + block_size], axis=0))
                    fields.append(np.expand_dims(data_fields, axis=0))

                n_seq = n_seq + 1
                if(n_data > 0 and n_seq >= n_data):  # If we have enough time-series samples break loop
                    break

        seq_tensor = torch.from_numpy(np.concatenate(seq, axis=0))
        fields_tensor = torch.from_numpy(np.concatenate(fields, axis=0))

        if seq_tensor.shape[0] < batch_size:
            batch_size = seq_tensor.size(0)

        return seq_tensor, fields_tensor

    def __getitem__(self, idx):
        return dict(a=self.a[idx].float(),
                    u=self.u[idx].float(),
                    f=self.f[idx].float())

epochs = 1000
lr = 1e-4
h = 1/64
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
loss_func = WeightedL2Loss2d(regularizer=True, h=h, gamma=0.1)

seq = []
block_size = 20
stride = 20
n_data = 1
batch_size = 2

train_dataset = MagDatasetLite(data_path="./data/problem4.h5",
                                n_data=-1, batch_size=batch_size, block_size=block_size, stride=stride)
train_loader = DataLoader(train_dataset,
                        batch_size=batch_size,
                        shuffle=True,
                        drop_last=True,
                        pin_memory=True)


f = h5py.File('./data/problem4.h5')
sample_idx = 1
prob_sample = np.array(f[str(sample_idx)]['sequence'])
prob_field = np.array( f[str(sample_idx)]['field'])
prob_field = torch.from_numpy(prob_field)[:2].unsqueeze(0).cuda().float()
prob_sample = torch.from_numpy(prob_sample).cuda().float()

for epoch in range(epochs):
    optimizer.zero_grad()
    loss_total = 0

    model.train()
    for batch in train_loader:
        
        f = batch['f'].cuda()
        u = batch['u'].cuda()
        a = batch['a'].cuda()
    
        for t in range(10):
            out = model(a, f)
            u_step = u[:, t:t+1]
            
            loss, reg, _, _ = loss_func(out[:, 0], u_step[:, 0])
            loss = loss + reg
            loss_total += loss

            a = torch.cat((a[:, 1:], out), dim=1)
    
    loss_total.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
    optimizer.step()
    print(loss_total.item()/len(train_loader))

    model.eval()
    if epoch % 50 == 50 - 1:
        with torch.no_grad():
            whole_seq = prob_sample[:10].unsqueeze(0)
            for t in range(0, 400 - 10):
                im = model(whole_seq[:, t:], prob_field)
                whole_seq = torch.cat((whole_seq, im), dim=1)
            plot_prediction(whole_seq[0], prob_sample, epoch)