import torch
import torch.nn.functional as F

def get_grid_1d(shape, device):
    b, size_x = shape[0], shape[2]
    grid_x = torch.linspace(0, 1, size_x, dtype=torch.float)
    grid_x = grid_x.reshape(1, 1, size_x).repeat([b, 1, 1])
    return grid_x.to(device)

def get_grid_2d(shape, device):
    b, _, size_x, size_y = shape
    grid_x = torch.linspace(0, 1, size_x, dtype=torch.float)
    grid_x = grid_x.reshape(1, 1, size_x, 1).repeat([b, 1, 1, size_y])
    grid_y = torch.linspace(0, 1, size_y, dtype=torch.float)
    grid_y = grid_y.reshape(1, 1, 1, size_y).repeat([b, 1, size_x, 1])
    return torch.cat((grid_x, grid_y), dim=1).to(device)

def get_grid_3d(shape, device):
    b, _, size_x, size_y, size_z = shape
    grid_x = torch.linspace(0, 1, size_x, dtype=torch.float)
    grid_x = grid_x.reshape(1, 1, size_x, 1, 1).repeat([b, 1, 1, size_y, size_z])
    grid_y = torch.linspace(0, 1, size_y, dtype=torch.float)
    grid_y = grid_y.reshape(1, 1, 1, size_y, 1).repeat([b, 1, size_x, 1, size_z])
    grid_z = torch.linspace(0, 1, size_z, dtype=torch.float)
    grid_z = grid_z.reshape(1, 1, 1, 1, size_z).repeat([b, 1, size_x, size_y, 1])
    return torch.cat((grid_x, grid_y, grid_z), dim=1).to(device)

def default(value, d):
    '''
    helper taken from https://github.com/lucidrains/linear-attention-transformer
    '''
    return d if value is None else value

def pair(t):
    return t if isinstance(t, tuple) else (t, t)

def triplet(t):
    return t if isinstance(t, tuple) else (t, t, t)

def exists(val):
    return val is not None