import torch

def get_1d_grid(shape, device):
    b, size_x = shape[0], shape[1]
    grid_x = torch.linspace(0, 1, size_x, dtype=torch.float)
    grid_x = grid_x.reshape(1, size_x, 1).repeat([b, 1, 1])
    return grid_x.to(device)

def get_2d_grid(shape, device):
    b, size_x, size_y = shape[0], shape[1], shape[2]
    grid_x = torch.linspace(0, 1, size_x, dtype=torch.float)
    grid_x = grid_x.reshape(1, size_x, 1, 1).repeat([b, 1, size_y, 1])
    grid_y = torch.linspace(0, 1, size_y, dtype=torch.float)
    grid_y = grid_y.reshape(1, 1, size_y, 1).repeat([b, size_x, 1, 1])
    return torch.cat((grid_x, grid_y), dim=-1).to(device)