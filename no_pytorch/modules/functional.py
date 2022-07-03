import torch
import torch.nn.functional as F

def central_2d_difference(x, h, s=(1, 2), padding=True, dilation=2, stride=1):
    """
        b w c
    """
    # TODO
    if padding:
            x = F.pad(x, ((0, 0), (1, 1), (1, 1), (0, 0)), "constant", 0)
    
    grad_x = (x[:, dilation:, stride:-stride, :] - x[:, :-dilation, stride:-stride, :])/dilation # (N, S_x, S_y, t)
    grad_y = (x[:, stride:-stride, dilation:, :] - x[:, stride:-stride, :-dilation, :])/dilation # (N, S_x, S_y, t)

    return grad_x/h, grad_y/h

def central_1d_difference(x, d=1, dilation=2):
    """
        b w c
    """
    d = dilation
    grad = (x[:, d:, :] - x[:, :-d, :])/d
    return grad

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

def get_laplacian_1d(node,
                     K=None,
                     weight=None,  # weight for renormalization
                     normalize=True,
                     smoother=None):
    '''
    Construct the 1D Laplacian matrix on the domain defined by node. 
    with a variable mesh size.

    Input:
        - node: array-like, shape (N, ) One dimensional mesh; or a positve integer.
        - normalize: apply D^{-1/2} A D^{-1/2} row and column scaling to the Laplacian 

    Output:
        - A : scipy sparse matrix, shape (N, N)
        Laplacian matrix.

    Reference:
        Code adapted to 1D from the 2D one in 
        Long Chen: iFEM: An innovative finite element method package in Matlab. 
        Technical report, University of California-Irvine, 2009
    '''
    if isinstance(node, int):
        node = np.linspace(0, 1, node)
    N = node.shape[0]
    h = node[1:] - node[:-1]
    elem = np.c_[np.arange(N-1), np.arange(1, N)]
    Dphi = np.c_[-1/h, 1/h]

    if K is None:
        K = 1

   # stiffness matrix
    A = csr_matrix((N, N))
    for i in range(2):
        for j in range(2):
            # $A_{ij}|_{\tau} = \int_{\tau}K\nabla \phi_i\cdot \nabla \phi_j dxdy$
            Aij = h*K*Dphi[:, i]*Dphi[:, j]
            A += csr_matrix((Aij, (elem[:, i], elem[:, j])), shape=(N, N))

    if weight is not None:
        A += diags(weight)

    if normalize:
        D = diags(A.diagonal()**(-0.5))
        A = (D.dot(A)).dot(D)

        if smoother == 'jacobi':
            I = identity(N)
            A = I-(2/3)*A  # jacobi
            A = csr_matrix(A)
        elif smoother == 'gs':
            raise NotImplementedError("Gauss-seidel not implemented")

    return A