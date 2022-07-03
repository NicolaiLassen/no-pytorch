'''
    @author: Zongyi Li
    https://github.com/zongyi-li/fourier_neural_operator/blob/74b1572d4e02f215728b4aa5bf46374ed7daba06/utilities3.py
'''

import torch
import torch.nn as nn

class LpLoss(object):
    def __init__(self, d=2, p=2, size_average=True, reduction=True):
        super(LpLoss, self).__init__()

        # dimension and Lp-norm type are postive
        assert d > 0 and p > 0

        self.d = d
        self.p = p
        self.reduction = reduction
        self.size_average = size_average
    
    @staticmethod
    def _noise(targets: torch.Tensor, n_targets: int, noise=0.0):
        assert 0 <= noise <= 0.2
        with torch.no_grad():
            targets = targets * (1.0 + noise*torch.rand_like(targets))
        return targets

    def abs(self, x, y):
        num_examples = x.size()[0]

        # assume uniform mesh
        h = 1.0 / (x.size()[1] - 1.0)

        all_norms = (h**(self.d/self.p))*torch.norm(x.view(num_examples,-1) - y.view(num_examples,-1), self.p, 1)

        if self.reduction:
            if self.size_average:
                return torch.mean(all_norms)
            else:
                return torch.sum(all_norms)

        return all_norms

    def rel(self, x, y):
        num_examples = x.size()[0]

        diff_norms = torch.norm(x.reshape(num_examples,-1) - y.reshape(num_examples,-1), self.p, 1)
        y_norms = torch.norm(y.reshape(num_examples,-1), self.p, 1)

        if self.reduction:
            if self.size_average:
                return torch.mean(diff_norms/y_norms)
            else:
                return torch.sum(diff_norms/y_norms)

        return diff_norms/y_norms

    def __call__(self, x, y):
        return self.rel(x, y)
    
class HsLoss(object):
    def __init__(self, d=2, p=2, k=1, a=None, group=False, size_average=True, reduction=True):
        super(HsLoss, self).__init__()

        # dimension and Lp-norm type are postive
        assert d > 0 and p > 0

        self.d = d
        self.p = p
        self.k = k
        self.balanced = group
        self.reduction = reduction
        self.size_average = size_average

        if a == None:
            a = [1,] * k
        self.a = a
        
    @staticmethod
    def _noise(targets: torch.Tensor, n_targets: int, noise=0.0):
        assert 0 <= noise <= 0.2
        with torch.no_grad():
            targets = targets * (1.0 + noise*torch.rand_like(targets))
        return targets

    def rel(self, x, y):
        num_examples = x.size()[0]
        diff_norms = torch.norm(x.reshape(num_examples,-1) - y.reshape(num_examples,-1), self.p, 1)
        y_norms = torch.norm(y.reshape(num_examples,-1), self.p, 1)
        if self.reduction:
            if self.size_average:
                return torch.mean(diff_norms/y_norms)
            else:
                return torch.sum(diff_norms/y_norms)
        return diff_norms/y_norms

    def __call__(self, x, y, a=None):
        nx = x.size()[1]
        ny = x.size()[2]
        k = self.k
        balanced = self.balanced
        a = self.a
        x = x.view(x.shape[0], nx, ny, -1)
        y = y.view(y.shape[0], nx, ny, -1)

        k_x = torch.cat((torch.arange(start=0, end=nx//2, step=1),torch.arange(start=-nx//2, end=0, step=1)), 0).reshape(nx,1).repeat(1,ny)
        k_y = torch.cat((torch.arange(start=0, end=ny//2, step=1),torch.arange(start=-ny//2, end=0, step=1)), 0).reshape(1,ny).repeat(nx,1)
        k_x = torch.abs(k_x).reshape(1,nx,ny,1).to(x.device)
        k_y = torch.abs(k_y).reshape(1,nx,ny,1).to(x.device)

        x = torch.fft.fftn(x, dim=[1, 2])
        y = torch.fft.fftn(y, dim=[1, 2])

        if balanced==False:
            weight = 1
            if k >= 1:
                weight += a[0]**2 * (k_x**2 + k_y**2)
            if k >= 2:
                weight += a[1]**2 * (k_x**4 + 2*k_x**2*k_y**2 + k_y**4)
            weight = torch.sqrt(weight)
            loss = self.rel(x*weight, y*weight)
        else:
            loss = self.rel(x, y)
            if k >= 1:
                weight = a[0] * torch.sqrt(k_x**2 + k_y**2)
                loss += self.rel(x*weight, y*weight)
            if k >= 2:
                weight = a[1] * torch.sqrt(k_x**4 + 2*k_x**2*k_y**2 + k_y**4)
                loss += self.rel(x*weight, y*weight)
            loss = loss / (k+1)

        return loss
    
# TODO
class WeightedL2Loss2d(nn.Module):
    def __init__(self,
                 dim=2,
                 dilation=2,  # central diff
                 regularizer=False,
                 h=1/421,  # mesh size
                 beta=1.0,  # L2 u
                 gamma=1e-1,  # \|D(N(u)) - Du\|,
                 alpha=0.0,  # L2 \|N(Du) - Du\|,
                 delta=0.0,  #
                 metric_reduction='L1',
                 return_norm=True,
                 noise=0.0,
                 eps=1e-10,
                 debug=False
                 ):
        super(WeightedL2Loss2d, self).__init__()
        self.noise = noise
        self.regularizer = regularizer
        assert dilation % 2 == 0
        self.dilation = dilation
        self.dim = dim
        self.h = h
        self.beta = beta  # L2
        self.gamma = gamma  # H^1
        self.alpha = alpha  # H^1
        self.delta = delta*h**dim  # orthogonalizer
        self.eps = eps
        self.metric_reduction = metric_reduction
        self.return_norm = return_norm
        self.debug = debug

    @staticmethod
    def _noise(targets: torch.Tensor, n_targets: int, noise=0.0):
        assert 0 <= noise <= 0.2
        with torch.no_grad():
            targets = targets * (1.0 + noise*torch.rand_like(targets))
        return targets

    def central_diff(self, u: torch.Tensor, h=None):
        '''
        u: function defined on a grid (bsz, n, n)
        out: gradient (N, n-2, n-2, 2)
        '''
        bsz = u.size(0)
        h = self.h if h is None else h
        d = self.dilation  # central diff dilation
        s = d // 2  # central diff stride
        if self.dim > 2:
            raise NotImplementedError(
                "Not implemented: dim > 2 not implemented")

        grad_x = (u[:, d:, s:-s] - u[:, :-d, s:-s])/d
        grad_y = (u[:, s:-s, d:] - u[:, s:-s, :-d])/d
        grad = torch.stack([grad_x, grad_y], dim=-1)
        return grad/h

    def forward(self, preds, targets,
                preds_prime=None, targets_prime=None,
                weights=None, K=None):
        r'''
        preds: (N, n, n, 1)
        targets: (N, n, n, 1)
        targets_prime: (N, n, n, 1)
        K: (N, n, n, 1)
        beta * \|N(u) - u\|^2 + \alpha * \| N(Du) - Du\|^2 + \gamma * \|D N(u) - Du\|^2
        weights has the same shape with preds on nonuniform mesh
        the norm and the error norm all uses mean instead of sum to cancel out the factor
        '''
        batch_size = targets.size(0) # for debug only

        h = self.h if weights is None else weights
        d = self.dim
        K = torch.tensor(1) if K is None else K
        if self.noise > 0:
            targets = self._noise(targets, targets.size(-1), self.noise)

        target_norm = targets.pow(2).mean(dim=(1, 2)) + self.eps

        if targets_prime is not None:
            targets_prime_norm = d * \
                (K*targets_prime.pow(2)).mean(dim=(1, 2, 3)) + self.eps
        else:
            targets_prime_norm = 1

        loss = self.beta*((preds - targets).pow(2)).mean(dim=(1, 2))/target_norm

        if preds_prime is not None and self.alpha > 0:
            grad_diff = (K*(preds_prime - targets_prime)).pow(2)
            loss_prime = self.alpha * \
                grad_diff.mean(dim=(1, 2, 3))/targets_prime_norm
            loss += loss_prime

        if self.metric_reduction == 'L2':
            metric = loss.mean().sqrt().item()
        elif self.metric_reduction == 'L1':  # Li et al paper: first norm then average
            metric = loss.sqrt().mean().item()
        elif self.metric_reduction == 'Linf':  # sup norm in a batch
            metric = loss.sqrt().max().item()

        loss = loss.sqrt().mean() if self.return_norm else loss.mean()

        if self.regularizer and targets_prime is not None:
            preds_diff = self.central_diff(preds)
            s = self.dilation // 2
            targets_prime = targets_prime[:, s:-s, s:-s, :].contiguous()

            if K.ndim > 1:
                K = K[:, s:-s, s:-s].contiguous()

            regularizer = self.gamma * h * ((K * (targets_prime - preds_diff))
                                            .pow(2)).mean(dim=(1, 2, 3))/targets_prime_norm

            regularizer = regularizer.sqrt().mean() if self.return_norm else regularizer.mean()

        else:
            regularizer = torch.tensor(
                [0.0], requires_grad=True, device=preds.device)
        norms = dict(L2=target_norm,
                     H1=targets_prime_norm)

        return loss, regularizer, metric, norms
    
class WeightedL2Loss(nn.Module):
    """Some Information about WeightedL2Loss"""
    def __init__(self):
        super(WeightedL2Loss, self).__init__()

    def forward(self, x):

        return x