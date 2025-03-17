import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Gumbel

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# The updated version of Multi-Mixup
def multi_mixup(x_sequence, y_sequence, num_classes, beta):
    
    assert x_sequence.shape[0] == y_sequence.shape[0]
    assert x_sequence.shape[0] % num_classes == 0

    batch_size = x_sequence.shape[0] // num_classes
        
    y_list = []
    for i in range(num_classes):
        y = y_sequence[i*batch_size:(i+1)*batch_size]
        assert torch.all(y[0].item() * torch.ones_like(y) == y).item()
        y_list.append(F.one_hot(y, num_classes=num_classes))
    
    x_list = []
    for i in range(num_classes):
        x = x_sequence[i*batch_size:(i+1)*batch_size]
        x_list.append(x)
    
    x_tensor = torch.stack(x_list, dim=0)
    y_tensor = torch.stack(y_list, dim=0)

    if beta > 0.:
        beta_vec = np.full(num_classes, beta)
    else:
        beta_vec = np.ones(num_classes)

    weight = torch.from_numpy(np.random.dirichlet(beta_vec, size=batch_size)).to(x_tensor.dtype).to(device)
    
    x_tensor = torch.permute(x_tensor, (2, 3, 4, 1, 0))
    y_tensor = torch.permute(y_tensor, (2, 1, 0))
    
    mixed_x = torch.sum(weight * x_tensor, dim=-1)
    mixed_y = torch.sum(weight * y_tensor, dim=-1)
    
    mixed_x = torch.permute(mixed_x, (3, 0, 1, 2))
    mixed_y = torch.permute(mixed_y, (1, 0))
    
    
    assert mixed_x.shape[0] == batch_size
    assert mixed_y.shape[0] == batch_size

    return mixed_x, mixed_y


# The old version of Multi-Mixup presented in the conference paper
def old_multi_mixup(x_sequence, y_sequence, num_classes, beta, shuffle_num):

    assert x_sequence.shape[0] == y_sequence.shape[0]
    assert x_sequence.shape[0] % num_classes == 0

    batch_size = x_sequence.shape[0] // num_classes
        
    y_list = []
    for i in range(num_classes):
        y = y_sequence[i*batch_size:(i+1)*batch_size]
        assert y[0].item() * torch.ones_like(y) == y
        y_list.append(F.one_hot(y, num_classes=num_classes))
    
    x_list = []
    for i in range(num_classes):
        x = x_sequence[i*batch_size:(i+1)*batch_size]
        x_list.append(x)

    mixed_x_list = []
    mixed_y_list = []
    for _ in range(shuffle_num):
        
        if beta > 0.:
            beta_vec = np.full(num_classes, beta)
            weight = np.random.dirichlet(beta_vec)
        else:
            beta_vec = np.ones(num_classes)
            weight = np.random.dirichlet(beta_vec)

        mixed_x = 0.
        mixed_y = 0.
        for i in range(num_classes):
            index = torch.randperm(x_list[i].shape[0]).to(device)
            mixed_x += weight[i] * x_list[i][index,:]
            mixed_y += weight[i] * y_list[i][index,:]
        
        mixed_x_list.append(mixed_x)
        mixed_y_list.append(mixed_y)

    mixed_x = torch.cat(mixed_x_list, dim=0)
    mixed_y = torch.cat(mixed_y_list, dim=0)
        
    assert mixed_x.shape[0] == batch_size
    assert mixed_y.shape[0] == batch_size

    return mixed_x.to(device), mixed_y.to(device)


class ConcreteDistributionLoss(nn.Module):
    def __init__(self, reduction='mean'):
        super(ConcreteDistributionLoss, self).__init__()
        self.reduction = reduction
        
    def forward(self, logit, temp, labels):
        
        assert logit.shape == labels.shape
        assert temp.shape[0] == labels.shape[0]

        alpha = torch.exp(logit).clamp(min=1e-8)
        assert torch.all(torch.isfinite(alpha)).item()
        
        lam = F.softplus(temp).clamp(min=1e-8)
        assert torch.all(torch.isfinite(lam)).item()
        
        expanded_lam = lam.expand(labels.shape[0], labels.shape[1])
        sum = torch.sum(alpha * (labels ** (-expanded_lam)), dim=1).unsqueeze(1).expand(labels.shape[0], labels.shape[1])
        if not torch.all(torch.isfinite(sum)).item():
            print(lam)
            print(labels ** (-expanded_lam))
            raise ValueError('"sum" is infinite')
        ln = torch.log(alpha) + (-expanded_lam-1) * torch.log(labels) - torch.log(sum.clamp(min=1e-8))
        
        if self.reduction == 'mean':
            return -1.0 * torch.mean(torch.sum(ln, dim=1), dim=0) - (labels.shape[1] - 1) * torch.mean(torch.log(lam), dim=0)
        elif self.reduction == 'sum':
            return -1.0 * torch.sum(torch.sum(ln, dim=1), dim=0) - (labels.shape[1] - 1) * torch.sum(torch.log(lam), dim=0)
        else:
            raise ValueError('Set reduction as "mean" or "sum".')


def gumbel_softmax_sample(x, lamb, sample_size):
    m = Gumbel(0,1)
    z = torch.stack([m.sample() for _ in range(x.shape[0] * x.shape[1] * sample_size)]).reshape(x.shape[0], x.shape[1], sample_size).to(x.device)
    return F.softmax((torch.log(x[:,:,None]+1e-12) + z)/lamb[:,:,None], dim=1)