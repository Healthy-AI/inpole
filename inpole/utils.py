import os

import torch


def seed_torch(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = False
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    torch.use_deterministic_algorithms(True)


def compute_squared_distances(x1, x2):
    """Compute squared distances using quadratic expansion.
    
    Reference: https://github.com/pytorch/pytorch/pull/25799.
    """
    adjustment = x1.mean(-2, keepdim=True)
    x1 = x1 - adjustment
    x2 = x2 - adjustment  # x1 and x2 should be identical in all dims except -2 at this point

    x1_norm = x1.pow(2).sum(dim=-1, keepdim=True)
    x1_pad = torch.ones_like(x1_norm)
    
    x2_norm = x2.pow(2).sum(dim=-1, keepdim=True)
    x2_pad = torch.ones_like(x2_norm)
    
    x1 = torch.cat([-2. * x1, x1_norm, x1_pad], dim=-1)
    x2 = torch.cat([x2, x2_pad, x2_norm], dim=-1)
    
    return x1.matmul(x2.transpose(-2, -1))
