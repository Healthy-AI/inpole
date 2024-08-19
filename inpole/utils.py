import os
import sys

import torch
import pandas as pd


def _print_log(s):
    print(s)
    sys.stdout.flush()


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


def merge_dicts(dicts):
    """Merge dictionaries.

    Parameters
    ----------
    dicts : list of dictionaries
        All dictionaries should have identical keys (not checked).

    Returns
    -------
    dict
        A single dictionary where each key maps to a list of values from the
        dictionaries in `dicts`.
    """
    if not isinstance(dicts, list):
        raise ValueError(f"`dicts` must be a list, got {type(dicts).__name__}.")
    
    if len(dicts) == 0:
        return {}

    if not all(isinstance(d, dict) for d in dicts):
        raise ValueError("All elements of `dicts` must be a dictionary.")

    keys = dicts[0].keys()
    values = [[d[k] for d in dicts] for k in keys]

    return dict(zip(keys, values))


def get_index_per_time_step(groups):
    n = len(groups)
    index = pd.RangeIndex.from_range(range(n))
    index_per_group = index.groupby(groups).values()
    num_time_steps = max(map(len, index_per_group))
    for t in range(num_time_steps):
        index_t = [v[t] for v in index_per_group if len(v) > t]
        yield index_t
