import numpy as np
from . import NNEncoder, RNNEncoder

from ..utils import seed_hash

def _hparams(estimator_name, experiment, seed):
    hparams = {}

    def _hparam(name, default_val, random_val_fn):
        assert(name not in hparams)
        random_state = np.random.RandomState(seed_hash(seed, name))
        hparams[name] = (default_val, random_val_fn(random_state))

    # =========================================================================
    # Experiment-INDEPENDENT parameters.
    # =========================================================================
    
    if estimator_name in ['sdt', 'rdt']:
        _hparam('initial_depth', 2, lambda r: r.choice([1, 2]))
        _hparam('lambda_', 1.0e-3, lambda r: 10. ** r.choice([-3, -2, -1]))
        _hparam('module__prediction', 'max', lambda r: r.choice(['max', 'mean']))
    
    if estimator_name == 'rdt':
        _hparam('delta1', 1.0e-3, lambda r: 10. ** r.choice([-3, -2, -1]))
        _hparam('delta2', 1.0e-3, lambda r: 10. ** r.choice([-3, -2, -1]))
    
    if estimator_name in ['pronet', 'prosenet']:
        _hparam('d_min', 1, lambda r: r.choice([1, 2, 3, 4, 5]))
        _hparam('lambda_div', 1.0e-3, lambda r: 10. ** r.choice([-5, -4, -3, -2, -1]))
        _hparam('module__num_prototypes', 10, lambda r: r.choice([10, 20, 30]))
        
    if estimator_name == 'pronet':
        _hparam('module__encoder', NNEncoder, lambda r: NNEncoder)
        hidden_dims = np.array([(32,), (64,), (32, 32), (64, 64)], dtype=object)
        _hparam('module__encoder__hidden_dims', (32,), lambda r: r.choice(hidden_dims))
        _hparam('module__encoder__output_dim', 32, lambda r: r.choice([32, 64]))
    
    if estimator_name == 'prosenet':
        _hparam('module__encoder', RNNEncoder, lambda r: RNNEncoder)
        _hparam('module__encoder__output_dim', 32, lambda r: r.choice([32, 64]))
        _hparam('module__encoder__num_layers', 1, lambda r: r.choice([1, 2]))
    
    if estimator_name == 'lr':
        _hparam('penalty', 'l2', lambda r: 'l2')
        _hparam('C', 1.0, lambda r: 10. ** r.choice([-3, -2, -1, 0, 1, 1, 2, 3]))
    
    if estimator_name == 'dt':
        _hparam('max_depth', None, lambda r: r.choice([3, 5, 7, 9, 11, 13, 15]))

    # =========================================================================
    # Experiment-DEPENDENT parameters.
    # =========================================================================

    if experiment == 'ra' and not estimator_name in ['lr', 'dt', 'dummy']:
        _hparam('lr', 1.0e-3, lambda r: 10. ** r.choice([-3, -2]))
        _hparam('max_epochs', 50, lambda r: 50)
        _hparam('batch_size', 32, lambda r: r.choice([32, 64]))
    
    if experiment == 'ra' and estimator_name in ['sdt', 'rdt']:
        _hparam('max_depth', 3, lambda r: r.choice([3, 4, 5]))
    
    if experiment == 'ra' and estimator_name == 'rdt':
        _hparam('module__hidden_dim', 10, lambda r: r.choice([5, 10, 15, 20]))
    
    return hparams

    
def default_hparams(estimator_name, experiment):
    return {a: b for a, (b, c) in _hparams(estimator_name, experiment, 0).items()}


def random_hparams(estimator_name, experiment, seed):
    return {a: c for a, (b, c) in _hparams(estimator_name, experiment, seed).items()}
