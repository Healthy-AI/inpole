import numpy as np
from amhelpers.amhelpers import seed_hash

from .. import OTHER_ESTIMATORS


# @TODO: Add hyperparameters for the following estimators and datasets:
# - rulefit
# - fasterrisk


def _hparams(estimator_name, experiment, seed):
    hparams = {}

    def _hparam(name, default_val, random_val_fn):
        assert(name not in hparams)
        random_state = np.random.RandomState(seed_hash(seed, name))
        random_val = random_val_fn(random_state)
        if np.issubdtype(type(random_val), np.integer):
            random_val = int(random_val)
        hparams[name] = (default_val, random_val)

    # =========================================================================
    # Experiment-INDEPENDENT parameters.
    # =========================================================================
    
    if estimator_name in ['sdt', 'rdt', 'truncated_rdt']:
        _hparam('initial_depth', 2, lambda r: r.choice([1, 2]))
        _hparam('lambda_', 1.0e-3, lambda r: 10. ** r.choice([-3, -2, -1]))
        _hparam('module__prediction', 'max', lambda r: r.choice(['max', 'mean']))
        _hparam('max_depth', 3, lambda r: r.choice([3, 4, 5]))
    
    if estimator_name in ['rdt', 'truncated_rdt']:
        _hparam('delta1', 1.0e-3, lambda r: 10. ** r.choice([-3, -2, -1]))
        _hparam('delta2', 1.0e-3, lambda r: 10. ** r.choice([-3, -2, -1]))
        _hparam('module__hidden_dim', 10, lambda r: r.choice([5, 10, 15, 20]))
    
    if estimator_name in ['pronet', 'prosenet', 'truncated_prosenet']:
        _hparam('d_min', 1, lambda r: r.choice([1, 2, 3, 4, 5]))
        _hparam('lambda_div', 1.0e-3, lambda r: 10. ** r.choice([-5, -4, -3, -2, -1]))
        
    if estimator_name in ['mlp', 'pronet']:
        hidden_dims = np.array([(16,), (32,), (64,), (16, 16), (32, 32), (64, 64)], dtype=object)
        _hparam('module__encoder__hidden_dims', (32,), lambda r: r.choice(hidden_dims))
        _hparam('module__encoder__output_dim', 32, lambda r: r.choice([16, 32, 64]))
    
    if estimator_name in ['rnn', 'prosenet', 'truncated_rnn', 'truncated_prosenet']:
        _hparam('module__encoder__output_dim', 32, lambda r: r.choice([16, 32, 64]))
        _hparam('module__encoder__num_layers', 1, lambda r: r.choice([1, 2]))
    
    if estimator_name == 'lr':
        _hparam('penalty', 'l2', lambda r: 'l2')
        _hparam('C', 1.0, lambda r: 10. ** r.choice([-3, -2, -1, 0, 1, 1, 2, 3]))
        _hparam('max_iter', 2000, lambda r: 2000)
    
    if estimator_name == 'dt':
        _hparam('max_depth', None, lambda r: r.choice([3, 5, 7, 9, 11, 13, 15]))
    
    if estimator_name == 'frl':
        _hparam('minsupport', 10, lambda r: r.choice([8, 10, 12, 14, 16, 18, 20]))
        _hparam('max_predicates_per_ant', 2, lambda r: r.choice([2, 3, 4, 5]))
        _hparam('w', 5, lambda r: r.choice([5, 6, 7]))
        _hparam('T', 3000, lambda r: r.choice([2000, 3000, 4000, 5000]))
        _hparam('lambda_', 0.8, lambda r: r.choice([0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]))

    if estimator_name == 'riskslim':
        _hparam('max_coefficient', 5, lambda r: r.choice([3, 4, 5, 6, 7, 8, 9, 10]))
        _hparam('max_L0_value', 5, lambda r: r.choice([3, 4, 5, 6, 7, 8, 9]))
        _hparam('max_offset', 50, lambda r: r.choice([50, 70, 90, 110, 130]))
        _hparam('w_pos', 1, lambda r: r.choice([3, 4, 5, 6, 7]))

    # =========================================================================
    # Experiment-DEPENDENT parameters (RA/Switch).
    # =========================================================================

    if experiment in ['ra', 'switch'] and estimator_name not in OTHER_ESTIMATORS:
        _hparam('lr', 1.0e-3, lambda r: 10. ** r.choice([-3, -2]))
        _hparam('max_epochs', 50, lambda r: 50)
        _hparam('batch_size', 32, lambda r: r.choice([32, 64]))
    
    if experiment == 'ra' and estimator_name in [
        'pronet', 'prosenet', 'truncated_prosenet'
    ]:
        _hparam('module__num_prototypes', 10, lambda r: r.choice([10, 20, 30]))
    
    if experiment == 'switch' and estimator_name in [
        'pronet', 'prosenet', 'truncated_prosenet'
    ]:
        _hparam('module__num_prototypes', 4, lambda r: r.choice([2, 4, 6, 8, 10]))

    # =========================================================================
    # Experiment-DEPENDENT parameters (ADNI).
    # =========================================================================
    
    if experiment == 'adni' and estimator_name not in OTHER_ESTIMATORS:
        _hparam('lr', 1.0e-3, lambda r: 10. ** r.choice([-3, -2]))
        _hparam('max_epochs', 20, lambda r: 20)
        _hparam('batch_size', 32, lambda r: r.choice([16, 32, 64]))

    if experiment == 'adni' and estimator_name in [
        'pronet', 'prosenet', 'truncated_prosenet'
    ]:
        _hparam('module__num_prototypes', 4, lambda r: r.choice([2, 4, 6, 8, 10]))
    
    # =========================================================================
    # Experiment-DEPENDENT parameters (Sepsis).
    # =========================================================================
    
    if experiment == 'sepsis' and estimator_name not in OTHER_ESTIMATORS:
        _hparam('lr', 1.0e-3, lambda r: 10. ** r.choice([-3, -2]))
        _hparam('max_epochs', 500, lambda r: 500)

    if experiment == 'sepsis' and estimator_name in ['sdt', 'mlp', 'pronet']:
        _hparam('batch_size', 512, lambda r: r.choice([256, 512, 1024]))

    if experiment == 'sepsis' and estimator_name in ['rdt', 'rnn', 'prosenet']:
        _hparam('batch_size', 32, lambda r: r.choice([16, 32, 64]))

    if experiment == 'sepsis' and estimator_name in [
        'truncated_rdt', 'truncated_rnn', 'truncated_prosenet'
    ]:
        _hparam('batch_size', 128, lambda r: r.choice([64, 128, 256]))

    if experiment == 'sepsis' and estimator_name in [
        'pronet', 'prosenet', 'truncated_prosenet'
    ]:
        _hparam('module__num_prototypes', 10, lambda r: r.choice([5, 10, 15, 20, 25]))

    return hparams

    
def default_hparams(estimator_name, experiment):
    return {a: b for a, (b, c) in _hparams(estimator_name, experiment, 0).items()}


def random_hparams(estimator_name, experiment, seed):
    return {a: c for a, (b, c) in _hparams(estimator_name, experiment, seed).items()}
