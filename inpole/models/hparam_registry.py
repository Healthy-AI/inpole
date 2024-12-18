import numpy as np
from amhelpers.amhelpers import seed_hash

from .. import OTHER_ESTIMATORS


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
        _hparam('module__prediction', 'mean', lambda r: r.choice(['max', 'mean']))
        _hparam('max_depth', 4, lambda r: r.choice([3, 4, 5]))
    
    if estimator_name in ['rdt', 'truncated_rdt']:
        _hparam('delta1', 1.0e-3, lambda r: 10. ** r.choice([-3, -2, -1]))
        _hparam('delta2', 1.0e-3, lambda r: 10. ** r.choice([-3, -2, -1]))
        _hparam('module__hidden_dim', 10, lambda r: r.choice([5, 10, 15, 20]))
    
    if estimator_name in ['pronet', 'prosenet', 'truncated_prosenet']:
        _hparam('d_min', 1, lambda r: r.choice([1, 2, 3, 4, 5]))
        _hparam('lambda_div', 1.0e-3, lambda r: 10. ** r.choice([-5, -4, -3, -2, -1, 0]))
        
    if estimator_name in ['mlp', 'pronet']:
        hidden_dims = np.array([(16,), (32,), (64,), (16, 16), (32, 32), (64, 64)], dtype=object)
        _hparam('module__encoder__hidden_dims', (32,), lambda r: r.choice(hidden_dims))
        _hparam('module__encoder__output_dim', 32, lambda r: r.choice([16, 32, 64]))
    
    if estimator_name in ['rnn', 'lstm', 'prosenet', 'truncated_rnn', 'truncated_prosenet']:
        _hparam('module__encoder__output_dim', 32, lambda r: r.choice([16, 32, 64]))
        _hparam('module__encoder__num_layers', 1, lambda r: r.choice([1, 2]))
    
    if estimator_name == 'lr':
        _hparam('penalty', 'l2', lambda r: 'l2')
        _hparam('C', 1.0, lambda r: 10. ** r.choice([-3, -2, -1, 0, 1, 2, 3]))
        _hparam('max_iter', 2000, lambda r: 2000)
    
    if estimator_name == 'dt':
        _hparam('max_depth', None, lambda r: r.choice([3, 5, 7, 9, 11, 13, 15]))
        _hparam('criterion', 'gini', lambda r: r.choice(['gini', 'entropy']))
        _hparam('min_samples_split', 2, lambda r: r.choice([2, 4, 8, 16, 32, 64, 128]))
    
    if estimator_name == 'frl':
        _hparam('minsupport', 10, lambda r: r.choice([8, 10, 12, 14, 16, 18, 20]))
        _hparam('max_predicates_per_ant', 2, lambda r: r.choice([2, 3, 4, 5]))
        _hparam('w', 4, lambda r: r.choice([3, 4, 5, 6]))

    if estimator_name == 'riskslim':
        _hparam('max_coefficient', 5, lambda r: r.choice([3, 4, 5, 6, 7, 8]))
        _hparam('max_L0_value', 5, lambda r: r.choice([3, 4, 5, 6, 7]))
        _hparam('w_pos', 1, lambda r: r.choice([1, 2, 3, 4, 5]))

    if estimator_name == 'rulefit':
        _hparam('tree_size', 4, lambda r: r.choice([2, 3, 4, 5]))
        _hparam('max_rules', 30, lambda r: r.choice([10, 20, 30, 40, 50, 75, 100]))
        _hparam('model_type', 'rl', lambda r: 'rl')
        _hparam('lin_standardise', False, lambda r: False)
        _hparam('memory_par', 0.1, lambda r: r.choice([0.01, 0.1]))
        
    # =========================================================================
    # Experiment-DEPENDENT parameters (RA).
    # =========================================================================

    if experiment == 'ra' and estimator_name not in OTHER_ESTIMATORS:
        _hparam('lr', 1.0e-3, lambda r: 10. ** r.choice([-3, -2]))
        _hparam('max_epochs', 50, lambda r: 50)
    
    if experiment == 'ra' and estimator_name in ['sdt', 'mlp', 'pronet']:
        _hparam('batch_size', 128, lambda r: r.choice([128, 256]))

    if experiment == 'ra' and estimator_name in ['rdt', 'rnn', 'lstm', 'prosenet']:
        _hparam('batch_size', 32, lambda r: r.choice([32, 64]))

    if experiment == 'ra' and estimator_name in [
        'truncated_rdt', 'truncated_rnn', 'truncated_prosenet'
    ]:
        _hparam('batch_size', 64, lambda r: r.choice([64, 128]))
    
    if experiment == 'ra' and estimator_name in [
        'pronet', 'prosenet', 'truncated_prosenet'
    ]:
        _hparam('module__num_prototypes', 10, lambda r: r.choice([5, 10, 15, 20, 25, 30]))

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
        _hparam('module__num_prototypes', 2, lambda r: r.choice([2, 4, 6, 8, 10]))
    
    # =========================================================================
    # Experiment-DEPENDENT parameters (Sepsis/COPD).
    # =========================================================================
    
    if experiment in ['sepsis', 'copd'] and estimator_name not in OTHER_ESTIMATORS:
        _hparam('lr', 1.0e-3, lambda r: 10. ** r.choice([-4, -3, -2]))
        _hparam('max_epochs', 500, lambda r: 500)

    if experiment in ['sepsis', 'copd'] and estimator_name in ['sdt', 'mlp', 'pronet']:
        _hparam('batch_size', 512, lambda r: r.choice([256, 512, 1024]))

    if experiment in ['sepsis', 'copd'] and estimator_name in ['rdt', 'rnn', 'lstm', 'prosenet']:
        _hparam('batch_size', 32, lambda r: r.choice([16, 32, 64]))

    if experiment in ['sepsis', 'copd'] and estimator_name in [
        'truncated_rdt', 'truncated_rnn', 'truncated_prosenet'
    ]:
        _hparam('batch_size', 128, lambda r: r.choice([64, 128, 256]))

    if experiment in ['sepsis', 'copd'] and estimator_name in [
        'pronet', 'prosenet', 'truncated_prosenet'
    ]:
        _hparam('module__num_prototypes', 10, lambda r: r.choice([5, 10, 15, 20, 25, 30]))

    return hparams

    
def default_hparams(estimator_name, experiment):
    return {a: b for a, (b, c) in _hparams(estimator_name, experiment, 0).items()}


def random_hparams(estimator_name, experiment, seed):
    return {a: c for a, (b, c) in _hparams(estimator_name, experiment, seed).items()}
