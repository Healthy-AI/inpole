import numpy as np

from ..utils import seed_hash


def _hparams(experiment, seed):
    hparams = {}

    def _hparam(name, default_val, random_val_fn):
        assert(name not in hparams)
        random_state = np.random.RandomState(seed_hash(seed, name))
        hparams[name] = (default_val, random_val_fn(random_state))

    if experiment == 'ra':
        pass
    
    return hparams


def default_hparams(experiment):
    return {a: b for a, (b, c) in _hparams(experiment, 0).items()}


def random_hparams(experiment, seed):
    return {a: c for a, (b, c) in _hparams(experiment, seed).items()}
