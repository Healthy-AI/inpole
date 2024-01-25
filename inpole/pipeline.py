import copy
import os
from os.path import join

import torch
import joblib
import pandas as pd
import sklearn.pipeline as pipeline
from sklearn.utils import _print_elapsed_time
from amhelpers.amhelpers import seed_hash

from .models import hparam_registry as hprm
from .models import SwitchPropensityEstimator
from .data.utils import *
from .data import get_data_handler_from_config
from . import (
    NET_ESTIMATORS,
    RECURRENT_NET_ESTIMATORS,
    OTHER_ESTIMATORS
)


ALL_NET_ESTIMATORS = NET_ESTIMATORS | RECURRENT_NET_ESTIMATORS


_default_net_params = {
    #module
    'criterion': torch.nn.CrossEntropyLoss,
    'optimizer': torch.optim.Adam,
    #lr
    #max_epochs
    #batch_size
    'iterator_train': torch.utils.data.DataLoader,
    'iterator_train__shuffle': True,
    'iterator_valid': torch.utils.data.DataLoader,
    'dataset': StandardDataset,
    'train_split': None,
    'callbacks': None,
    'predict_nonlinearity': 'auto',
    'warm_start': False,
    'verbose': 1,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu'
}


def _get_estimator_params(config, estimator_name, input_dim=None, output_dim=None):
    experiment = config['experiment']
    hparams_seed = config['hparams']['seed']
    if hparams_seed == 0:
        hparams = hprm.default_hparams(estimator_name, experiment)
    else:
        data_seed = config['data']['seed']
        seed = seed_hash(hparams_seed, data_seed)
        hparams = hprm.random_hparams(estimator_name, experiment, seed)
    
    if estimator_name in ALL_NET_ESTIMATORS:
        params = copy.deepcopy(_default_net_params)        
        params.update(
            {
                'results_path': config['results']['path'],
                'seed': config['estimators']['seed'],
                'module__input_dim': input_dim,
                'module__output_dim': output_dim,
            }
        )
    else:
        params = {'random_state': config['estimators']['seed']}
    
    params.update(hparams)

    if estimator_name in config['estimators']:
        params.update(config['estimators'][estimator_name])

    return params


def _create_estimator(
    config,
    estimator_name,
    **kwargs
):
    params = _get_estimator_params(config, estimator_name, **kwargs)

    if estimator_name in NET_ESTIMATORS:
        return NET_ESTIMATORS[estimator_name](**params)
    
    if estimator_name in RECURRENT_NET_ESTIMATORS:
        if estimator_name.startswith('truncated'):
            params['dataset'] = TruncatedHistoryDataset
            params['dataset__periods'] = config['data']['shift_periods']
        else:
            params['dataset'] = SequentialDataset
        params['iterator_train__collate_fn'] = pad_pack_sequences
        params['iterator_valid__collate_fn'] = pad_pack_sequences
        return RECURRENT_NET_ESTIMATORS[estimator_name](**params)
    
    if estimator_name in OTHER_ESTIMATORS:
        return OTHER_ESTIMATORS[estimator_name](**params)


class Pipeline(pipeline.Pipeline):
    def __init__(self, steps, *, memory=None, verbose=False):
        super().__init__(steps, memory=memory, verbose=verbose)

    def fit(self, X, y=None, **fit_params):
        fit_params_steps = self._check_fit_params(**fit_params)
        Xt = self._fit(X, y, **fit_params_steps)
        with _print_elapsed_time('Pipeline', self._log_message(len(self.steps) - 1)):
            if self._final_estimator != 'passthrough':
                fit_params_last_step = fit_params_steps[self.steps[-1][0]]
                Xt_valid = fit_params_last_step.get('X_valid', None)
                if Xt_valid is not None:
                    for _, _, transform in self._iter(with_final=False):
                        Xt_valid = transform.transform(Xt_valid)
                    fit_params_last_step['X_valid'] = Xt_valid
                self._final_estimator.fit(Xt, y, **fit_params_last_step)
        return self
    
    def score(self, X, y=None, **score_params):
        estimator = self.steps[-1][1]
        if isinstance(X, torch.utils.data.Dataset):
            return estimator.score(X, y, **score_params)
        for _, _, transform in self._iter(with_final=False):
            X = transform.transform(X)
        return estimator.score(X, y, **score_params)


def create_pipeline(
    config,
    estimator_name
):
    data_handler = get_data_handler_from_config(config)

    # @TODO: Should this seed depend on the estimator?
    preprocessor = data_handler.get_preprocessor(config['hparams']['seed'])

    is_switch_estimator = False
    if estimator_name.startswith('switch'):
        estimator_name = estimator_name.split('_')[1]
        is_switch_estimator = True

    if estimator_name in ALL_NET_ESTIMATORS:
        # Infer input/output dimensions from training data.
        X_train, y_train = data_handler.get_splits()[0]
        if estimator_name.startswith('truncated'):
            c_shifted = get_shifted_column_names(X_train)
            X_train = X_train.drop(columns=c_shifted)
        preprocessor.fit(X_train, y_train)
        input_dim = len(preprocessor.get_feature_names_out()) - 1
        output_dim = len(set(y_train))
    else:
        input_dim = output_dim = None
    
    if is_switch_estimator:
        estimator_s = _create_estimator(
            config,
            estimator_name,
            input_dim=input_dim,
            output_dim=2
        )
        estimator_t = _create_estimator(
            config,
            estimator_name,
            input_dim=input_dim,
            output_dim=output_dim
        )
        estimator = SwitchPropensityEstimator(estimator_s, estimator_t)
    else:
        estimator = _create_estimator(
            config,
            estimator_name,
            input_dim=input_dim,
            output_dim=output_dim
        )

    steps = [('preprocessor', preprocessor), ('estimator', estimator)]
    return Pipeline(steps)
    

def load_best_pipeline(
    experiment_path,
    trial,
    estimator_name,
    sweep_parameter_value=None
):
    scores_path = join(experiment_path, 'scores.csv')
    scores = pd.read_csv(scores_path)
    if sweep_parameter_value is None:
        sweep = 'sweep'
        mask = (scores.trial == trial) & \
            (scores.estimator_name == estimator_name)
    else:
        sweep_parameter = scores.columns[0]
        sweep = f'sweep_{sweep_parameter}_{sweep_parameter_value}'
        mask = (scores[sweep_parameter] == sweep_parameter_value) & \
            (scores.trial == trial) & \
            (scores.estimator_name == estimator_name)
    experiment = scores[mask].exp.iat[0]
    results_path = join(experiment_path, sweep, f'trial_{trial:02d}', experiment)
    pipeline_path = join(results_path, 'pipeline.pkl')
    return joblib.load(pipeline_path)


def load_experiment_pipeline(experiment_path, trial, estimator_name):
    pipelines = {}
    trial_path = join(experiment_path, 'sweep', f'trial_{trial:02d}')
    if not os.path.exists(trial_path):
        raise FileNotFoundError(f"No trial directory found for trial {trial}")
    for experiment_dir in sorted(os.listdir(trial_path)):
        if estimator_name in experiment_dir:
            model_pipeline_path = join(trial_path, experiment_dir, 'pipeline.pkl')
            if os.path.exists(model_pipeline_path):
                pipelines[experiment_dir] = joblib.load(model_pipeline_path)
    if not pipelines:
        raise FileNotFoundError(f"No pipelines found for model '{estimator_name}' in trial {trial}")
    return pipelines
