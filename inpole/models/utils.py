import torch
import copy
import joblib
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import sklearn.pipeline as pipeline
from matplotlib.ticker import MaxNLocator
from skorch import NeuralNetClassifier
from os.path import join, isfile
from ..data.utils import (
    SequentialDataset,
    get_data_splits_from_config
)
from amhelpers.amhelpers import (
    get_class_from_str,
    create_object_from_dict
)
from amhelpers.config_parsing import get_net_params, _check_value
from sklearn.utils import _print_elapsed_time


NET_SETTINGS = [
    'sdt',
    'rdt',
    'pronet',
    'prosenet'
]


def expects_groups(estimator):
    if estimator.__class__.__name__ == 'SwitchPropensityEstimator':
        return all([expects_groups(e) for e in estimator.estimators])
    return (
        hasattr(estimator, 'dataset') and
        estimator.dataset == SequentialDataset
    )


def create_estimator(config, setting, input_dim=None, output_dim=None):
    config = copy.deepcopy(config)
    
    estimator_dict = config[setting]['estimator']
    estimator_dict.pop('is_called', None)
    
    is_switch_estimator = \
        estimator_dict['type'].endswith('SwitchPropensityEstimator')

    if is_switch_estimator:
        estimator_params = {}
        for k, v in estimator_dict.items():
            if k == 'estimator_s' and (v in config):
                # If the switch estimator is a neural network, we
                # assume that the cross-entropy loss is used,
                # which requires the output dimension to be 2.
                v = create_estimator(config, v, input_dim, 2)
            elif k == 'estimator_t' and (v in config):
                v = create_estimator(config, v, input_dim, output_dim)
            else:
                v = _check_value(v)
            estimator_params[k] = v
    else:
        estimator_params = {
            k: _check_value(v) for k, v in estimator_dict.items()
        }
    
    net_params = get_net_params(config['default'], config[setting])
    if input_dim is not None:
        net_params['module__input_dim'] = input_dim
    if output_dim is not None:
        net_params['module__output_dim'] = output_dim
    net_params['results_path'] = config['results']['path']

    try:
        return create_object_from_dict(estimator_params | net_params)
    except TypeError:
        # Not a skorch estimator.
        return create_object_from_dict(estimator_params)


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


def _create_net_pipeline(config, setting):
    # Use training data to infer input/output dimensions.
    X, y = get_data_splits_from_config(config)[0]
    
    data_handler = get_class_from_str(config['data']['handler'])
    preprocessor = data_handler.get_preprocessor()
    preprocessor.fit(X)

    input_dim = preprocessor.transform(X).shape[1] - 1
    output_dim = len(set(y))
    
    estimator = create_estimator(config, setting, input_dim, output_dim)
    
    #if load_parameters:
    #    f_preprocessor = join(
    #        config['results']['path'],
    #        'preprocessor.pkl'
    #    )
    #    preprocessor = joblib.load(f_preprocessor)
    #   estimator = load_net_parameters(estimator)

    steps = [
        ('preprocessor', preprocessor),
        ('estimator', estimator)
    ]
    return Pipeline(steps)


def create_pipeline(config, setting, load=False):
    if load:
        f_pipeline = join(
            config['results']['path'],
            'model.pkl'
        )
        return joblib.load(f_pipeline)
    
    if setting in NET_SETTINGS or 'switch' in setting:
        return _create_net_pipeline(config, setting)

    data_handler = get_class_from_str(config['data']['handler'])
    preprocessor = data_handler.get_preprocessor()

    estimator = create_estimator(config, setting)
    
    steps = [
        ('preprocessor', preprocessor),
        ('estimator', estimator)
    ]
    return Pipeline(steps)


def extract_validation_data_from_model(model, data_train):
    preprocessor, classifier = model.named_steps.values()
    if not isinstance(classifier, NeuralNetClassifier):
        raise ValueError(
            "Validation data can only be extracted from an "
            "instance of `skorch.NeuralNetClassifier`. Got "
            f"{type(classifier).__name__} instead."
        )
    assert classifier.train_split is not None
    X, y = data_train
    if not expects_groups(classifier):
        X = X.iloc[:, :-1]
    X = preprocessor.transform(X)
    _, dataset_valid = classifier.get_split_datasets(X, y)
    y = classifier.collect_labels_from_dataset(dataset_valid)
    return dataset_valid, y


def load_net_parameters(net):
    if not net.initialized_:
        net.initialize()

    results_path = net.results_path
    f_params = join(results_path, 'best_params.pt')
    f_optimizer = join(results_path, 'best_optimizer.pt')
    f_criterion = join(results_path, 'best_criterios.pt')
    f_history = join(results_path, 'best_history.json')

    if not isfile(f_params): f_params = None
    if not isfile(f_optimizer): f_optimizer = None
    if not isfile(f_criterion): f_criterion = None
    if not isfile(f_history): f_history = None

    net.load_params(f_params, f_optimizer, f_criterion, f_history)
    
    classes = net.history[0].pop('classes_', None)
    if isinstance(classes, list):
        classes = np.array(classes)
    if classes is not None:
        net.classes = classes

    return net


def plot_stats(history, ykey, from_batches, ax, **kwargs):
    x, y = [], []
    for h in history:
        if from_batches:
            for hi in h['batches']:
                if ykey in hi:
                     x.append(h['epoch'])
                     y.append(hi[ykey])
        else:
            if ykey in h:
                x.append(h['epoch'])
                y.append(h[ykey])
    no_valid_y = np.isnan(y).all() or np.isinf(y).all()
    if len(y) > 0 and not no_valid_y:
        sns.lineplot(x=x, y=y, ax=ax, **kwargs)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))


def save_fig(fig, file):
    fig.savefig(file, bbox_inches='tight')
    plt.close(fig)


def plot_save_stats(history, keys, path, name, from_batches=True):
    if not isinstance(keys, list):
        keys = [keys]
    fig, ax = plt.subplots()
    for key in keys:
        plot_stats(history, key, from_batches, ax, label=key)
    ax.legend()
    file = join(path, name + '.pdf')
    save_fig(fig, file)
