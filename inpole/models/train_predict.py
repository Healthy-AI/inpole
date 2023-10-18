import warnings
import pandas as pd
import numpy as np
from os.path import join
from amhelpers.amhelpers import get_class_from_str
from .utils import (
    create_pipeline,
    extract_validation_data_from_model,
    expects_groups
)
from joblib import dump
from .models import SwitchPropensityEstimator
from skorch import NeuralNetClassifier


def _get_previous_therapy_index(feature_names, prev_therapy_prefix):
    feature_names = np.array([s.split('__')[1] for s in feature_names])
    prev_therapy_columns = [s for s in feature_names if s.startswith(prev_therapy_prefix)]
    prev_therapy_index = np.array(
        [np.flatnonzero(feature_names == c).item() for c in prev_therapy_columns]
    )
    return prev_therapy_index


def train_model(config, setting):
    model = create_pipeline(config, setting)
    estimator = model.named_steps['estimator']

    data_handler = get_class_from_str(config['data']['handler'])
    data_train, data_valid, _ = data_handler.get_splits(
        config['data']['path'],
        config['data']['valid_size'],
        config['data']['test_size'],
        config['data']['seed']
    )

    X, y = data_train
    X_valid, y_valid = data_valid

    if not expects_groups(estimator):
        X = X.drop(columns=data_handler.GROUP)
        X_valid = X_valid.drop(columns=data_handler.GROUP)

    fit_params = {}
    
    if (
        isinstance(estimator, NeuralNetClassifier) or
        (
            isinstance(estimator, SwitchPropensityEstimator) and
            all([isinstance(e, NeuralNetClassifier) for e in estimator.estimators])
        )
    ):
        fit_params['estimator__X_valid'] = X_valid
        fit_params['estimator__y_valid'] = y_valid
    
    if isinstance(estimator, SwitchPropensityEstimator):
        model.named_steps['preprocessor'].fit(X)
        feature_names = model.named_steps['preprocessor'].get_feature_names_out()
        prefix = data_handler.TREATMENT + '_1_'
        prev_therapy_index = _get_previous_therapy_index(feature_names, prefix)
        fit_params['estimator__prev_therapy_index'] = prev_therapy_index

    model.fit(X, y, **fit_params)

    # Save model.
    f_model = join(
        config['results']['path'],
        'model.pkl'
    )
    dump(model, f_model)

    return model


def collect_scores(model, X, y, metrics, columns=[], data=[]):
    for metric in metrics:
        score = model.score(X, y, metric=metric)
        class_scores = model.score(X, y, metric=metric, average=None)
        if isinstance(class_scores, float):
            class_scores = len(model.classes_) * [float('nan')]
        data.append(score)
        data.extend(class_scores)
        suffixes = ['average'] + model.classes_.tolist()
        for suffix in suffixes:
            column = f'{metric}_{suffix}'
            columns.append(column)
    return pd.Series(data=data, index=columns)


def predict_model(
    config,
    setting,
    model,
    subset,
    metrics
):    
    data_handler = get_class_from_str(config['data']['handler'])
    data_train, data_valid, data_test = data_handler.get_splits(
        config['data']['path'],
        config['data']['valid_size'],
        config['data']['test_size'],
        config['data']['seed']
    )

    if subset == 'valid':
        X, y = data_valid
        if X is None and y is None:
            X, y = extract_validation_data_from_model(model, data_train)
    elif subset == 'test':
        X, y = data_test
    
    if (
        isinstance(X, pd.DataFrame) and
        not expects_groups(model.named_steps['estimator'])
    ):
        X = X.drop(columns=data_handler.GROUP)
    
    columns = ['setting', 'subset']
    data = [setting, subset]
    scores = collect_scores(model, X, y, metrics, columns, data)

    scores_file = join(config['results']['path'], 'scores.csv')
    try:
        df = pd.read_csv(scores_file)
        df = pd.concat([df, scores.to_frame().T], ignore_index=True)
        df.to_csv(scores_file, index=False)
    except FileNotFoundError:
        df = pd.DataFrame(scores)
        df.T.to_csv(scores_file, index=False)
