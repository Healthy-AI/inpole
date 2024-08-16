from os.path import join

import pandas as pd
import numpy as np

from .models.utils import expects_groups
from .data.utils import (
    get_aggregation_index,
    check_fit_preprocessor,
    get_feature_names
)
from .models import (
    SwitchPropensityEstimator,
    RiskSlimClassifier,
    FasterRiskClassifier,
    FRLClassifier,
    RuleFitClassifier,
    DecisionTreeClassifier,
    CalibratedClassifierCV
)
from .data.data import get_data_handler_from_config
from .data.utils import drop_shifted_columns
from .pipeline import create_pipeline
from . import NET_ESTIMATORS, RECURRENT_NET_ESTIMATORS


ALL_NET_ESTIMATORS = NET_ESTIMATORS | RECURRENT_NET_ESTIMATORS


def is_net_estimator(estimator_name):
    return estimator_name in ALL_NET_ESTIMATORS or (
        estimator_name.startswith('switch') and
        estimator_name.split('_')[-1] in ALL_NET_ESTIMATORS
    )


def _get_previous_therapy_index(feature_names, prev_therapy_prefix):
    prev_therapy_columns = [
        s for s in feature_names 
        if (s.startswith(prev_therapy_prefix) and not 'agg' in s)
    ]
    prev_therapy_index = np.array(
        [np.flatnonzero(feature_names == c).item() for c in prev_therapy_columns]
    )
    return prev_therapy_index


def _separate_switches(preprocessor, treatment, X, y):
    assert isinstance(treatment, str)

    feature_names = get_feature_names(preprocessor)
    prefix = f'prev_{treatment}_'
    prev_therapy_index = _get_previous_therapy_index(feature_names, prefix)
    
    Xt = preprocessor.transform(X)

    y_prev = np.argmax(Xt[:, prev_therapy_index], axis=1)
    switch = (y_prev != y)
    return Xt[switch], y[switch]


def train(config, estimator_name, calibrate=False):
    pipeline = create_pipeline(config, estimator_name)
    preprocessor, estimator = pipeline.named_steps.values()

    data_handler = get_data_handler_from_config(config)

    if data_handler.aggregate_history:
        assert not expects_groups(estimator)

    data_train, data_valid, _ = data_handler.get_splits()
    X_train, y_train = data_train
    X_valid, y_valid = data_valid

    if estimator_name.startswith('truncated'):
        X_train = drop_shifted_columns(X_train)
        X_valid = drop_shifted_columns(X_valid)

    if not expects_groups(estimator) and not data_handler.aggregate_history:
        X_train = X_train.drop(columns=data_handler.GROUP)
        X_valid = X_valid.drop(columns=data_handler.GROUP)

    fit_params = {}

    if data_handler.aggregate_history:
        agg_index = get_aggregation_index(preprocessor, X_train, y_train)
        fit_params['preprocessor__feature_selector__aggregator__agg_index'] = agg_index
    
    if is_net_estimator(estimator_name) or isinstance(estimator, DecisionTreeClassifier):
        fit_params['estimator__X_valid'] = X_valid
        fit_params['estimator__y_valid'] = y_valid
    
    if isinstance(estimator, RiskSlimClassifier):
        feature_names = get_feature_names(preprocessor, X_train, y_train)
        outcome_name = data_handler.TREATMENT
        fit_params['estimator__feature_names'] = feature_names
        fit_params['estimator__outcome_name'] = outcome_name
    
    if isinstance(estimator, FasterRiskClassifier) or isinstance(estimator, RuleFitClassifier):
        feature_names = get_feature_names(preprocessor, X_train, y_train)
        fit_params['estimator__feature_names'] = feature_names
    
    if isinstance(estimator, FRLClassifier):
        preprocessor = check_fit_preprocessor(preprocessor, X_train, y_train)
        fit_params['estimator__preprocessor'] = preprocessor

    if isinstance(estimator, SwitchPropensityEstimator):
        feature_names = get_feature_names(preprocessor, X_train, y_train)
        prefix = f'prev_{data_handler.TREATMENT}_'
        prev_therapy_index = _get_previous_therapy_index(feature_names, prefix)
        fit_params['estimator__prev_therapy_index'] = prev_therapy_index

    pipeline.fit(X_train, y_train, **fit_params)
    
    if calibrate:
        pipeline = CalibratedClassifierCV(pipeline, method='isotonic', cv='prefit')
        pipeline.fit(X_valid, y_valid)
    
    return pipeline


def collect_scores(model, X, y, metrics, columns=[], data=[], labels=None):
    if labels is None:
        labels = model.classes_.tolist()
    for metric, kwargs in metrics:
        score = model.score(X, y, metric=metric, **kwargs)
        if isinstance(score, np.ndarray):
            data.extend(score)
            for suffix in labels:
                columns.append(f'{metric}_{suffix}')
        else:
            data.append(score)
            columns.append(metric)
    return pd.Series(data=data, index=columns)


def predict(
    config,
    pipeline,
    estimator_name,
    subset,
    metrics,
    switches_only=False
):    
    data_handler = get_data_handler_from_config(config)
    _, data_valid, data_test = data_handler.get_splits()

    if subset == 'valid':
        X, y = data_valid
    elif subset == 'test':
        X, y = data_test
    
    if estimator_name.startswith('truncated'):
        X = drop_shifted_columns(X)

    estimator = pipeline.estimator[-1] \
        if isinstance(pipeline, CalibratedClassifierCV) else pipeline[-1]
    
    if not expects_groups(estimator) and not data_handler.aggregate_history:
       X = X.drop(columns=data_handler.GROUP)

    metrics = [
        (metric, {}) if isinstance(metric, str) else tuple(*metric.items())
        for metric in metrics
    ]

    columns = ['estimator_name', 'subset']
    data = [estimator_name, subset]
    labels = data_handler.get_labels()

    if switches_only:
        assert not isinstance(pipeline, CalibratedClassifierCV)
        preprocessor = pipeline.named_steps['preprocessor']
        Xt_s, y_s = _separate_switches(preprocessor, data_handler.TREATMENT, X, y)
        data[-1] += '_s'
        scores = collect_scores(pipeline[-1], Xt_s, y_s, metrics, columns, data, labels)
    else:
        scores = collect_scores(pipeline, X, y, metrics, columns, data, labels)

    scores_file = join(config['results']['path'], 'scores.csv')
    try:
        df = pd.read_csv(scores_file)
        df = pd.concat([df, scores.to_frame().T], ignore_index=True)
        df.to_csv(scores_file, index=False)
    except FileNotFoundError:
        df = pd.DataFrame(scores)
        df.T.to_csv(scores_file, index=False)
