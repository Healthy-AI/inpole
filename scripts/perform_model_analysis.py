import os
from os.path import join
import argparse
import collections
import joblib

import pandas as pd
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    roc_auc_score
)
from amhelpers.config_parsing import load_config

from inpole.data import get_data_handler_from_config
from inpole.pipeline import load_best_pipeline
from inpole.utils import print_log
from inpole.data import RAData, SepsisData
from inpole import ESTIMATORS
from inpole.data.data import discretize_doses


metrics = ['auc_macro', 'auc_weighted']


sepsis_paths = [
    (r'$X_t$',                '/mimer/NOBACKUP/groups/inpole/results/sepsis/20241015_1351_sweep'),
    (r'$A_{t-1}$',            '/mimer/NOBACKUP/groups/inpole/results/sepsis/20241015_1352_sweep'),
    (r'$H_{(0)}$',            '/mimer/NOBACKUP/groups/inpole/results/sepsis/20241015_1353_sweep'),
    (r'$\bar{H}_t$',          '/mimer/NOBACKUP/groups/inpole/results/sepsis/20241015_1354_sweep'),  # max
    (r'$H_{(0)}, \bar{H}_t$', '/mimer/NOBACKUP/groups/inpole/results/sepsis/20241015_1355_sweep'),  # max
    (r'$H_{(1)}, \bar{H}_t$', '/mimer/NOBACKUP/groups/inpole/results/sepsis/20241015_1356_sweep'),  # max
    (r'$H_{(2)}, \bar{H}_t$', '/mimer/NOBACKUP/groups/inpole/results/sepsis/20241015_1357_sweep'),  # max
    (r'$\bar{H}_t$',          '/mimer/NOBACKUP/groups/inpole/results/sepsis/20241015_1358_sweep'),  # sum
    (r'$H_{(0)}, \bar{H}_t$', '/mimer/NOBACKUP/groups/inpole/results/sepsis/20241015_1359_sweep'),  # sum
    (r'$H_{(1)}, \bar{H}_t$', '/mimer/NOBACKUP/groups/inpole/results/sepsis/20241015_1400_sweep'),  # sum
    (r'$H_{(2)}, \bar{H}_t$', '/mimer/NOBACKUP/groups/inpole/results/sepsis/20241015_1401_sweep'),  # sum
    (r'$\bar{H}_t$',          '/mimer/NOBACKUP/groups/inpole/results/sepsis/20241015_1402_sweep'),  # mean
    (r'$H_{(0)}, \bar{H}_t$', '/mimer/NOBACKUP/groups/inpole/results/sepsis/20241015_1403_sweep'),  # mean
    (r'$H_{(1)}, \bar{H}_t$', '/mimer/NOBACKUP/groups/inpole/results/sepsis/20241015_1404_sweep'),  # mean
    (r'$H_{(2)}, \bar{H}_t$', '/mimer/NOBACKUP/groups/inpole/results/sepsis/20241015_1405_sweep'),  # mean
    (r'$H_t$',                '/mimer/NOBACKUP/groups/inpole/results/sepsis/20241015_1406_sweep'),
]

sepsis_bins = [-0.4, -0.15, 0, 0.15, 0.4]  # Quantiles: [0.1, 0.3, 0.5, 0.7, 0.9]


ra_paths = [
    (r'$X_t$',                '/mimer/NOBACKUP/groups/inpole/results/ra/20241015_0912_sweep'),
    (r'$A_{t-1}$',            '/mimer/NOBACKUP/groups/inpole/results/ra/20241015_0913_sweep'),
    (r'$H_{(0)}$',            '/mimer/NOBACKUP/groups/inpole/results/ra/20241015_0914_sweep'),
    (r'$\bar{H}_t$',          '/mimer/NOBACKUP/groups/inpole/results/ra/20241015_0915_sweep'),  # max
    (r'$H_{(0)}, \bar{H}_t$', '/mimer/NOBACKUP/groups/inpole/results/ra/20241015_0916_sweep'),  # max
    (r'$H_{(1)}, \bar{H}_t$', '/mimer/NOBACKUP/groups/inpole/results/ra/20241015_0917_sweep'),  # max
    (r'$H_{(2)}, \bar{H}_t$', '/mimer/NOBACKUP/groups/inpole/results/ra/20241015_0918_sweep'),  # max
    (r'$\bar{H}_t$',          '/mimer/NOBACKUP/groups/inpole/results/ra/20241015_0919_sweep'),  # sum
    (r'$H_{(0)}, \bar{H}_t$', '/mimer/NOBACKUP/groups/inpole/results/ra/20241015_0920_sweep'),  # sum
    (r'$H_{(1)}, \bar{H}_t$', '/mimer/NOBACKUP/groups/inpole/results/ra/20241015_0921_sweep'),  # sum
    (r'$H_{(2)}, \bar{H}_t$', '/mimer/NOBACKUP/groups/inpole/results/ra/20241015_0922_sweep'),  # sum
    (r'$\bar{H}_t$',          '/mimer/NOBACKUP/groups/inpole/results/ra/20241015_0923_sweep'),  # mean
    (r'$H_{(0)}, \bar{H}_t$', '/mimer/NOBACKUP/groups/inpole/results/ra/20241015_0924_sweep'),  # mean
    (r'$H_{(1)}, \bar{H}_t$', '/mimer/NOBACKUP/groups/inpole/results/ra/20241015_0925_sweep'),  # mean
    (r'$H_{(2)}, \bar{H}_t$', '/mimer/NOBACKUP/groups/inpole/results/ra/20241015_0926_sweep'),  # mean
    (r'$H_t$',                '/mimer/NOBACKUP/groups/inpole/results/ra/20241015_0927_sweep'),
]

ra_bins = [-9.0, -2.8, -1.0, 0.0, 2.6]  # Quantiles: [0.1, 0.3, 0.5, 0.7, 0.9]


def get_patient_groups(data, group, variable, bin_values):
    def segment_data(series, bin_values):
        segments = []
        for bin_value in bin_values:
            segment = series[series <= bin_value]
            segments.append(segment)
            series = series.drop(segment.index)
        segments.append(series)
        return segments
    
    rates = data.set_index(group).groupby(group)[variable].diff()
    means = rates.groupby(group).mean()

    segments = segment_data(means, bin_values)
    
    return [s.index for s in segments]


def get_score(y, y_proba, metric):
    if metric == 'accuracy':
        y_pred = y_proba.argmax(axis=1)
        return accuracy_score(y, y_pred)
    elif metric == 'balanced_accuracy':
        y_pred = y_proba.argmax(axis=1)
        return balanced_accuracy_score(y, y_pred)
    elif metric == 'auc_macro' or 'auc_weighted':
        average = metric.split('_')[-1]
        kwargs = {'average': average, 'multi_class': 'ovr'}
        try:
            return roc_auc_score(y, y_proba, **kwargs)
        except ValueError:
            return float('nan')
    else:
        raise ValueError(f"Unknown metric '{metric}'.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment', type=str, required=True)
    parser.add_argument('--data_path', type=str, required=True)
    parser.add_argument('--out_path', type=str, required=True)
    args = parser.parse_args()

    print_log(f"Experiment: {args.experiment}")

    if args.data_path.endswith('.csv'):
        data = pd.read_csv(args.data_path)
    elif args.data_path.endswith('.pkl'):
        data = pd.read_pickle(args.data_path)
    else:
        file_format = args.data_path.split('.')[-1]
        raise ValueError(f"Unknown file format: {file_format}.")

    if args.experiment == 'sepsis':
        all_paths = sepsis_paths
        patient_groups = get_patient_groups(data, 'icustayid', 'NEWS2', sepsis_bins)
        config_path = join(dict(sepsis_paths)['$A_{t-1}$'], 'default_config.yaml')
        config = load_config(config_path)
        Y_prev, y, _groups = SepsisData(**config['data']).load()  # S_t = A_{t-1} => X = Y_prev
        Y_prev_discrete = Y_prev.apply(discretize_doses, raw=True, num_levels=5)
        _, y_prev = np.unique(Y_prev_discrete, axis=0, return_inverse=True)
        switch = (y != y_prev)
    elif args.experiment == 'ra':
        all_paths = ra_paths
        data = data[data.stage.ge(1)]
        patient_groups = get_patient_groups(data, 'id', 'cdai', ra_bins)
        config_path = join(dict(ra_paths)['$A_{t-1}$'], 'default_config.yaml')
        config = load_config(config_path)
        X, y, _groups = RAData(**config['data']).load()
        y_prev = X.prev_therapy
        switch = (y != y_prev)
    else:
        raise ValueError(f"Unknown experiment '{args.experiment}'.")
    
    out = collections.defaultdict(list)
    out['probas'] = [('State', 'Reduction', 'Trial', 'Estimator', 'Probas')]
    out['groups'] += [('State', 'Reduction', 'Trial', 'Estimator', 'Group', 'Metric', 'Score')]
    out['stage'] += [('State', 'Reduction', 'Trial', 'Estimator', 'Stage', 'Metric', 'Score')]
    out['switch'] += [('State', 'Reduction', 'Trial', 'Estimator', 'Switch', 'Metric', 'Score')]
    out['switch_stage'] += [('State', 'Reduction', 'Trial', 'Estimator', 'Switch', 'Stage', 'Metric', 'Score')]
    out['rho'] += [('State', 'Reduction', 'Trial', 'Estimator', 'Stage', 'Rho')]
    
    for state, experiment_path in all_paths:
        trial_dirs = os.listdir(join(experiment_path, 'sweep'))
        for trial in range(1, len(trial_dirs) + 1):
            for estimator_name in ESTIMATORS:
                try:
                    pipeline, results_path = load_best_pipeline(
                        experiment_path, trial, estimator_name, return_results_path=True
                    )
                except FileNotFoundError:
                    continue
                
                config_path = join(results_path, 'config.yaml')
                config = load_config(config_path)
                data_handler = get_data_handler_from_config(config)
                X, y = data_handler.get_splits()[-1]  # Test data
                Xg = X.groupby(data_handler.GROUP)
    
                reduction = data_handler.reduction if r'\bar{H}_t' in state else 'none'
    
                print_log(f"State: {state} | Reduction: {reduction} | Trial: {trial} | Estimator: {estimator_name}")
    
                # Collect probabilities.
                probas = pipeline.predict_proba(X)
                out['probas'] += [(state, reduction, trial, estimator_name, probas)]
    
                # Measure performance w.r.t. patient groups.
                group_indices = [
                    np.concatenate([Xg.indices.get(id) for id in ids if id in Xg.groups])
                    for ids in patient_groups
                ]
                for group, indices in enumerate(group_indices, start=1):
                    for metric in metrics:
                        score = get_score(y[indices], probas[indices], metric=metric)
                        out['groups'] += [(state, reduction, trial, estimator_name, group, metric, score)]
                
                # Measure performance w.r.t. stage.
                stages = Xg.cumcount() + 1
                for t in range(1, stages.max() + 1):
                    for metric in metrics:
                        score = get_score(y[stages==t], probas[stages==t], metric=metric)
                        out['stage'] += [(state, reduction, trial, estimator_name, t, metric, score)]
    
                # Measure performance w.r.t. treatment switching.
                s = switch[X.index]
                for metric in metrics:
                    score1 = get_score(y[s], probas[s], metric=metric)
                    out['switch'] += [(state, reduction, trial, estimator_name, 'yes', metric, score1)]
                    score2 = get_score(y[~s], probas[~s], metric=metric)
                    out['switch'] += [(state, reduction, trial, estimator_name, 'no', metric, score2)]
                
                # Measure performance w.r.t. stage and treatment switching.
                for t in range(1, stages.max() + 1):
                    s = switch[X.index] & stages.eq(t)
                    if s.any():
                        for metric in metrics:
                            score1 = get_score(y[s], probas[s], metric=metric)
                            out['switch_stage'] += [(state, reduction, trial, estimator_name, 'yes', t, metric, score1)]
                            score2 = get_score(y[~s], probas[~s], metric=metric)
                            out['switch_stage'] += [(state, reduction, trial, estimator_name, 'no', t, metric, score2)]
            
                # Compute probability products.
                probas_y = probas[np.arange(len(y)), y]
                probas_y = pd.Series(probas_y, index=X.index)
                probas_y_grouped = probas_y.groupby(X[data_handler.GROUP])
                for t in range(1, stages.max() + 1):
                    probas_yt = probas_y_grouped.filter(lambda x: len(x) >= t)
                    probas_yt = probas_yt.groupby(X[data_handler.GROUP]).head(t)
                    rho = probas_yt.groupby(X[data_handler.GROUP]).prod()
                    out['rho'] += [(state, reduction, trial, estimator_name, t, rho.tolist())]
                rho = probas_y_grouped.prod()
                out['rho'] += [(state, reduction, trial, estimator_name, -1, rho.tolist())]

    print_log("Saving final data...")
    out_file_name = f'results_{args.experiment}.pickle'
    joblib.dump(out, join(args.out_path, out_file_name))
