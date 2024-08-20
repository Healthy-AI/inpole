import os
from os.path import join
import argparse
import collections

import pandas as pd
import numpy as np
from amhelpers.config_parsing import load_config

from inpole.data import get_data_handler_from_config
from inpole.pipeline import load_best_pipeline
from inpole.utils import _print_log
from inpole.data import RAData


all_estimators = ['riskslim', 'lr', 'dt', 'rulefit', 'pronet', 'mlp', 'rdt', 'prosenet', 'rnn']


sepsis_paths = {
    r'$X_t$': '/mimer/NOBACKUP/groups/oovgen/inpole/results/sepsis/20240304_1126_sweep',
    r'$A_{t-1}$': '/mimer/NOBACKUP/groups/oovgen/inpole/results/sepsis/20240402_1810_sweep',
    r'$H_{(t-0):t}$': '/mimer/NOBACKUP/groups/oovgen/inpole/results/sepsis/20240304_1127_sweep',
    r'$\bar{H}_t$': '/mimer/NOBACKUP/groups/oovgen/inpole/results/sepsis/20240304_2152_sweep',
    r'$H_{(t-0):t}, \bar{H}_t$': '/mimer/NOBACKUP/groups/oovgen/inpole/results/sepsis/20240304_2153_sweep',
    r'$H_{(t-1):t}$': '/mimer/NOBACKUP/groups/oovgen/inpole/results/sepsis/20240304_1131_sweep',
    r'$H_{(t-1):t}, \bar{H}_t$': '/mimer/NOBACKUP/groups/oovgen/inpole/results/sepsis/20240320_2051_sweep',
    r'$H_{(t-2):t}$': '/mimer/NOBACKUP/groups/oovgen/inpole/results/sepsis/20240304_1132_sweep',
    r'$H_{(t-2):t}, \bar{H}_t$': '/mimer/NOBACKUP/groups/oovgen/inpole/results/sepsis/20240402_1811_sweep',
    r'$H_t$': '/mimer/NOBACKUP/groups/oovgen/inpole/results/sepsis/20240324_1040_sweep',
}

sepsis_bins = [-0.4, -0.15, 0, 0.15, 0.4]


ra_paths = {
    r'$X_t$':                    '/mimer/NOBACKUP/groups/inpole/results/ra/20240819_2221_sweep',
    r'$A_{t-1}$':                '/mimer/NOBACKUP/groups/inpole/results/ra/20240819_2222_sweep',
    r'$H_{(t-0):t}$':            '/mimer/NOBACKUP/groups/inpole/results/ra/20240819_2223_sweep',
    r'$\bar{H}_t$':              '/mimer/NOBACKUP/groups/inpole/results/ra/20240819_2224_sweep',  # max
    r'$H_{(t-0):t}, \bar{H}_t$': '/mimer/NOBACKUP/groups/inpole/results/ra/20240819_2225_sweep',  # max
    r'$H_{(t-1):t}, \bar{H}_t$': '/mimer/NOBACKUP/groups/inpole/results/ra/20240819_2226_sweep',  # max
    r'$H_{(t-2):t}, \bar{H}_t$': '/mimer/NOBACKUP/groups/inpole/results/ra/20240819_2227_sweep',  # max
    #r'$\bar{H}_t$':              '/mimer/NOBACKUP/groups/inpole/results/ra/20240819_2228_sweep',  # sum
    #r'$H_{(t-0):t}, \bar{H}_t$': '/mimer/NOBACKUP/groups/inpole/results/ra/20240819_2229_sweep',  # sum
    #r'$H_{(t-1):t}, \bar{H}_t$': '/mimer/NOBACKUP/groups/inpole/results/ra/20240819_2230_sweep',  # sum
    #r'$H_{(t-2):t}, \bar{H}_t$': '/mimer/NOBACKUP/groups/inpole/results/ra/20240819_2231_sweep',  # sum
    #r'$\bar{H}_t$':              '/mimer/NOBACKUP/groups/inpole/results/ra/20240819_2232_sweep',  # mean
    #r'$H_{(t-0):t}, \bar{H}_t$': '/mimer/NOBACKUP/groups/inpole/results/ra/20240819_2233_sweep',  # mean
    #r'$H_{(t-1):t}, \bar{H}_t$': '/mimer/NOBACKUP/groups/inpole/results/ra/20240819_2234_sweep',  # mean
    #r'$H_{(t-2):t}, \bar{H}_t$': '/mimer/NOBACKUP/groups/inpole/results/ra/20240819_2235_sweep',  # mean
    r'$H_t$':                    '/mimer/NOBACKUP/groups/inpole/results/ra/20240819_2236_sweep',
}

ra_bins = [-8.0, -2.4, -0.6, 0.4, 4.4]


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


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment', type=str, required=True)
    parser.add_argument('--data_path', type=str, required=True)
    parser.add_argument('--out_path', type=str, required=True)
    args = parser.parse_args()

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
    elif args.experiment == 'ra':
        all_paths = ra_paths
        data = data[data.stage.ge(1)]
        patient_groups = get_patient_groups(data, 'id', 'cdai', ra_bins)
        config = load_config(join(ra_paths['$A_{t-1}$'], 'default_config.yaml'))
        X, y, _groups = RAData(**config['data']).load()
        switch = (y != X.prev_therapy)
    else:
        raise ValueError(f"Unknown experiment '{args.experiment}'.")
    
    scores = collections.defaultdict(list)
    
    for state, experiment_path in all_paths.items():
        trial_dirs = os.listdir(join(experiment_path, 'sweep'))
        for trial in range(1, len(trial_dirs) + 1):
            for estimator in all_estimators:
                try:
                    pipeline, results_path = load_best_pipeline(
                        experiment_path, trial, estimator, return_results_path=True)
                except FileNotFoundError:
                    continue

                _print_log(f"State: {state} | Estimator: {estimator} | Trial: {trial}")
                
                config_path = join(results_path, 'config.yaml')
                config = load_config(config_path)
                data_handler = get_data_handler_from_config(config)
                X, y = data_handler.get_splits()[-1]  # Test data
                
                # Stratification w.r.t. time.
                stages = X.groupby(data_handler.GROUP).cumcount()
                for t in range(stages.max()):
                    score = pipeline.score(X[stages==t], y[stages==t], metric='auc')
                    scores['time'] += [(t + 1, state, estimator, score)]

                # Stratification w.r.t. patient groups.
                Xg = X.groupby(data_handler.GROUP)
                group_indices = [
                    np.concatenate([Xg.indices.get(id) for id in ids if id in Xg.groups])
                    for ids in patient_groups
                ]
                for group, indices in enumerate(group_indices, start=1):
                    score = pipeline.score(X.iloc[indices], y[indices], metric='auc')
                    scores['groups'] += [(group, state, estimator, score)]

                if args.experiment == 'ra':
                    # Evalute model on switches only.
                    s = switch[X.index]
                    score = pipeline.score(X[s], y[s], metric='auc')
                    scores['switches'] += [(state, estimator, score)]

    df = pd.DataFrame(scores['time'], columns=['Stage', 'State', 'Estimator', 'Score'])
    file_name = f'scores_time_{args.experiment}.csv'
    df.to_csv(join(args.out_path, file_name), index=False)

    df = pd.DataFrame(scores['groups'], columns=['Group', 'State', 'Estimator', 'Score'])
    file_name = f'scores_groups_{args.experiment}.csv'
    df.to_csv(join(args.out_path, file_name), index=False)

    if args.experiment == 'ra':
        df = pd.DataFrame(scores['switches'], columns=['State', 'Estimator', 'Score'])
        df.to_csv(join(args.out_path, 'scores_switches_ra.csv'), index=False)
