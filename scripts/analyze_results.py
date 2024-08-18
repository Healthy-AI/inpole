import os
from os.path import join
import argparse

import pandas as pd
import numpy as np
from amhelpers.config_parsing import load_config

from inpole.data import get_data_handler_from_config
from inpole.pipeline import load_best_pipeline


all_estimators = ['riskslim', 'lr', 'dt', 'pronet', 'mlp', 'rdt', 'prosenet', 'rnn']


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
    #stds = rates.groupby(id).std()

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
    else:
        raise ValueError(f"Unknown experiment '{args.experiment}'.")
    
    out = []
    
    for state, experiment_path in all_paths.items():
        trial_dirs = os.listdir(join(experiment_path, 'sweep'))
        for trial in range(1, len(trial_dirs) + 1):
            for estimator in all_estimators:
                try:
                    pipeline, results_path = load_best_pipeline(
                        experiment_path, trial, estimator, return_results_path=True)
                except FileNotFoundError:
                    continue
                
                config_path = join(results_path, 'config.yaml')
                config = load_config(config_path)
                data_handler = get_data_handler_from_config(config)
                X, y = data_handler.get_splits()[-1]  # Test data
            
                Xg = X.groupby(data_handler.GROUP)
                group_indices = [
                    np.concatenate([Xg.indices.get(id) for id in ids if id in Xg.groups])
                    for ids in patient_groups
                ]

                for group, indices in enumerate(group_indices, start=1):
                    print(f"State: {state} | Estimator: {estimator} | Trial: {trial} | Group: {group}")
                    score = pipeline.score(X.iloc[indices], y[indices], metric='auc')
                    out += [(group, state, estimator, score)]
            
    out = pd.DataFrame(out, columns=['Group', 'State', 'Estimator', 'Score'])
    out.to_csv(join(args.out_path, 'stratified_scores.csv'), index=False)
