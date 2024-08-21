import os
from os.path import join
import argparse
import collections
import pickle

import pandas as pd
import numpy as np
from amhelpers.config_parsing import load_config

from inpole.data import get_data_handler_from_config
from inpole.pipeline import load_best_pipeline
from inpole.utils import _print_log
from inpole.data import RAData, SepsisData
from inpole import ESTIMATORS
from inpole.data.data import discretize_doses


metrics = ['accuracy', 'auc']


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
    r'$X_t$':                    '/mimer/NOBACKUP/groups/inpole/results/ra/20240820_1335_sweep',
    r'$A_{t-1}$':                '/mimer/NOBACKUP/groups/inpole/results/ra/20240820_1336_sweep',
    r'$H_{(t-0):t}$':            '/mimer/NOBACKUP/groups/inpole/results/ra/20240820_1337_sweep',
    r'$\bar{H}_t$':              '/mimer/NOBACKUP/groups/inpole/results/ra/20240820_1338_sweep',  # max
    r'$H_{(t-0):t}, \bar{H}_t$': '/mimer/NOBACKUP/groups/inpole/results/ra/20240820_1339_sweep',  # max
    r'$H_{(t-1):t}, \bar{H}_t$': '/mimer/NOBACKUP/groups/inpole/results/ra/20240820_1340_sweep',  # max
    r'$H_{(t-2):t}, \bar{H}_t$': '/mimer/NOBACKUP/groups/inpole/results/ra/20240820_1341_sweep',  # max
    #r'$\bar{H}_t$':              '/mimer/NOBACKUP/groups/inpole/results/ra/20240820_1342_sweep',  # sum
    #r'$H_{(t-0):t}, \bar{H}_t$': '/mimer/NOBACKUP/groups/inpole/results/ra/20240820_1343_sweep',  # sum
    #r'$H_{(t-1):t}, \bar{H}_t$': '/mimer/NOBACKUP/groups/inpole/results/ra/20240820_1344_sweep',  # sum
    #r'$H_{(t-2):t}, \bar{H}_t$': '/mimer/NOBACKUP/groups/inpole/results/ra/20240820_1345_sweep',  # sum
    #r'$\bar{H}_t$':              '/mimer/NOBACKUP/groups/inpole/results/ra/20240820_1346_sweep',  # mean
    #r'$H_{(t-0):t}, \bar{H}_t$': '/mimer/NOBACKUP/groups/inpole/results/ra/20240820_1347_sweep',  # mean
    #r'$H_{(t-1):t}, \bar{H}_t$': '/mimer/NOBACKUP/groups/inpole/results/ra/20240820_1348_sweep',  # mean
    #r'$H_{(t-2):t}, \bar{H}_t$': '/mimer/NOBACKUP/groups/inpole/results/ra/20240820_1349_sweep',  # mean
    r'$H_t$':                    '/mimer/NOBACKUP/groups/inpole/results/ra/20240820_1350_sweep',
}

ra_bins = [-9.0, -2.8, -1.0, 0.0, 2.6]


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

    _print_log(f"Experiment: {args.experiment}")

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
        config = load_config(join(sepsis_paths['$A_{t-1}$'], 'default_config.yaml'))
        Y_prev, y, _groups = SepsisData(**config['data']).load()  # S_t = A_{t-1}
        Y_prev_discrete = Y_prev.apply(discretize_doses, raw=True, num_levels=5)
        _, y_prev = np.unique(Y_prev_discrete, axis=0, return_inverse=True)
        switch = (y != y_prev)
    elif args.experiment == 'ra':
        all_paths = ra_paths
        data = data[data.stage.ge(1)]
        patient_groups = get_patient_groups(data, 'id', 'cdai', ra_bins)
        config = load_config(join(ra_paths['$A_{t-1}$'], 'default_config.yaml'))
        X, y, _groups = RAData(**config['data']).load()
        y_prev = X.prev_therapy
        switch = (y != y_prev)
    else:
        raise ValueError(f"Unknown experiment '{args.experiment}'.")
    
    out = collections.defaultdict(list)
    out['probas'] = [('State', 'Trial', 'Estimator', 'Probas')]
    out['groups'] += [('State', 'Trial', 'Estimator', 'Group', 'Metric', 'Score')]
    out['stage'] += [('State', 'Trial', 'Estimator', 'Stage', 'Metric', 'Score')]
    out['switch'] += [('State', 'Trial', 'Estimator', 'Switch', 'Metric', 'Score')]
    out['switch_stage'] += [('State', 'Trial', 'Estimator', 'Switch', 'Stage', 'Metric', 'Score')]
    out['rho'] += [('State', 'Trial', 'Estimator', 'Stage', 'Rho')]
    
    for state, experiment_path in all_paths.items():
        trial_dirs = os.listdir(join(experiment_path, 'sweep'))
        for trial in range(1, len(trial_dirs) + 1):
            for estimator_name in ESTIMATORS:
                try:
                    pipeline, results_path = load_best_pipeline(
                        experiment_path, trial, estimator_name, return_results_path=True)
                except FileNotFoundError:
                    continue

                _print_log(f"State: {state} | Estimator: {estimator_name} | Trial: {trial}")
                
                config_path = join(results_path, 'config.yaml')
                config = load_config(config_path)
                data_handler = get_data_handler_from_config(config)
                X, y = data_handler.get_splits()[-1]  # Test data
                Xg = X.groupby(data_handler.GROUP)

                # Preprocess the input only once to save time.
                #
                # Note that we cannot reuse the preprocessed input for different 
                # estimators because the preprocessor depends on the estimator.
                preprocessor, estimator = pipeline.named_steps.values()
                Xt = preprocessor.transform(X)

                # Collect probabilities.
                probas = estimator.predict_proba(Xt)
                out['probas'] += [(state, trial, estimator_name, probas)]

                # Performance w.r.t. patient groups.
                group_indices = [
                    np.concatenate([Xg.indices.get(id) for id in ids if id in Xg.groups])
                    for ids in patient_groups
                ]
                for group, indices in enumerate(group_indices, start=1):
                    for metric in metrics:
                        score = estimator.score(Xt[indices], y[indices], metric=metric)
                        out['groups'] += [(state, trial, estimator_name, group, metric, score)]
                
                # Performance w.r.t. time.
                stages = Xg.cumcount() + 1
                for t in range(1, stages.max() + 1):
                    for metric in metrics:
                        score = estimator.score(Xt[stages==t], y[stages==t], metric=metric)
                        out['stage'] += [(state, trial, estimator_name, t, metric, score)]

                # Performance w.r.t. switches.
                s = switch[X.index]
                for metric in metrics:
                    score1 = estimator.score(Xt[s], y[s], metric=metric)
                    out['switch'] += [(state, trial, estimator_name, 'yes', metric, score1)]
                    score2 = estimator.score(Xt[~s], y[~s], metric=metric)
                    out['switch'] += [(state, trial, estimator_name, 'no', metric, score2)]
                
                # Performance w.r.t. time and switches.
                for t in range(1, stages.max() + 1):
                    s = switch[X.index] & stages.eq(t)
                    for metric in metrics:
                        score1 = estimator.score(Xt[s], y[s], metric=metric)
                        out['switch_stage'] += [(state, trial, estimator_name, 'yes', t, metric, score1)]
                        score2 = estimator.score(Xt[~s], y[~s], metric=metric)
                        out['switch_stage'] += [(state, trial, estimator_name, 'no', t, metric, score2)]
            
                # Probability products.
                probas = estimator.predict_proba(Xt)
                probas_y = probas[np.arange(len(y)), y]
                probas_y = pd.Series(probas_y, index=X.index)
                probas_y_grouped = probas_y.groupby(X[data_handler.GROUP])
                for t in range(1, stages.max() + 1):
                    probas_yt = probas_y_grouped.filter(lambda x: len(x) >= t)
                    rho = probas_yt.groupby(X[data_handler.GROUP]).prod()
                    out['rho'] += [(state, trial, estimator_name, t, rho.tolist())]
                rho = probas_y_grouped.prod()
                out['rho'] += [(state, trial, estimator_name, -1, rho.tolist())]
    
    _print_log("Saving data...")
    file_name = f'results_{args.experiment}.pickle'
    with open(file_name, 'wb') as handle:
        pickle.dump(out, handle, protocol=pickle.HIGHEST_PROTOCOL)
