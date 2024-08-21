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
from inpole.data import RAData, SepsisData
from inpole import ESTIMATORS
from inpole.data.data import discretize_doses


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
    
    scores = collections.defaultdict(list)
    
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

                preprocessor, estimator = pipeline.named_steps.values()
                Xt = preprocessor.transform(X)
                
                # Stratification w.r.t. time.
                stages = X.groupby(data_handler.GROUP).cumcount()
                for t in range(stages.max()):
                    #score = pipeline.score(X[stages==t], y[stages==t], metric='auc')
                    score = estimator.score(Xt[stages==t], y[stages==t], metric='auc')
                    scores['time'] += [(t + 1, state, estimator_name, score)]

                # Stratification w.r.t. patient groups.
                Xg = X.groupby(data_handler.GROUP)
                group_indices = [
                    np.concatenate([Xg.indices.get(id) for id in ids if id in Xg.groups])
                    for ids in patient_groups
                ]
                for group, indices in enumerate(group_indices, start=1):
                    #score = pipeline.score(X.iloc[indices], y[indices], metric='auc')
                    score = estimator.score(Xt[indices], y[indices], metric='auc')
                    scores['groups'] += [(group, state, estimator_name, score)]

                # Misclassification.
                #yp = pipeline.predict(X)
                yp = estimator.predict(Xt)
                stage = X.groupby(data_handler.GROUP).cumcount() + 1
                incorrect = stage[y != yp].value_counts().sort_index()
                frequency = incorrect / stage.value_counts().sort_index()
                scores['frequency'] += [(state, estimator_name, frequency.tolist())]
                
                # Compute probability products.
                #probas = pipeline.predict_proba(X)
                probas = estimator.predict_proba(Xt)
                probas_y = probas[np.arange(len(y)), y]
                probas_y = pd.Series(probas_y, index=X.index)
                probas_y_grouped = probas_y.groupby(X[data_handler.GROUP])
                for t in range(1, stages.max() + 1):
                    probas_yt = probas_y_grouped.filter(lambda x: len(x) >= t)
                    rho = probas_yt.groupby(X[data_handler.GROUP]).prod()
                    scores['rho'] += [(t, state, estimator_name, rho.tolist())]
                rho = probas_y_grouped.prod()
                scores['rho'] += [(-1, state, estimator_name, rho.tolist())]

                # Evaluate model on treatment switches and non_treatment switches.
                s = switch[X.index]
                #score1 = pipeline.score(X[s], y[s], metric='auc')
                score1 = estimator.score(Xt[s], y[s], metric='auc')
                scores['switch'] += [('yes', state, estimator_name, score1)]
                #score2 = pipeline.score(X[~s], y[~s], metric='auc')
                score2 = estimator.score(Xt[~s], y[~s], metric='auc')
                scores['switch'] += [('no', state, estimator_name, score2)]

    df = pd.DataFrame(scores['time'], columns=['Stage', 'State', 'Estimator', 'Score'])
    file_name = f'scores_time_{args.experiment}.csv'
    df.to_csv(join(args.out_path, file_name), index=False)

    df = pd.DataFrame(scores['groups'], columns=['Group', 'State', 'Estimator', 'Score'])
    file_name = f'scores_groups_{args.experiment}.csv'
    df.to_csv(join(args.out_path, file_name), index=False)

    df = pd.DataFrame(scores['frequency'], columns=['State', 'Estimator', 'Frequency'])
    file_name = f'scores_frequency_{args.experiment}.csv'
    df.to_csv(join(args.out_path, file_name), index=False)

    df = pd.DataFrame(scores['rho'], columns=['Stage', 'State', 'Estimator', 'Score'])
    file_name = f'scores_rho_{args.experiment}.csv'
    df.to_csv(join(args.out_path, file_name), index=False)

    df = pd.DataFrame(scores['switch'], columns=['Switch', 'State', 'Estimator', 'Score'])
    file_name = f'scores_switch_{args.experiment}.csv'
    df.to_csv(join(args.out_path, file_name), index=False)
