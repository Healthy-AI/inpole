import os
import warnings
from pathlib import Path
from datetime import timedelta
from os.path import join

import joblib
import pandas as pd
import numpy as np
import colorcet as cc
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib import gridspec
from seaborn._statistics import EstimateAggregator
from sklearn.decomposition import PCA
from amhelpers.config_parsing import load_config

from inpole.pipeline import _get_estimator_params
from inpole.utils import merge_dicts
from inpole.models.models import get_model_complexity


def visualize_encodings(encodings, prototype_indices, frac=0.1, 
                        annotations=None, figsize=(6,4)):
    
    pca = PCA(n_components=2).fit(encodings)
    
    # Transform the encodings.
    encodings_pca = pca.transform(encodings)
    _encodings = {
        'PC 1': encodings_pca[:, 0],
        'PC 2': encodings_pca[:, 1],
        'Prototype': 'No'
    }
    _encodings = pd.DataFrame(_encodings)
    
    # Sample a fraction of the encodings.
    _encodings = _encodings.sample(frac=frac, axis='index')
    
    # Transform the prototypes.
    prototypes_pca = encodings_pca[prototype_indices]
    _prototypes = {
        'PC 1': prototypes_pca[:, 0], 
        'PC 2': prototypes_pca[:, 1],
        'Prototype': 'Yes',
    }
    _prototypes = pd.DataFrame(_prototypes)
    
    fig, ax = plt.subplots(figsize=figsize)
    
    common_kwargs = {'x': 'PC 1', 'y': 'PC 2', 'ax': ax}
    
    sns.scatterplot(data=_encodings, alpha=0.7, size='Prototype', 
                    sizes=(20, 100), size_order=['Yes', 'No'],
                    **common_kwargs)
    
    n_prototypes = prototypes_pca.shape[0]
    sns.scatterplot(data=_prototypes, alpha=1, s=n_prototypes*[100],
                    legend=False, **common_kwargs)
    
    # Annotate the prototypes.
    for i, a in enumerate(prototypes_pca, start=1):
        try:
            xytext = (a[0]+annotations[i][0], a[1]+annotations[i][1])
            ax.annotate(i, xy=a, xytext=xytext, 
                        arrowprops={'arrowstyle': '-'})
        except TypeError:
            ax.annotate(i, xy=(a[0]+0.1, a[1]))
    
    ax.legend(loc="upper left", bbox_to_anchor=(1, 1), title='Prototype')

    return fig, ax


def visualize_prototype_ra(X, prototype_index, color_mapper=None):
    if color_mapper is None:
        labels = X.therapy.cat.categories.tolist()
        colors = sns.color_palette(cc.glasbey, n_colors=len(labels))
        color_mapper = {t: c for t, c in zip(labels, colors)}

    pid = X.id.iat[prototype_index]
    x = X[X.id == pid]
    
    d = (x.date.max() - x.date.min()).days / 10
    if d == 0: d = 10
    
    fig = plt.figure(figsize=(6, 4))
    
    gs = gridspec.GridSpec(2, 1, height_ratios=[4, 1])
    
    # Plot CDAI.
    ax1 = plt.subplot(gs[0])
    ax1.set_title('CDAI')
    
    i = X.index[prototype_index]
    t = list(x.index).index(i)
    plot_kwargs = {'c': 'k', 'marker': 'o', 'ms': 5}
    ax1.plot(x.date.iloc[:t+1], x.cdai.iloc[:t+1], **plot_kwargs)
    ax1.plot(x.date.iloc[t:], x.cdai.iloc[t:], ls='dashed', **plot_kwargs)
    ax1.scatter(x.date.iloc[t], x.cdai.iloc[t], c='k', marker='o', s=50)
    
    dates = x.date.tolist()
    dates += [dates[-1] + timedelta(days=d)]
    ones = np.ones(len(dates))
    ax1.plot(dates, ones, alpha=0)
    
    ax1.set_xticks(x.date)
    ax1.set_xticklabels([])
    ax1.set_ylim(0, 76)
    
    kwargs = {'alpha': 0.15, 'zorder': 0}
    ax1.axhspan(0, 2.8, facecolor='g', **kwargs)
    ax1.axhspan(2.8, 10, facecolor='y', **kwargs)
    ax1.axhspan(10, 22, facecolor='orange', **kwargs)
    ax1.axhspan(22, 76, facecolor='r', **kwargs)
    
    # Plot therapy.
    ax2 = plt.subplot(gs[1])
    ax2.set_title('Therapy')
    
    for i in range(len(x)):
        xmin = x.date.iat[i]
        xmax = x.date.iat[i+1] if i+1 < len(x) else xmin + timedelta(days=d)
        color = color_mapper[x.therapy.iat[i]]
        ax2.axvspan(xmin, xmax, facecolor=color)
    
    ax2.set_xticks(x.date)
    ax2.set_xticklabels(x.date.dt.year, rotation=90)
    ax2.set_yticks([])
    
    unique_therapies = x.therapy.unique()
    artists = [Patch(color=color_mapper[t]) for t in unique_therapies]
    ax2.legend(artists, unique_therapies, bbox_to_anchor=(1, 0),
               loc='lower left', title='Therapy')
    
    return fig


def get_score_table(scores, metric, subset='test', ci=95, n_boot=1000, seed=0,
                    factor=1, precision=1):
    estimator = next(c for c in scores.columns if c.startswith('estimator'))
    g = scores[scores.subset==subset].groupby(estimator)
    agg = EstimateAggregator('mean', ('ci', ci), n_boot=n_boot, seed=seed)
    table = g.apply(agg, var=metric)
    table = table * factor
    return table.style.format(precision=precision)


def _get_hparam_names(params):
    hparams = merge_dicts(params)
    hparams = {k: v for k, v in hparams.items() if len(set(v)) > 1}
    hparams.pop('results_path', None)
    hparams.pop('seed', None)
    return list(hparams)


def get_params_and_scores(sweep_path, estimator_name, trials=None):
    params, scores = [], []

    if trials is None:
        # Get all sorted trial directories.
        trial_dirs = sorted(os.listdir(sweep_path))
    else:
        trial_dirs = [join(sweep_path, f'trial_{trial:02d}') for trial in trials]
    
    for trial_dir in trial_dirs:
        trial_path = join(sweep_path, trial_dir)
        
        for d in sorted(Path(trial_path).iterdir()):
            exp = str(d).split('/')[-1]  # `exp` is on the form estimator_XX
            name = '_'.join(exp.split('_')[:-1])  # Get the estimator name
            
            if name == estimator_name:
                try:
                    scores_path = join(d, 'scores.csv')
                    _scores = pd.read_csv(scores_path)
                    scores.append(_scores)
                except FileNotFoundError:
                    continue

                config_path = join(d, 'config.yaml')
                _config = load_config(config_path)
                _params = _get_estimator_params(_config, estimator_name)
                _params.pop('random_state', None)
                params.append(_params)
    
    return params, scores


def inspect_hyperparameters(sweep_path, estimator_name, metric='auc', 
                            trials=None, **plot_kwargs):
    # Get all parameters and scores.
    params, scores = get_params_and_scores(sweep_path, estimator_name, trials)

    if len(scores) == 0:
        raise ValueError(f"No scores were found for estimator {estimator_name}.")

    # Get hyperparameter names.
    hparams = _get_hparam_names(params)

    # Plot hyperparameter values against scores.
    num_subplots = len(hparams)
    _fig, axes = plt.subplots(num_subplots, 1, sharex=True, **plot_kwargs)
    axes = axes.flatten() if num_subplots > 1 else [axes]

    num_candidates = len(params)
    if num_candidates > 10:
        colors = sns.color_palette(cc.glasbey, n_colors=num_candidates)
    else:
        colors = sns.color_palette('colorblind', n_colors=num_candidates)
    
    for ax, hparam in zip(axes, hparams):
        ax.set_title(hparam)

    value_mapper = {}

    for ax, hparam in zip(axes, hparams):
        values = [p[hparam] for p in params]
        if set(values) == {0, 1}:
            continue
        if all([isinstance(v, tuple) for v in values]):
            unique_values = list(set(values))
            unique_values = sorted(unique_values, key=lambda x: (len(x), x[0]))
            value_mapper[hparam] = {k: v for v, k in enumerate(unique_values)}
            ax.set_yticks(range(len(unique_values)))
            ax.set_yticklabels(unique_values)
        try:
            log_values = np.log10(values)
            if log_values.max() - log_values.min() >= 2:
                ax.set_yscale('log')
        except:
            pass
    
    for _params, _scores, color in zip(params, scores, colors):
        score = _scores[_scores.subset=='valid'][metric].item()
        for ax, hparam in zip(axes, hparams):
            value = _params[hparam]
            try:
                if hparam in value_mapper:
                    v = value_mapper[hparam][value]
                    ax.scatter(score, v, color=color)
                else:
                    ax.scatter(score, value, color=color)
            except ValueError:
                warnings.warn(f"Failed to plot parameter value {value}.")


def get_model_complexities_and_scores(trial_path, estimator_name, metric='auc'):
    complexities, scores = [], []

    for experiment_dir in os.listdir(trial_path):
        exp = experiment_dir.split('/')[-1]  # `exp` is on the form estimator_XX
        name = '_'.join(exp.split('_')[:-1])  # Get the estimator name
        
        if name == estimator_name:
            pipeline_path = join(trial_path, experiment_dir, 'pipeline.pkl')
            if os.path.exists(pipeline_path):
                pipeline = joblib.load(pipeline_path)
                estimator = pipeline.named_steps['estimator']
                complexity = get_model_complexity(estimator)
            else:
                complexity = np.nan
            complexities.append(complexity)
            
            scores_path = join(trial_path, experiment_dir, 'scores.csv')
            if os.path.exists(scores_path):
                s = pd.read_csv(scores_path)
                mask = s.subset == 'test'
                if name in ['rdt', 'truncated_rdt']:
                    mask &= s.estimator_name == f'{name}_aligned'
                score = s[mask][metric].item()
            else:
                score = np.nan
            scores.append(score)

    return np.array(complexities), np.array(scores)
