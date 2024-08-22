import os
import re
import warnings
from pathlib import Path
from os.path import join

import joblib
import pandas as pd
import numpy as np
import colorcet as cc
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from seaborn._statistics import EstimateAggregator
from sklearn.decomposition import PCA
from amhelpers.config_parsing import load_config
from sklearn.tree._export import _MPLTreeExporter
from sklearn.tree._reingold_tilford import Tree
from pandas.api.types import is_categorical_dtype, is_bool_dtype

from inpole.pipeline import _get_estimator_params, load_best_pipeline
from inpole.utils import merge_dicts
from inpole.models.models import get_model_complexity
from inpole.data import get_data_handler_from_config


# @TODO: Rename this module to `postprocessing.py`.


def visualize_encodings(encodings, prototype_indices, frac=0.1, figsize=(6,4),
                        annotations=None, hue=None, hue_key=None):
    
    pca = PCA(n_components=2).fit(encodings)
    
    # Transform the encodings.
    encodings_pca = pca.transform(encodings)

    if hue is not None and hue_key is not None:
        hue_mapping = {0: 'No', 1: 'Yes'}
        hue = [hue_mapping.get(item, item) for item in hue]
    
    _encodings = {
        'PC 1': encodings_pca[:, 0],
        'PC 2': encodings_pca[:, 1],
        'Prototype': 'No',
        hue_key: hue
    }
    _encodings = pd.DataFrame(_encodings)
    
    # Sample a fraction of the encodings.
    _encodings = _encodings.sample(frac=frac, axis='index')
    
    # Transform the prototypes.
    prototypes_pca = encodings_pca[prototype_indices]
    prototypes_hue = [hue[i] for i in prototype_indices] if hue is not None else None    
    _prototypes = {
        'PC 1': prototypes_pca[:, 0], 
        'PC 2': prototypes_pca[:, 1],
        'Prototype': 'Yes',
        hue_key: prototypes_hue
    }
    _prototypes = pd.DataFrame(_prototypes)
    
    fig, ax = plt.subplots(figsize=figsize)

    common_kwargs = {'x': 'PC 1', 'y': 'PC 2', 'ax': ax}
    if hue is not None:
        common_kwargs['hue'] = hue_key
    
    sns.scatterplot(data=_encodings, alpha=0.7, size='Prototype', 
                    sizes=(20, 100), size_order=['Yes', 'No'], hue_order=['No','Yes'],
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

    title = 'Prototype' if hue is None else None
    ax.legend(loc="upper right", title=title)

    return fig, ax


def find_median_IQR(x):
    x_median = int(x.median())
    x_25th = int(x.quantile(0.25))
    x_75th = int(x.quantile(0.75))
    return x_median, x_25th, x_75th


def visualize_prototype_ra(X, prototype_indices, closest_sequences, color_mapper = None, 
                           comorbidities = None, targeted_adverse_events = None, infections = None):
    labels = X.therapy.cat.categories.tolist()
    colors_list = sns.color_palette("tab10", n_colors=len(prototype_indices))  
    markers_list = ['o', 's', '^', 'p', '*', 'D', 'v', '<', '>'] 

    fig, ax = plt.subplots(figsize=(6,4))
    all_max_time_steps = []
    prototypes_dfs = {}
    prototype_info = ""

    for idx, (prototype_idx, sequences) in enumerate(closest_sequences.items()):
        Xt = pd.DataFrame(columns=X.columns.tolist() + ['time_step'])
        pid = X.id.iloc[sequences].unique()
        proto_id = X.id.iloc[prototype_idx]
        
        n_switch = 0
        for p in pid:
            xs_date = X[X.id == p].sort_values(by='date')
            xs_date['time_step'] = xs_date.groupby('id').cumcount() + 1
            Xt = pd.concat([Xt, xs_date], ignore_index = False)
            unique_therapy = X[X.id == p].therapy.unique()
            
            if len(unique_therapy) == 1:
                switch = 0
            else:
                switch = len(unique_therapy)-1
            n_switch += switch
                    
        prototypes_dfs[prototype_idx] = Xt
                
        # find prototype index
        t = list(Xt.index).index(prototype_idx)

        cdai_stat = Xt.groupby('time_step')['cdai'].agg(['mean', 'std'])
        cdai_stat = cdai_stat.dropna()
        cdai_stat['std_pos'] = cdai_stat['mean'] + cdai_stat['std']
        cdai_stat['std_neg'] = cdai_stat['mean'] - cdai_stat['std']
        cdai_stat.reset_index(inplace=True)

        # visulize cdai
        color = colors_list[idx % len(colors_list)]  
        marker = markers_list[idx % len(markers_list)]

        ax.plot(cdai_stat['time_step'].iloc[:t+1], cdai_stat['mean'].iloc[:t+1], marker=marker, color=color, markersize=5)
        ax.plot(cdai_stat['time_step'].iloc[t:], cdai_stat['mean'].iloc[t:], ls='dashed', marker=marker, color=color, markersize=5)
        ax.scatter(cdai_stat['time_step'].iloc[t], cdai_stat['mean'].iloc[t], marker=marker, color=color, s=60, label=f'Prototype {idx+1}')
        ax.fill_between(cdai_stat['time_step'], cdai_stat['std_pos'], cdai_stat['std_neg'], color=color, alpha=0.2)
        all_max_time_steps.append(cdai_stat['time_step'].max()) 

        # Info box
        x = Xt[Xt.index.isin(sequences)]
        age, age_25th, age_75th = find_median_IQR(x['age'])
        duration_ra, duration_ra_25th, duration_ra_75th = find_median_IQR(x['duration_ra'])
        pain, pain_25th, pain_75th = find_median_IQR(x['pt_pain'])
        seatedbp1, seatedbp1_25th, seatedbp1_75th = find_median_IQR(x['seatedbp1'])
        n_comor = sum(x[col].eq(1).any() for col in comorbidities) 
        n_preg = x['pregnant_current'].sum()
        n_tae = sum(x[col].eq(1).any() for col in targeted_adverse_events)
        n_infection = sum(x[col].eq(1).any() for col in infections)
        
        prototype_info += f"Prototype {idx + 1}:\n Age: {age} ({age_25th}, {age_75th})\n" \
                            f"RA duration: {duration_ra} ({duration_ra_25th}, {duration_ra_75th})\n" \
                            f"Systolic BP: {seatedbp1} ({seatedbp1_25th}, {seatedbp1_75th})\n" \
                            f"Pain (1--100): {pain} ({pain_25th}, {pain_75th})\n Comorbidities: {n_comor}\n" \
                            f"Infections: {n_infection}\n Switches: {n_switch}\n\n"
        
    max_time_step = max(all_max_time_steps)
    ax.set_xticks(range(1, int(max_time_step) + 1))
    ax.set_xlabel('Stage')
    ax.set_ylabel('CDAI')
    ax.legend(loc='upper left')
    ax.set_ylim(0, 60) 

    kwargs = {'alpha': 0.15, 'zorder': 0}
    ax.axhspan(0, 2.8, facecolor='g', **kwargs)
    ax.axhspan(2.8, 10, facecolor='y', **kwargs)
    ax.axhspan(10, 22, facecolor='orange', **kwargs)
    ax.axhspan(22, 60, facecolor='r', **kwargs)
    fig.text(0.92, 0.12, prototype_info.strip(), fontsize=12, bbox=dict(facecolor='white', alpha=0.2))
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


def get_model_complexities_and_scores(trial_path, estimator_name, subset='test', metric='auc'):
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
                mask = s.subset == subset
                if name in ['rdt', 'truncated_rdt']:
                    mask &= s.estimator_name == f'{name}_aligned'
                score = s[mask][metric].item()
            else:
                score = np.nan
            scores.append(score)

    return np.array(complexities), np.array(scores)


def plot_model_complexity(estimators, trial_paths, estimator_mapper, subset='test', metric='auc'):
    for estimator, (bins, xlabel, xlog, ylim, ax) in estimators.items():
        trial_path = trial_paths[estimator]
        complexities, scores = get_model_complexities_and_scores(trial_path, estimator, subset, metric)
        
        assert len(complexities) == len(scores)
        
        if len(complexities) > 0:
            label = estimator_mapper[estimator]
            if bins is not None:
                indices = np.digitize(complexities, bins)
                x, y = [], []
                xticks, xticklabels = [], []
                for i in range(1, len(bins)):
                    le, re = bins[i-1], bins[i]
                    m = le + (re - le) / 2
                    xticks.append(m)
                    xticklabels.append(f'{le}–{re}')
                    if i in indices:
                        s = max(scores[indices==i])
                        x.append(m)
                        y.append(s)
                ax.plot(x, y, 'ko-', label=label)
                ax.set_xticks(xticks)
                ax.set_xticklabels(xticklabels, rotation=90)
            else:
                unique = np.unique(complexities)
                unique = unique[~np.isnan(unique)]
                max_scores = [max(scores[complexities==x]) for x in unique]
                ax.plot(unique, max_scores, 'ko-', label=label)
                ax.set_xticks(unique)
            
            ax.set_xlabel(xlabel)
            if xlog:
                ax.set_xscale('log')
            if ylim is not None:
                ax.set_ylim(ylim)
            ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%0.2f'))
            ax.legend()
            ax.grid('on')


class TreeExporter(_MPLTreeExporter):
    def __init__(
        self,
        max_depth=None,
        feature_names=None,
        class_names=None,
        filled=False,
        proportion=False,
        rounded=False,
        precision=3,
        fontsize=None,
        node_ids_to_include=None,
    ):
        self.node_ids_to_include = node_ids_to_include
        super().__init__(
            max_depth=max_depth,
            feature_names=feature_names,
            class_names=class_names,
            label='all',
            filled=filled,
            impurity=False,
            node_ids=False,
            proportion=proportion,
            rounded=rounded,
            precision=precision,
            fontsize=fontsize,
        )
    
    def _make_tree(self, node_id, et, criterion, depth=0):
        name = self.node_to_str(et, node_id, criterion=criterion)
        if (
            self.node_ids_to_include is not None
            and node_id not in self.node_ids_to_include
        ):
            name = self.node_to_str(et, node_id, criterion=criterion)
            if not name.startswith('samples'):
                splits = name.split('\n')
                splits[0] = 'null'                
                name = '\n'.join(splits)
            return Tree(name, node_id)
        return super()._make_tree(node_id, et, criterion, depth)

    def recurse(self, node, tree, ax, max_x, max_y, depth=0):
        kwargs = dict(
            bbox=self.bbox_args.copy(),
            ha="center",
            va="center",
            zorder=100 - 10 * depth,
            xycoords="axes fraction",
            arrowprops=self.arrow_args.copy(),
        )
        kwargs["arrowprops"]["edgecolor"] = plt.rcParams["text.color"]

        if self.fontsize is not None:
            kwargs["fontsize"] = self.fontsize

        xy = ((node.x + 0.5) / max_x, (max_y - node.y - 0.5) / max_y)
            
        if self.max_depth is None or depth <= self.max_depth:
            if self.filled:
                kwargs["bbox"]["fc"] = self.get_fill_color(tree, node.tree.node_id)
            else:
                kwargs["bbox"]["fc"] = ax.get_facecolor()

            if node.parent is None:
                ax.annotate(node.tree.label, xy, **kwargs)
            else:
                xy_parent = (
                    (node.parent.x + 0.5) / max_x,
                    (max_y - node.parent.y - 0.5) / max_y,
                )
                if node.tree.label.startswith('null'):
                    samples_match = re.search(r'samples = ([\d.]+)%', node.tree.label)
                    value_match = re.search(r'value = \[[\d.]+, ([\d.]+)\]', node.tree.label)
                    value_group = str(value_match.group(0))
                    sample_group = str(samples_match.group(0))

                    kwargs["bbox"]["fc"] = "lightgrey"
                    ax.annotate("(...)\n" + sample_group + "\n" + value_group, xy_parent, xy, **kwargs)
                else:
                    ax.annotate(node.tree.label, xy_parent, xy, **kwargs)
            if not node.tree.label.startswith('null'):
                for child in node.children:
                    self.recurse(child, tree, ax, max_x, max_y, depth=depth + 1)
        else:
            xy_parent = (
                (node.parent.x + 0.5) / max_x,
                (max_y - node.parent.y - 0.5) / max_y,
            )
            kwargs["bbox"]["fc"] = "lightgrey"
            ax.annotate("\n  (...)  \n", xy_parent, xy, **kwargs)


def get_node_ids_along_path(tree, path):
    node_ids = [0]

    for direction in path:
        i = node_ids[-1]
        if direction == 'l':
            child = tree.tree_.children_left[i]
            if child != -1:
                node_ids.append(child)
        elif direction == 'r':
            child = tree.tree_.children_right[i]
            if child != -1:
                node_ids.append(child)
        else:
            raise ValueError("Invalid direction. Use 'l' for left or 'r' for right.")

    return node_ids


def plot_tree(
    decision_tree,
    max_depth=None,
    feature_names=None,
    filled=True,
    proportion=True,
    rounded=True,
    precision=2,
    fontsize=None,
    ax=None,
    node_ids_to_include=None,
    label_mapper={},
    formatter=None,
    annotate_arrows=False,
):
    exporter = TreeExporter(
        max_depth=max_depth,
        feature_names=feature_names,
        filled=filled,
        proportion=proportion,
        rounded=rounded,
        precision=precision,
        fontsize=fontsize,
        node_ids_to_include=node_ids_to_include,
    )
    annotations = exporter.export(decision_tree, ax=ax)

    if ax is None:
        ax = plt.gca()

    x0, y0 = annotations[0].get_position()
    x1, y1 = annotations[1].get_position()
    #x2 = y2 = 0

    renderer = ax.figure.canvas.get_renderer()
    for annotation in annotations:
        #x, y = annotation.get_position()
        #if x > x0 and y > y2:
        #    x2 = x
        #    y2 = y

        text = annotation.get_text()
        if text.startswith('samples'):
            # Leaf node
            if formatter is not None:
                s, v = text.split('\n')
                s, v = formatter(s, v)
                text = '\n'.join([s, v])
        elif text.startswith('\n'):
            _, l, s, v = text.split('\n')
            if formatter is not None:
                s, v = formatter(s, v)
            text = '\n'.join([l, s, v])
        else:
            # Inner node
            l, s, v = text.split('\n')
            if l in label_mapper:
                l = label_mapper[l]
            elif re.match(r'duration_ra\s+<=\s+\w+', l):
                l1, l2 = l.split(' <= ')
                l1 = label_mapper.get(l1, l1)
                l2 = float(l2)
                l = l1 + ' $\leq$ ' + '{:.{prec}f} years'.format(l2, prec=precision)
            elif re.match(r'\w+\s+<=\s+\w+', l):
                l1, l2 = l.split(' <= ')
                l1 = label_mapper.get(l1, l1)
                l2 = float(l2)
                l = l1 + ' $\leq$ ' + '{:.{prec}f}'.format(l2, prec=precision)
            if formatter is not None:
                s, v = formatter(s, v)
            text = '\n'.join([l, s, v])
        annotation.set_text(text)
        annotation.set(ha='center')
        annotation.draw(renderer)

    if annotate_arrows:
        kwargs = dict(
            ha='center',
            va='center',
            fontsize=fontsize,
        )
        ax.annotate('True', (x1 + (x0-x1) / 2, y0 - (y0-y1) / 3), **kwargs)
        ax.annotate('False', (x0 + (x0-x1) / 2, y0 - (y0-y1) / 3), **kwargs)


def describe_categorical(D):
    assert (D.apply(is_categorical_dtype) | D.apply(is_bool_dtype)).all()
    
    index_tuples = []
    out = {'Counts': [], 'Proportions': []}

    for v in D:
        s = D[v]

        if is_bool_dtype(s):
            s = s.astype('category')
            s = s.cat.rename_categories({True: 'yes', False: 'no'})
        
        table = pd.Categorical(s).describe()
        
        # Exclude NaNs when computing proportions.
        N = table.counts.values[:-1].sum() if -1 in table.index.codes \
            else table.counts.values.sum()
        proportion = [round(100 * x, 1) for x in table.counts / N]
        table.insert(1, 'proportion', proportion)

        try:
            from .corevitas import variables
            categories = variables[s.name].pop('categories', None)
        except ModuleNotFoundError:
            categories = None
        except KeyError:
            categories = None
        
        if categories is not None:
            table.index = table.index.rename_categories(
                categories
            )

        for c in table.index:
            index_tuples.append((v, N, c))
        out['Counts'].extend(table.counts)
        out['Proportions'].extend(table.proportion)

    index = pd.MultiIndex.from_tuples(
        index_tuples, names=['Variable', 'No. samples', 'Value']
    )
    return pd.DataFrame(out, index=index)


def describe_numerical(data):
    table = data.describe().T
    table.rename(columns={'count': 'No. samples'}, inplace=True)
    return table.drop(
        columns=['mean', 'std', 'min', 'max']
    )


def display_dataframe(
    df,
    caption=None,
    new_index=None,
    new_columns=None,
    hide_index=False,
    precision=2
):
    def set_style(styler):
        if caption is not None:
            styler.set_caption(caption)
        if new_index is not None:
            styler.relabel_index(new_index, axis=0)
        if new_columns is not None:
            styler.relabel_index(new_columns, axis=1)
        if hide_index:
            styler.hide(axis='index')
        styler.format(precision=precision)
        return styler
    
    display_everything = (
        'display.max_rows', None,
        'display.max_columns', None,
    )
    
    with pd.option_context(*display_everything):
        return df.style.pipe(set_style)


def get_all_scores(all_experiment_paths):
    all_scores = []
    
    for experiment, experiment_paths in all_experiment_paths.items():
        if experiment_paths is None:
            continue
        for state, experiment_path in experiment_paths.items():
            if experiment_path is None:
                continue
            scores_path = os.path.join(experiment_path, 'scores.csv')
            if not os.path.exists(scores_path):
                continue
            scores = pd.read_csv(scores_path)
    
            # Load a pipeline to compute the dimensionality of the input.
            estimator = 'rnn' if state == '$H_t$' else 'lr'
            pipeline, results_path = load_best_pipeline(
                experiment_path, 1, estimator, return_results_path=True)
    
            # Load data handler.
            config_path = os.path.join(results_path, 'config.yaml')
            config = load_config(config_path)
            data_handler = get_data_handler_from_config(config)
            
            scores['data'] = experiment
            scores['state'] = state
            state_dim = pipeline.n_features_in_
            if data_handler.GROUP in pipeline.feature_names_in_:
                state_dim -= 1
            scores['state_dim'] = state_dim
            all_scores.append(scores)
    
    all_scores = pd.concat(all_scores)
    all_scores.rename(columns={'estimator_name': 'estimator'}, inplace=True)
    
    return all_scores


def get_scoring_table(
    all_scores,
    metric='auc',
    include_cis=False,
    exclude_models=[],
    experiment_order=None,
    model_order=None,
    index=None,
):
    by = ['data', 'state', 'state_dim', 'estimator']
    g = all_scores[all_scores.subset == 'test'].groupby(by)
    
    agg = EstimateAggregator(np.mean, 'ci', n_boot=1000, seed=0)
    
    table = g.apply(agg, var=metric)
    table = table * 100  # Convert to percentage

    if include_cis:
        a = r'\begin{tabular}[c]{@{}l@{}}'
        b = r'\end{tabular}'
        f = lambda r: a + f"{r[metric]:.1f}\\({r[f'{metric}min']:.1f}, {r[f'{metric}max']:.1f})" + b
    else:
        f = lambda r: f'{r[metric]:.1f}'
    table[metric] = table[[metric, f'{metric}min', f'{metric}max']].apply(f, axis=1)
    table = table.drop(columns=[f'{metric}min', f'{metric}max'])
    
    table = table.unstack(-1)
    table.columns = table.columns.droplevel()  # Drop metric level
    
    sequence_models = ['prosenet', 'rdt', 'rdt_aligned', 'rdt_pruned', 'rnn']
    for c in sequence_models:
        c_truncated = f'truncated_{c}'
        if c_truncated in table:
            table[c].fillna(table[c_truncated], inplace=True)
            table.drop(columns=c_truncated, inplace=True)
    
    exclude_models = [m for m in exclude_models if m in table.columns]
    table = table.drop(columns=exclude_models)
        
    table = table.fillna('-')

    if experiment_order is not None:
        table = table.reindex(experiment_order, level=0)

    if model_order is not None:
        table = table[[m for m in model_order if m in table.columns]]
        table = table.rename(columns=model_order)
    
    if index is not None:
        table = table.reindex(index, level=1)

    return table


def compare_ra_models(
    probas,
    data_path,
    state_true='$H_t$',
    estimator_true='rnn',
    state_pred='$A_{t-1}$',
    estimator_pred='lr',
    num_trials=5,
    switches_only=False,
    compare_with_gt=False,
    return_proba=False,
):
    from inpole.data import RAData
    
    y_true_all, y_pred_all = [], []
    yp_true_all, yp_pred_all = [], []
    X_all, y_all = [], []
    
    for trial in range(1, num_trials + 1):
        data_handler = RAData(path=data_path, seed=trial)
        _train_data, _valid_data, test_data = data_handler.get_splits()
        X, y = test_data
        X['stage'] = X.groupby(data_handler.GROUP).cumcount() + 1
        
        yp_true = probas[
            probas.State.eq(state_true)
            & probas.Estimator.eq(estimator_true) 
            & probas.Trial.eq(trial)
        ].Probas.item()
    
        yp_pred = probas[
            probas.State.eq(state_pred) 
            & probas.Estimator.eq(estimator_pred) 
            & probas.Trial.eq(trial)
        ].Probas.item()
    
        y_true = np.argmax(yp_true, axis=1)
        y_pred = np.argmax(yp_pred, axis=1)

        if compare_with_gt:
            labels = data_handler.get_labels()
            yp_true = np.eye(len(labels))[y]
            y_true = y

        if switches_only:
            labels = data_handler.get_labels()
            switch = (labels[y] != X.prev_therapy)
            
            yp_true = yp_true[switch]
            yp_pred = yp_pred[switch]
            
            y_true = y_true[switch]
            y_pred = y_pred[switch]
            
            X = X[switch]
            y = y[switch]
        
        X['correct'] = y_true == y_pred

        yp_true_all.append(yp_true)
        yp_pred_all.append(yp_pred)
        
        y_true_all.append(y_true)
        y_pred_all.append(y_pred)
        
        X_all.append(X)
        y_all.append(y)

    yp_true_all = np.concatenate(yp_true_all)
    yp_pred_all = np.concatenate(yp_pred_all)
    
    y_true_all = np.concatenate(y_true_all)
    y_pred_all = np.concatenate(y_pred_all)
    
    X_all = pd.concat(X_all)
    y_all = np.concatenate(y_all)

    _, unique_indices = np.unique(X_all.index, return_index=True)
    unique_indices = np.sort(unique_indices)

    yp_true_all = yp_true_all[unique_indices]
    yp_pred_all = yp_pred_all[unique_indices]
    
    y_true_all = y_true_all[unique_indices]
    y_pred_all = y_pred_all[unique_indices]

    X_all = X_all.iloc[unique_indices]
    y_all = y_all[unique_indices]

    if return_proba:            
        return y_true_all, y_pred_all, yp_true_all, yp_pred_all, X_all, y_all
    else:
        return y_true_all, y_pred_all, X_all, y_all
