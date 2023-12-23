from os.path import join

import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator

from ..data.utils import SequentialDataset, TruncatedHistoryDataset


def expects_groups(estimator):
    if estimator.__class__.__name__ == 'SwitchPropensityEstimator':
        return all([expects_groups(e) for e in estimator.estimators])
    return (
        hasattr(estimator, 'dataset') and
        (
            estimator.dataset == SequentialDataset or
            estimator.dataset == TruncatedHistoryDataset
        )
    )


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
