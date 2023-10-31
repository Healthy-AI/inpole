from datetime import timedelta

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib import gridspec

from sklearn.decomposition import PCA


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


def visualize_prototype_ra(X, prototype_index, color_mapper):
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
    