from scipy.stats.kde import gaussian_kde
from scipy.stats import norm

import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt


def ridgeplot(long_df, categorical_key, value_key, overlap=0, cat_order=None, palette=None, labels=None, n_points=150):
    """
    Creates a standard ridgeline plot.

    data, list of lists.
    overlap, overlap between distributions. 1 max overlap, 0 no overlap.
    fill, matplotlib color to fill the distributions.
    n_points, number of points to evaluate each distribution function.
    labels, values to place on the y axis to describe the distributions.
    """    
    if overlap > 1 or overlap < 0:
        raise ValueError('overlap must be in [0 1]')
    
    m = long_df[categorical_key].notnull()
    cats = sorted(long_df[m][categorical_key].unique())
    my_cats = cat_order or cats
    n_ridges = len(my_cats)
    pal = palette or sns.color_palette(n_colors=n_ridges)
    color_dict = dict((c, pal[i]) for i, c in enumerate(my_cats))

    all_vals = long_df[long_df[value_key].notnull()][value_key].values
    n_ridges = long_df[categorical_key].unique().size
    xx = np.linspace(np.min(all_vals), np.max(all_vals), n_points)
    curves = []
    ys = []
    
    for i, (gid) in enumerate(my_cats):
        gdf = long_df[long_df[categorical_key] == gid]
        mask = gdf[value_key].notnull()
        vals = gdf[mask][value_key].values
        pdf = gaussian_kde(vals)
        y = i*(1.0-overlap)
        ys.append(y)
        
        # calculate the smoothed PDF and normalize to 0-1 Y range
        curve = pdf(xx)
        curve /= curve.max()
        
        # plot it and fill.
        plt.fill_between(xx, np.ones(n_points)*y, 
                         curve+y, zorder=n_ridges-i+1, color=color_dict[gid])
        plt.plot(xx, curve+y, c='k', lw=0.75, zorder=n_ridges-i+1)
        plt.axhline(y, c='k', lw=1, zorder=n_ridges-i+1)
    
    # label the curves on the y-axis
    my_labels = my_cats
    if labels:
        my_labels = [labels[c] for c in my_cats]
    plt.yticks(ys, my_labels, fontsize=8)