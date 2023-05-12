from scipy.stats.kde import gaussian_kde
from scipy.stats import norm

import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

__author__ = 'Avi Flamholz'

"""
Utility functions for plotting.
"""


def ridgeplot(long_df, categorical_key, value_key, overlap=0, cat_order=None, palette=None, labels=None, n_points=150):
    """
    Creates a standard ridgeline plot -- plotting smoothed distributions for each category.

    Args:
        long_df (pandas.DataFrame): long format dataframe with the data to plot.
        categorical_key (str): name of the column defining categories to plot ridges for.
        value_key (str): name of the column containing values to plot distributions for.
        overlap (float): overlap between adjacent plots. 1 max overlap, 0 no overlap.
        cat_order (list): order to plot the categories in.
        palette (list): list of colors to use for the categories.
        labels (dict): dictionary mapping categories to labels.
        n_points (int): number of points to evaluate each distribution function.
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