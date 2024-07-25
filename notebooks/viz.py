import pandas as pd
import squarify as sq
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.path import Path
from matplotlib.patches import BoxStyle
from matplotlib.offsetbox import AnchoredText
import seaborn as sns

AX_FACECOLOR = '#E3DCD0'

# # Add Lucida Sans to the font manager
# import matplotlib
# fnames = 'LSANS,LSANSD,LSANSI,LSANSDI'.split(',')
# for fname in fnames:
#     matplotlib.font_manager.fontManager.addfont('/users/flamholz/Library/Fonts/{}.ttf'.format(fname))

# Style and useful function definitions.
def titlebox(
    ax, text, color, bgcolor=None, size=8, boxsize=0.1, pad=0.05, loc=10, **kwargs
):
    """Sets a colored box about the title with the width of the plot"""
    boxsize=str(boxsize * 100)  + '%'
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("top", size=boxsize, pad=pad)
    cax.get_xaxis().set_visible(False)
    cax.get_yaxis().set_visible(False)
    cax.spines["top"].set_visible(False)
    cax.spines["right"].set_visible(False)
    cax.spines["bottom"].set_visible(False)
    cax.spines["left"].set_visible(False)

    plt.setp(cax.spines.values(), color=color)
    if bgcolor != None:
        cax.set_facecolor(bgcolor)
    else:
        cax.set_facecolor("white")
    at = AnchoredText(text, loc=loc, frameon=False, prop=dict(size=size, color=color))
    cax.add_artist(at)

def color_palette():
    """
    Returns a dictionary of the PBOC color palette
    """
    return {'green': '#7AA974', 'light_green': '#BFD598',
            'pale_green': '#DCECCB', 'yellow': '#EAC264',
            'light_yellow': '#F3DAA9', 'pale_yellow': '#FFEDCE',
            'blue': '#738FC1', 'light_blue': '#A9BFE3',
            'pale_blue': '#C9D7EE', 'red': '#D56C55', 'light_red': '#E8B19D',
            'pale_red': '#F1D4C9', 'purple': '#AB85AC',
            'light_purple': '#D4C2D9', 'dark_green':'#7E9D90', 'dark_brown':'#905426',
            'dark_blue': '#535D87'}

def plotting_style(grid=False):
    """
    Sets the style to the publication style
    """
    rc = {
          #'axes.facecolor': '#FFFFFF',
          #'axes.facecolor': '#E3DCD0',
          'font.family': 'sans-serif',
          'font.sans-serif': ['Myriad Pro', 'Roboto', 'Tahoma', 'DejaVu Sans',
                              'Lucida Grande', 'Verdana'],
          'grid.linestyle': '-',
          'grid.linewidth': 0.5,
          'grid.alpha': 0.75,
          'grid.color': '#ffffff',
          'axes.grid': grid,
          'ytick.direction': 'in',
          'xtick.direction': 'in',
          'xtick.gridOn': grid,
          'ytick.gridOn': grid,
          'ytick.major.width': 5,
          'xtick.major.width': 5,
          'ytick.major.size': 5,
          'xtick.major.size': 5,
          'mathtext.fontset': 'stixsans',
          'mathtext.sf': 'sans',
          'legend.frameon': True,
          'legend.framealpha': 0,
          'legend.facecolor': '#FFEDCE',
          'figure.dpi': 150,
           'xtick.color': 'k',
           'ytick.color': 'k'}
    plt.rc('text.latex', preamble=r'\usepackage{sfmath}')
    plt.rc('mathtext', fontset='stixsans', sf='sans')
    sns.set_style('ticks', rc=rc)
    #sns.set_style('darkgrid', rc=rc)
    return color_palette()

