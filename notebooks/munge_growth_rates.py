import numpy as np
import pandas as pd 
import viz

colors = viz.plotting_style()
lin_colors = colors.values()

# Manually curated collection of fast growers
fast_growers_df = pd.read_csv('../data/physiology/fastest_growers.csv')
# Filter out the ones we don't want to plot, aggregate remainder by species and DOI
agg_dict = {
    'group': 'first',
    'category': 'first',
    'generation_time_hr': 'mean',
    'growth_rate_hr': 'mean',
    'tmp_C': 'first',
    'growth_mode': 'first',
    'high_CO2': 'first',
    'light': 'first',
}
fast_growers_df = fast_growers_df[fast_growers_df.to_plot].groupby(
    'short_name,DOI'.split(',')).agg(agg_dict).reset_index()

# Add color and positional information
my_colors = 'purple,green'.split(',')
pal = [colors[c] for c in my_colors]

np.random.seed(42)
fast_growers_df['xpos'] = fast_growers_df['group'].map({'autotroph': 1, 'heterotroph': 0})
fast_growers_df['xpos'] += np.random.uniform(-0.2, 0.2, size=fast_growers_df.xpos.size)
color_map = {'Cyanobacteria': colors['green'], 'Eukaryotic alga': colors['dark_green'], 
             'Heterotrophic bacteria': colors['purple'], 'Chemoautotroph': colors['yellow']}
fast_growers_df['color'] = fast_growers_df.category.map(color_map)
fast_growers_df['marker'] = fast_growers_df.short_name.map(
    {'C. ohadii': 'o', 'V. natrigens': '^', 'S. elongatus PCC 11801': '>', 'T. crunogena': '<',
     'E. coli': 'h', 'C. perfingens': 'd', 'C. necator': '*'})

fast_growers_df.to_csv('../data/physiology/fastest_growers_4plotting.csv', index=False)