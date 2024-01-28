import numpy as np
import pandas as pd
import seaborn as sns

from scipy.stats import linregress
from matplotlib import pyplot as plt

print('Loading lipid data...')

# Read and analyze Marr et al. Coli fatty-acid data. 
marr_conds = pd.read_excel('data/lipids/Coli/Marr1962_JBac.xlsx', sheet_name='conditions', index_col=0)
marr_fas = pd.read_excel('data/lipids/Coli/Marr1962_JBac.xlsx', sheet_name='fatty_acids', index_col=1)
marr_data = pd.read_excel('data/lipids/Coli/Marr1962_JBac.xlsx', sheet_name='tables_1_2_3',
                          index_col=1).replace({np.NaN: 0, '': 0}).infer_objects()

# Grab the relevant data and convert to a long-form dataframe. Values are in weight percent. 
marr_data_long = marr_data.reset_index().melt(
    id_vars='short_name', value_vars=marr_data.columns[1:], var_name='condition', value_name='weight_percent')

# Add NC, MW and NOSC information to the long-form DF
NC = marr_fas.loc[marr_data_long.short_name].NC.values
mw = marr_fas.loc[marr_data_long.short_name].mw_daltons.values
NOSC = marr_fas.loc[marr_data_long.short_name].NOSC.values

marr_data_long['NC'] = NC
marr_data_long['mw_daltons'] = mw
marr_data_long['NOSC'] = NOSC

print('Calculating C mass fractions for lipids...')
# Calculate the fraction of lipid C atoms on each molecule
# Convert to carbon units and then rescale. 
marr_data_long['C_fraction'] = marr_data_long.weight_percent*NC/mw
C_sums = marr_data_long.groupby('condition').sum().C_fraction
marr_data_long.C_fraction /= C_sums.loc[marr_data_long.condition].values

# Save the long-form version
marr_data_long.to_csv('data/lipids/Coli/Marr1962_JBac_long.csv', index=False)

print('Calculating total lipid Z_C...')

# Fraction of C atoms x NOSC is related to total valence e- on that lipid.
# The sum of these terms is just the total lipid NOSC. 
marr_data_long['lipid_NOSC'] = marr_data_long.C_fraction * marr_data_long.NOSC

# Model for T-dependent growth rate of E. coli in a similar glucose
# minimal medium to Marr. See Table 1 of Gill 1985 for fit parameters 
b = 0.0262
c = 0.298
Tmax = 47.3
Tmin = 4.9
temps = np.arange(10, 36, 5)
pred_mu = lambda temps: np.power(b*(temps-Tmin)*(1-np.exp(c*(temps-Tmax))), 2)

# Calculate the per-sample total lipid NOSC (with all the assumptions)
total_lipids_nosc = marr_data_long.groupby('condition').agg(dict(lipid_NOSC='sum'))
total_lipids_nosc['reported_growth_rate_hr'] = marr_conds.loc[total_lipids_nosc.index].growth_rate_hr.values
total_lipids_nosc['temp_C'] = marr_conds.loc[total_lipids_nosc.index].temp_C.values
total_lipids_nosc['experiment'] = marr_conds.loc[total_lipids_nosc.index].experiment.values
total_lipids_nosc['inferred_growth_rate_hr'] = np.NaN

# Calculate the inferred growth rate for "temp" experiment
# The model above is only appropriate for that medium. 
mask = total_lipids_nosc.experiment == 'temp'
ts = total_lipids_nosc[mask].temp_C.values
inferred_mus = pred_mu(ts)
total_lipids_nosc.loc[mask, 'inferred_growth_rate_hr'] = np.round(inferred_mus, 3)

total_lipids_nosc.to_csv('data/lipids/Coli/Marr1962_total_lipids_NOSC.csv')


