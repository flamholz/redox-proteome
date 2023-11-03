import numpy as np
import pandas as pd

# Load biomass macromolecular composition from Bremer & Dennis 2008
bd_biomass_df = pd.read_excel('../data/physiology/BremerDennis_EcoSalPlus_2008.xlsx', index_col=0).T

# Calculate percent of biomass composition for RNA, protein and DNA as a function of growth rate.
cols = ['protein_ug_per_cell',
        'RNA_ug_per_cell',
        'DNA_ug_per_cell']
locs = [v for v in bd_biomass_df.index.values
        if v.startswith('t_')]

# Extract only the rows/columns we want
biomass_composition_df = bd_biomass_df[cols].loc[locs].copy().infer_objects().round(4)

# Get the total mass per cell from the parent DF
total_ug_per_cell = bd_biomass_df.loc[locs, 'mass_ug_per_cell']

# Bremer and Dennis have an odd convention where the factor of log(2) 
# is missing from the exponential growth rate reported there as mu. 
# We extract the doubling time from the index and convert to growth rate. 
biomass_composition_df['doubling_time_min'] = [float(v.split('_')[1]) for v in biomass_composition_df.index.values]
biomass_composition_df['doubling_time_hr'] = biomass_composition_df.doubling_time_min / 60
biomass_composition_df['growth_rate_hr'] = np.log(2) / biomass_composition_df.doubling_time_hr

# Calculate the mass fractions
biomass_percent_df = biomass_composition_df.copy()
biomass_percent_df[cols] = biomass_percent_df[cols].divide(total_ug_per_cell, axis=0)
biomass_percent_df[cols] = biomass_percent_df[cols].multiply(100) # pct
biomass_percent_df.columns = [c.replace('_ug_per_cell', '_percent') for c in biomass_percent_df.columns]
biomass_percent_df.round(3).to_csv(
    '../data/physiology/BremerDennis2008_BiomassComposition_pct.csv', float_format='%.3f')