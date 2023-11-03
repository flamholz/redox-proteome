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

# To estimate the ZC effects we need to know the C mass fraction
# for each class of each macromolecule, which we now estimate.

# BNID 109413: chemical formula for protein C100 H159 N26 O32 S0.7
prot_Cmass_frac = 100*12/(100*12 + 159 + 26*14 + 32*16 + 0.7*32)

# Calculate approximate C mass fractions for DNA and RNA
# First build a dictionary mapping the names of the nucleotides
# to dictionaries of their elemental compositions
nucleotide_compositions = {
        "AMP": {"C": 10, "H": 12, "N": 5, "O": 7, "P": 2},
        "GMP": {"C": 10, "H": 12, "N": 5, "O": 8, "P": 2},
        "CMP": {"C": 9, "H": 12, "N": 3, "O": 8, "P": 2},
        "UMP": {"C": 9, "H": 11, "N": 2, "O": 9, "P": 2},
        "dAMP": {"C": 10, "H": 12, "N": 5, "O": 6, "P": 2},
        "dGMP": {"C": 10, "H": 12, "N": 5, "O": 7, "P": 2},
        "dCMP": {"C": 9, "H": 12, "N": 3, "O": 7, "P": 2},
        "dTMP": {"C": 10, "H": 14, "N": 2, "O": 6, "P": 2},
}
elemental_masses = {
        "C": 12,
        "H": 1,
        "N": 14,
        "O": 16,
        "P": 31,
}
elemental_order = ["C", "H", "N", "O", "P"]
masses_vec = np.array([elemental_masses[e] for e in elemental_order])
# Calculate the C mass fraction of each nucleotide
c_mass_fracs = {}
for nuc, composition in nucleotide_compositions.items():
    elt_counts = np.array([composition[e] for e in elemental_order])
    c_mass_fracs[nuc] = elt_counts[0]*masses_vec[0]/np.sum(elt_counts*masses_vec)

# Calculate the C mass composition of a uniform mixture of RNA nucleotides
RNA_nuc = ["{0}MP".format(l) for l in "AGCU"]
RNA_Cmass_frac = np.round(np.mean([c_mass_fracs[n] for n in RNA_nuc]), 2)
# Calculate the C mass composition of a uniform mixture of DNA nucleotides
DNA_nuc = ["d{0}MP".format(l) for l in "AGCT"]
DNA_Cmass_frac = np.round(np.mean([c_mass_fracs[n] for n in DNA_nuc]), 2)

# Estimate ZC changes
# TODO: need to account for the slightly different carbon content of RNA, DNA and protein
cols = ['protein_percent', 'RNA_percent', 'DNA_percent']
ZCs = np.array([-0.15, 0.9, 0.6])
Cmass_fracs = np.array([prot_Cmass_frac, RNA_Cmass_frac, DNA_Cmass_frac])

# Convert back to fractions
tmp = biomass_percent_df[cols].copy() / 100
inferred_ZCBs = (tmp * ZCs * Cmass_fracs / np.sum(Cmass_fracs)).sum(axis=1)
inferred_ZCBs.name = 'infered_ZCB'
inferred_ZCBs.round(3).to_csv(
        '../data/physiology/BremerDennis2008_InferredZCB.csv', float_format='%.3f')