import numpy as np
import pandas as pd
import seaborn as sns

from matplotlib import pyplot as plt

from collections import Counter

"""
# Analyzing NOSC of the E. coli proteome
This script calculates the condition dependent proteome NOSC for the datasets aggregated by Chure & Bellivue et al. Cell Systems 2021. To calculate the proteome NOSC we need near-complete proteome datasets such that the uncertainty related to unmeasured proteins is small. Recent datasets of this quality were collected by Griffin Chure and Nathan Bellivue. 

Since proteome NOSC is a weighted average of contributions from individual proteins, we do not need absolute measurements with real units. Rather, we need accurate and complete compositional data, i.e. protein A makes P% of the proteome. 

# Differences between lab practices
Valgepea 2013 uses data collected in Valgepea 2010, where E. coli was cultured in a custom minimal medium documented in Nahku et al. 2010. Peebo et al. 2015 is from the same group (Vilu) and appears to use the same growth medium base with some amino acid supplementation. 

Minimal media used by Schmidt et al. 2015 uses an M9 base that is distinct from the medium in the Vilu group. 

# Known issues
I am using MG1655 coding sequences for all samples, but Schmidt et al. and Peebo et al. are working with BW25113. The BW25113 is derived from MG1655 with a small number deletions and other changes documented in the genome announcement. https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4200154/

# Some additional E. coli data sources might be considered in future. 
* Hui et al. MSB 2015 is from Hwa & Williamson groups
* Maser et al. 2021 from Vilu and Nahku groups
* Mori et al. 2022 
"""

print('Loading data...')
raw_abund_df = pd.read_csv(
    'data/proteomes/Coli/Chure_compiled_absolute_measurements.csv', index_col=0).reset_index().set_index('b_number')
nosc_df = pd.read_csv(
    'data/genomes/Coli/MG1655/MG1655_ref_prot_NOSC.csv').set_index('b_number')

# Take the mean of replicates for the same gene. 
print('Aggregating replicate measurements...')
# Note 1: have to do this because didn't report which measurement is from which replicate.
# Note 2: the Schmidt data is already aggregated across two measurement methods.
# NB: Assuming same dataset, strain, cond, gene and growth rate implies replicate.
index_cols = 'dataset,strain,condition,b_number,growth_rate_hr'.split(',')
counts = raw_abund_df.reset_index().groupby(index_cols).mean(numeric_only=True)
# renaming tot_per_cell for clarity and to match how we treat other datasets
abund_df = counts.reset_index().set_index('b_number').rename(
    columns=dict(tot_per_cell='copies_per_cell'))

# Keep only the b-numbers that we have abundance and NOSC data for.
print('Filtering to b-numbers with Z_C data...')
overlapping_idx = list(set(nosc_df.index.values).intersection(abund_df.index.values))
abund_df = abund_df.loc[overlapping_idx].copy()
nosc_df = nosc_df.loc[overlapping_idx].copy()

# Add NOSC and NC data to the abundance dataframe
print('Adding sequence characteristics...')
abund_df['NOSC'] = nosc_df.loc[abund_df.index.values].NOSC
abund_df['NC_per'] = nosc_df.loc[abund_df.index.values].NC
abund_df['num_aas'] = nosc_df.loc[abund_df.index.values].num_aas
abund_df['mw_daltons'] = nosc_df.loc[abund_df.index.values].mw_daltons
abund_df['organism_key'] = 'coli'
abund_df['species'] = 'E. coli'
abund_df['majority_protein_ids'] = nosc_df.loc[abund_df.index.values].primary_accession
abund_df['fraction_transmembrane'] = nosc_df.loc[abund_df.index.values].fraction_transmembrane
abund_df['fraction_transmembrane_C'] = nosc_df.loc[abund_df.index.values].fraction_transmembrane

# Add annotation of the growth mode -- presumed batch
print('Adding metadata about the experiment...')
abund_df['growth_mode'] = 'batch'
# Conditions called "chemostat" are definitely chemostat conds
abund_df.loc[abund_df.condition.str.startswith('chemostat'), 'growth_mode'] = 'chemostat'
# Li et al. data is all batch culture - can leave untouched.
# Peebo et al. data is all in "accelerostats" which are chemostats
abund_df.loc[abund_df.dataset == 'peebo_2015', 'growth_mode'] = 'chemostat'
# Valgepea et al. data is also likewise 2013
abund_df.loc[abund_df.dataset == 'valgepea_2013', 'growth_mode'] = 'chemostat'

# Add annotation of stress conditions -- presumed non-stess
abund_df['stress'] = False
abund_df.loc[abund_df.condition == '42C', 'stress'] = True
abund_df.loc[abund_df.condition == 'osmotic_stress_glucose', 'stress'] = True
abund_df.loc[abund_df.condition == 'pH6', 'stress'] = True

# Formal Carbon-bound e- per protein copy (NOSC*NC)
abund_df['Ce_per'] = nosc_df.loc[abund_df.index].Ce
# Total Carbon-bound e-/cell on this protein, i.e. weighted by copies/cell 
abund_df['Ce_total'] = abund_df.Ce_per * abund_df.copies_per_cell
# Total Carbon atoms on this protein, i.e. weighted by copies/cell 
abund_df['NC_total'] = abund_df.NC_per * abund_df.copies_per_cell
abund_df.to_csv('data/proteomes/Coli/Chure_mean_absolute_measurements.csv')

# Calculate the proteome NOSC
print('Calculating proteome NOSC...')
index_cols = 'dataset,strain,condition,growth_rate_hr,growth_mode'.split(',')
proteome_nosc_df = abund_df.groupby(index_cols).sum()
proteome_nosc_df = proteome_nosc_df[['Ce_total', 'NC_total']].copy() 
proteome_nosc_df['proteome_NOSC'] = proteome_nosc_df.Ce_total / proteome_nosc_df.NC_total
proteome_nosc_df = proteome_nosc_df.reset_index()
proteome_nosc_df.to_csv('data/proteomes/Coli/Chure_proteome_NOSC.csv', index=False)