import numpy as np
import pandas as pd
import seaborn as sns

from collections import Counter

"""
# Analyzing NOSC of the S. cerevisiae proteome
Data from Xia et al. Nat Comms 2022. According to the "description of supplementary files" for that paper, the expression data are in units of molecules per cell. 

# Known issues
Expression data contains ≈50 proteins with multiple majority IDs. This is currently resolved by adding fictional IDs with data for the first of those IDs. Mostly these are small sequence variants with almost no effects on NOSC. 
"""

print('Loading data...')
samples_df = pd.read_excel('data/proteomes/Scer/Xia_ScerCEN.PK.xlsx', sheet_name='samples', index_col=0)
raw_abund_df = pd.read_excel('data/proteomes/Scer/Xia_ScerCEN.PK.xlsx', sheet_name='data')
nosc_df = pd.read_csv('data/genomes/Scer/S288c/S288c_ref_prot_NOSC.csv')

abund_ids = set(raw_abund_df.majority_protein_ids.values.tolist())
cds_ids = set(nosc_df.primary_accession.values.tolist())

# There are ≈50 entries where there was > 1 majority hit.
# That is: the relevant peptides report on multiple proteins, often
# because they are alternate translations of the same gene. 
missing_ids = abund_ids.difference(cds_ids)
print('Missing {0} ids'.format(len(missing_ids)))
shared_ids = abund_ids.intersection(cds_ids)

# The missing IDs are mostly due to isoforms of proteins that differ slightly in sequence. 
# This code identifies the individual IDs and makes a fictional row that represents the average
# of each of the isoforms.
print('Adding fictional IDs for those representing a mixture of isoforms')
lookup_table = nosc_df.set_index('primary_accession')
fakes = []
for my_id in missing_ids:
    NCs = []
    Ces = []
    for x in my_id.split(':'):
        if x in lookup_table.index:
            row = lookup_table.loc[x]
            NCs.append(row.NC)
            Ces.append(row.Ce)
            
    if len(NCs) == 0:
        continue
    print('Adding fictional protein for {0} representing {1} isoforms'.format(
        my_id, len(NCs)))
    
    NC = np.mean(NCs)
    Ce = np.mean(Ces)
    fake_protein = dict(primary_accession=my_id, NC=NC, Ce=Ce, NOSC=(Ce/NC))
    fakes.append(fake_protein)
    
extended_nosc_df = pd.concat([nosc_df, pd.DataFrame(fakes)], ignore_index=True)

# recheck which IDs are missing
cds_ids = set(extended_nosc_df.primary_accession.values.tolist())
missing_ids = abund_ids.difference(cds_ids)
print('After update, missing {0} IDs'.format(len(missing_ids)))
shared_ids = abund_ids.intersection(cds_ids)

# Checking the percentage of unmapped genes. 
data_cols = raw_abund_df.columns[2:-1]
shared_ids_list = list(shared_ids)
mapped_sum = raw_abund_df.set_index('majority_protein_ids')[data_cols].loc[shared_ids_list].sum()
total = raw_abund_df[data_cols].sum()
pct_diff = 100*(total - mapped_sum)/total

# Now that we've handled the isoforms, we're counting all the expression data
print('Consistency check: have we accounted for all the expression data?')
print(pct_diff.head())

# Reshape the data to long-form
long_abund_df = raw_abund_df.drop('gene_function', axis=1).melt(
    id_vars=['majority_protein_ids', 'gene_name'], var_name='sample_name',
    value_name='copies_per_cell')

growth_rates = samples_df.loc[long_abund_df.sample_name].mu_per_hr
long_abund_df['growth_rate_hr'] = growth_rates.values

# use the extended_nosc_df to calculate the condition-dependent proteome NOSC
print('Adding sequence metadata...')
my_nosc_df = extended_nosc_df.set_index('primary_accession')
NCs = my_nosc_df.loc[long_abund_df.majority_protein_ids].NC.values
Ces = my_nosc_df.loc[long_abund_df.majority_protein_ids].Ce.values
NOSCs = my_nosc_df.loc[long_abund_df.majority_protein_ids].NOSC.values
long_abund_df['NC_per'] = NCs
long_abund_df['Ce_per'] = Ces
long_abund_df['NOSC'] = NOSCs
long_abund_df['NC_total'] = long_abund_df.copies_per_cell.multiply(NCs)
long_abund_df['Ce_total'] = long_abund_df.copies_per_cell.multiply(Ces)
long_abund_df['dataset'] = 'xia_2022'
long_abund_df['strain'] = 'CEN.PK113-7D'
long_abund_df['species'] = 'S. cerevisiae'
long_abund_df['organism_key'] = 'yeast'
long_abund_df['condition'] = 'chemostat_u' + samples_df.loc[long_abund_df.sample_name].mu_per_hr.astype(str).values
long_abund_df['fraction_transmembrane'] = my_nosc_df.loc[long_abund_df.majority_protein_ids].fraction_transmembrane.values
long_abund_df['fraction_transmembrane_C'] = my_nosc_df.loc[long_abund_df.majority_protein_ids].fraction_transmembrane_C.values

# Add annotation of the growth mode -- everything in this ref was done in chemostats
long_abund_df['growth_mode'] = 'chemostat'
# Add annotation of stress conds -- these are all glucose chemostat conds
long_abund_df['stress'] = False

# Save to CSV
print('Saving to abundance CSV...')
long_abund_df.to_csv('data/proteomes/Scer/Xia_protein_measurements.csv', index=False)

agg_dict = dict(NC_total='sum', Ce_total='sum', growth_rate_hr='first', growth_mode='first')
sample_noscs = long_abund_df.groupby(['sample_name']).agg(agg_dict)
sample_noscs['proteome_NOSC'] = sample_noscs.Ce_total / sample_noscs.NC_total

# reset growth rates -- didn't want to sum them
sample_noscs['growth_rate_hr'] = samples_df.loc[sample_noscs.index].mu_per_hr

print('Saving full replicate Z_C data to CSV...')
sample_noscs.to_csv('data/proteomes/Scer/Xia_proteome_NOSC_full.csv', index=False)

# Mean of S. cer data since the replicates are reported separately
print('Saving mean Z_C data to CSV...')
agg_dict = dict(NC_total='mean', Ce_total='mean', proteome_NOSC='mean', sample_name='count')
sample_noscs_mean = sample_noscs.reset_index().groupby('growth_rate_hr').agg(agg_dict).rename(
    columns=dict(sample_name='sample_count'))
sample_noscs_mean['dataset'] = 'xia_2022'
sample_noscs_mean['strain'] = 'CEN.PK113-7D'
sample_noscs_mean['condition'] = 'chemostat_u' + sample_noscs_mean.index.astype(str)
sample_noscs_mean['growth_mode'] = 'chemostat'

sample_noscs_mean.to_csv('data/proteomes/Scer/Xia_proteome_NOSC.csv', index=True)
sample_noscs_mean