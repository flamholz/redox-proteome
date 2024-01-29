import numpy as np
import pandas as pd
import seaborn as sns


"""
# Analyzing NOSC of the Cyanobacterial (PCC6803) proteome
Data from Zavrel et al. Elife 2019

# Known issues
Using a reference genome for 6803 GT-S, but the data for GT-L. Mapping is by gene name, which is imperfect. 
"""

print('Loading data...')
conds_df = pd.read_csv('data/proteomes/Synechocystis/Zavrel_PCC6803_conditions.csv', dtype=dict(cond_id='str')).set_index('cond_id')
nosc_df = pd.read_csv('data/genomes/Synechocystis/PCC6803/PCC6803_ref_prot_NOSC.csv')

# add a majority_protein_ids in a format matching the yeast data
raw_abund_df = pd.read_csv('data/proteomes/Synechocystis/Zavrel_PCC6803_proteome.csv')
raw_abund_df['majority_protein_ids'] = [':'.join(ids.split(';')) for ids in raw_abund_df['Majority protein IDs']]
raw_abund_df = raw_abund_df.set_index('majority_protein_ids')

# Find the overlap between the two sets of data
print('Harmonizing expression and genome data...')
abund_ids = set(raw_abund_df.index.values.tolist())
cds_ids = set(nosc_df.primary_accession.values.tolist())

# There are ≈50 entries where there was > 1 majority hit.
# That is: the relevant peptides report on multiple proteins, often
# because they are alternate translations of the same gene. 

# missing IDs have abundance data but no NOSC data
missing_ids = abund_ids.difference(cds_ids)
print(missing_ids)
print(len(missing_ids), 'not found')

# shared_ids have both. 
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

# Recheck which IDs are missing
cds_ids = set(extended_nosc_df.primary_accession.values.tolist())
missing_ids = abund_ids.difference(cds_ids)
print('After update, missing {0} IDs'.format(len(missing_ids)))
print(missing_ids)
shared_ids = abund_ids.intersection(cds_ids)

# Checking the percentage of expression accounted for by unmapped genes. 
data_cols = ['27.5', '55', '110', '220', '440', '1100']
mapped_sum = raw_abund_df[data_cols].loc[list(shared_ids)].sum()
total = raw_abund_df[data_cols].sum()
pct_diff = 100*(total - mapped_sum)/total

# Missing only ≈0.05% of the proteome due to mapping failure.
# Should be OK to proceed. 
print('Percent of proteome unmapped due to mapping failure:')
print(pct_diff)

data_cols = ['27.5', '55', '110', '220', '440', '1100']
long_abund_df = raw_abund_df.reset_index().melt(
    id_vars=['majority_protein_ids', 'Gene names', 'Length', 'Mol. weight [kDa]'],
    value_vars=data_cols,
    var_name='red_light_intensity_uE_m_s',
    value_name='copies_per_cell').rename(
    columns={'Gene names': 'gene_name',
             'Mol. weight [kDa]': 'mw_daltons',
             'Length': 'num_aas'})

# use the extended_nosc_df to calculate the condition-dependent proteome NOSC
# first need to add a few empty rows for which we don't have NOSC data for some reason.
# TODO: figure out why not? we already know they account for very little expression (above).
missing_ids = set(long_abund_df.majority_protein_ids).difference(
    extended_nosc_df.primary_accession)
empty_rows = pd.DataFrame(dict(primary_accession=list(missing_ids)))
my_nosc_df = pd.concat([extended_nosc_df, empty_rows]).set_index('primary_accession')

print('Adding sequence metadata...')
NCs = my_nosc_df.loc[long_abund_df.majority_protein_ids].NC.values
Ces = my_nosc_df.loc[long_abund_df.majority_protein_ids].Ce.values
NOSCs = my_nosc_df.loc[long_abund_df.majority_protein_ids].NOSC.values
ftm = my_nosc_df.loc[long_abund_df.majority_protein_ids].fraction_transmembrane.values
ftmC = my_nosc_df.loc[long_abund_df.majority_protein_ids].fraction_transmembrane_C.values
long_abund_df['NC_per'] = NCs
long_abund_df['Ce_per'] = Ces
long_abund_df['NC_total'] = long_abund_df.copies_per_cell.multiply(NCs)
long_abund_df['Ce_total'] = long_abund_df.copies_per_cell.multiply(Ces)
long_abund_df['NOSC'] = NOSCs 
long_abund_df['fraction_transmembrane'] = ftm 
long_abund_df['fraction_transmembrane_C'] = ftmC 
long_abund_df['condition'] = (
    'photobio_' + long_abund_df.red_light_intensity_uE_m_s.astype(str).values + '_uE')

# Add annotation of the growth mode -- here photobioreactors in turbidostat mode
long_abund_df['growth_mode'] = 'photobioreactor'
# Add annotation of stress conds -- these are all growth in variable light conds
long_abund_df['stress'] = False

# convert to daltons
long_abund_df.mw_daltons *= 1000
# commas for multiple gene names
long_abund_df.gene_name = long_abund_df.gene_name.str.replace(' ', ',')

# Data re in fg/ul, convert to copies/ul
# TODO: save this unit conversion
data_cols = ['27.5', '55', '110', '220', '440', '1100']
# fg/mol = kg/mol * 1e18 fg/kg
shared_ids_list = list(shared_ids)
mws_fg_mol = raw_abund_df.loc[shared_ids_list]['Mol. weight [kDa]'] * 1e18 
# copies/ul = 6.02e23 copies/mol * fg/ul / (fg/mol)
copies_per_ul = raw_abund_df[data_cols].loc[shared_ids_list].multiply(6.02e23/mws_fg_mol, axis=0)

Ces = my_nosc_df.loc[shared_ids_list].Ce.values
NCs = my_nosc_df.loc[shared_ids_list].NC.values
Ce_total = copies_per_ul.multiply(Ces, axis=0)
NC_total = copies_per_ul.multiply(NCs, axis=0)

cond_NOSC = Ce_total.sum()/NC_total.sum()
conds_df['proteome_NOSC'] = cond_NOSC
conds_df['growth_mode'] = 'photobioreactor'
conds_df['dataset'] = 'zavrel_2019'
conds_df['strain'] = 'PCC6803'
conds_df['condition'] = 'photobioreactor_uE' + conds_df['red_light_intensity_uE_m_s'].astype(str)

conds_df.to_csv('data/proteomes/Synechocystis/Zavrel_proteome_NOSC.csv')

# Saving the full dataset with per-row annotations
tmp = long_abund_df.copy()

tmp['dataset'] = 'zavrel_2019'
tmp['strain'] = 'PCC6803'
tmp['species'] = 'Synechocystis sp.'
tmp['organism_key'] = 'PCC6803'
tmp['growth_rate_hr'] = conds_df.loc[tmp.red_light_intensity_uE_m_s].growth_rate_hr.values

tmp.to_csv('data/proteomes/Synechocystis/Zavrel_protein_measurements.csv')