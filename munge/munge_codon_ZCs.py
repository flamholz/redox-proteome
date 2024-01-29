import numpy as np
import pandas as pd
import seaborn as sns

from Bio.Data import CodonTable
from matplotlib import pyplot as plt
from scipy.stats import linregress, gmean
from skbio.sequence.distance import hamming
from skbio.sequence import Sequence

"""
# Analysis of Z_C conservation in the genetic code.

1. Hydrophobicity and Z_C correlate. Code is conservative for hydrophobicity, e.g. Haig & Hurst 1991. 

* This data is in the aa_nosc.csv file now. Just plotting correlations.

2. Calculate Z_C consequences of transitions in the standard code. Need multiple subs to change Z_C. 

* We tabulate all the transitions and their Z_C consequences below. 

3. Use an empirical contingency table to estimate real-ish rates. 

* Above analysis ignores relative rates of transitions. 
"""

# Amino acid properties - ZC and hydrophobicity etc. 
aa_nosc_df = pd.read_csv('data/aa_nosc.csv')

# Grab the standard codon table
cdn_table = CodonTable.unambiguous_dna_by_name["Standard"]

# DataFrame mapping codons => amino acids, Z_C
print('Calculating Z_C effects of substitutions...')
fwd_dict = cdn_table.forward_table
codons = sorted(fwd_dict.keys())
aas = [fwd_dict[k] for k in codons]

cdn_df = pd.DataFrame(dict(aa=aas, codon=codons))
cdn_df['NOSC'] = aa_nosc_df.set_index('letter_code').loc[cdn_df.aa.values].NOSC.values

# Consider all possible codon transitions.
subs_dict = dict(from_codon=[], to_codon=[], from_NOSC=[], to_NOSC=[], from_aa=[], to_aa=[], hamming_dist=[])
for idx, row1 in cdn_df.iterrows():
    for jdx, row2 in cdn_df.iterrows():
        if idx >= jdx:
            # Only consider half the symmetric matrix
            continue 
        
        subs_dict['from_codon'].append(row1.codon)
        subs_dict['to_codon'].append(row2.codon)
        subs_dict['from_aa'].append(row1.aa)
        subs_dict['to_aa'].append(row2.aa)
        subs_dict['from_NOSC'].append(row1.NOSC)
        subs_dict['to_NOSC'].append(row2.NOSC)
        
        # skbio hamming is fractional change.
        # hence hamming*length = number of mutations between seqs
        hd = hamming(Sequence(row1.codon), Sequence(row2.codon))*3
        subs_dict['hamming_dist'].append(hd)
        
# Convert to DF
subs_df = pd.DataFrame(subs_dict)
# positive dNOSC = more reduced sub
subs_df['dNOSC'] = subs_df.to_NOSC - subs_df.from_NOSC
# Since codon subs can go in either direction, abs dNOSC is the value of interest
subs_df['abs_dNOSC'] = subs_df.dNOSC.abs()

print('Saving codon substitution data as CSV...')
subs_df.to_csv('data/genetic_code/all_codon_substitutions.csv', index=False)

# Bin them by absolute delta NOSC and take the mean. 
print('Binning codon substitution data...')
bins = pd.cut(subs_df.abs_dNOSC, 10) # 10 bins
bins.name = 'abs_dNOSC_bin'
codon_subs_mean_df = subs_df.groupby(bins).mean(numeric_only=True)
codon_subs_std_df = subs_df.groupby(bins).std(numeric_only=True)

# Calculate the bin midpoint for plotting below
codon_subs_mean_df['midpoint_abs_dNOSC'] = [np.mean([a.left, a.right]) for a in codon_subs_mean_df.index]
codon_subs_mean_df['abs_dNOSC_std'] = codon_subs_std_df['abs_dNOSC']
codon_subs_mean_df['hamming_dist_std'] = codon_subs_std_df['hamming_dist']

print('Saving binned codon substitution data as CSV...')
codon_subs_mean_df.to_csv('data/genetic_code/binned_codon_substitutions.csv')

