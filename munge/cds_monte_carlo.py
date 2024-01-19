import numpy as np
import pandas as pd
import seaborn as sns

from sklearn.mixture import GaussianMixture
from matplotlib import pyplot as plt
from scipy.stats import gmean, gstd

"""
# Estimating proteome NOSC using coding sequences alone

Some previous work, mostly from Jeff Dick and co-authors, investigates
the relationships between environmental chemistry (redox) and the $Z_C$
values of protein coding sequences in prokaryotic genomes. Here I ask
how different genomic values are from those inferred from high-quality
proteomics datasets to ascertain if there is additional chemical
information in the expression data. 

In *E. coli*, 1000 proteins account for more than 95% of total protein C.
Therefore, I repeatedly sample 1000 proteins from the coding sequence and ask
what their typical $Z_C$ value is. This gives an unweighted estimate of the
range of $Z_C$ values accomodated by the genome. Since gene expression spans
5-6 orders and is roughly log-normally distributed, I also sample 1000
proteins from a lognormal spanning 5 orders and calculate $Z_C$ values
from those samples. This latter approach asks what range of protein
$Z_{C,P}$ is plausible for expression datasets.

# References 

* J. M. Dick, G. M. Boyer, P. A. Canovas 3rd, E. L. Shock, Using thermodynamics to obtain geochemical information from genomes. Geobiology (2022) https:/doi.org/10.1111/gbi.12532.

* J. M. Dick, Average oxidation state of carbon in proteins. J. R. Soc. Interface 11, 20131095 (2014).
"""

print('Loading coding sequences...')

# Load the reference proteomes with NOSC information
long_nosc_df = pd.read_csv('data/genomes/all_ref_prot_NOSC.csv')
long_nosc_df.head()

# Monte Carlo sampling to estimate plausible range of proteome NOSC
eq_weight_estimates = dict(organism=[], NOSC=[], eC_ratio=[])
weighted_estimates = dict(organism=[], NOSC=[], eC_ratio=[])
# estimate a range on the coding sequence NOSC
cols2sum = 'NC,Ce'.split(',')
rows_mask = long_nosc_df[cols2sum].notnull().all(axis=1)

# see the np random number generator
print('Performing weighted and unweighted sampling...')
np.random.seed(42)

# For each organism, we sample 1000 genes across 5 orders of expression
# 1e5 times and record the NOSC of the samples. We also make an unweighted 
# estimate, but they should have the same mean. 
for gid, gdf in long_nosc_df.loc[rows_mask].groupby('organism'):
    print('Sampling {0} coding sequences'.format(gid))
    for _ in range(10000):
        # In E. coli, 1000 genes covers ≈99% of protein C atoms
        sample = gdf.sample(1000)
        s = sample[cols2sum].sum()
        
        sample_nosc = s.Ce / s.NC

        eq_weight_estimates['organism'].append(gid)
        eq_weight_estimates['NOSC'].append(sample_nosc)
        eq_weight_estimates['eC_ratio'].append(4 - sample_nosc)
        
        # E. coli expression data are approximately log-normal with sigma ≈ 2.3
        # TODO: should we use empirical std deviation from each organism? 
        # It looks like the E. coli data are the highest quality. 
        expression = np.random.lognormal(size=1000, sigma=2.3)
        Ce_weighted = sample.Ce * expression
        NC_weighted = sample.NC * expression
        sample_nosc_weighted = Ce_weighted.sum() / NC_weighted.sum()
        
        weighted_estimates['organism'].append(gid)
        weighted_estimates['NOSC'].append(sample_nosc_weighted)
        weighted_estimates['eC_ratio'].append(4 - sample_nosc_weighted)
        
eq_weight_nosc_est_df = pd.DataFrame(eq_weight_estimates)
eq_weight_nosc_est_df.to_csv('data/genomes/all_ref_prot_NOSC_unweighted_monte_carlo_samples.csv', index=False)
weighted_nosc_est_df = pd.DataFrame(weighted_estimates)
weighted_nosc_est_df.to_csv('data/genomes/all_ref_prot_NOSC_weighted_monte_carlo_samples.csv', index=False)