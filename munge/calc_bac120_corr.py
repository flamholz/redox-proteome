import numpy as np
import pandas as pd
import pingouin as pg
import seaborn as sns

from scipy.stats import linregress, gmean, pearsonr
from sklearn.metrics import r2_score
from os import path

# Read the ZC values for bac120 as a wide-form matrix
print('Reading calculated bac120 NOSC values...')
bac120_nosc_df = pd.read_csv('data/gtdb/r207/bac120_nosc_vals_wide.csv', index_col=0)

# Trim the ".1" from the end of the accession numbers so they match other GTDB data
trimmed_ids = [x.split('.')[0] for x in bac120_nosc_df.index.tolist()]
bac120_nosc_df['trimmed_accession'] = trimmed_ids

# Reset the index to the trimmed accession number
bac120_nosc_df = bac120_nosc_df.reset_index().set_index('trimmed_accession')

# Read the annotated bac120 metadata
bac120_meta_df = pd.read_csv('data/gtdb/r207/bac120_msa_marker_info_r207_annot.csv',
                             index_col=0)

# Read the genome average ZC values for representative genomes
# These are calculated via munge/calc_genome_nosc_batch.py
reps_genome_nosc_df = pd.read_csv('data/gtdb/r207/genome_average_nosc.csv', index_col=0)

# Merge the genome average values into the bac120 nosc dataframe
print('Merging in genome average NOSC values...')
bac120_nosc_df = bac120_nosc_df.merge(reps_genome_nosc_df, left_index=True, right_index=True)

# Load metadata about all the genomes
print('Merging in GTDB metadata...')
gtdb_genome_metadata = pd.read_csv('data/gtdb/r207/bac120_metadata_r207.tar.gz',
                                   compression='gzip', sep='\t')

# Extract the phylogeny from the metadata
gtdb_phylo = gtdb_genome_metadata.gtdb_taxonomy.str.split(';', expand=True)
gtdb_phylo.columns = ['gtdb_domain', 'gtdb_phylum', 'gtdb_class', 'gtdb_order',
                      'gtdb_family', 'gtdb_genus', 'gtdb_species']

# Strip the underscores from the names
gtdb_phylo = gtdb_phylo.applymap(lambda x: x.split('__')[1] if type(x) == str else x)
gtdb_genome_metadata = pd.concat([gtdb_genome_metadata, gtdb_phylo], axis=1)

# Rename the first column to "genome_accession" -- for some reason defaults to the filename
# Also drop the rows with no accession number
gtdb_genome_metadata = gtdb_genome_metadata.rename(
    columns={gtdb_genome_metadata.columns[0]: 'genome_accession'}).dropna(
        axis=0, subset=['genome_accession']).set_index('genome_accession')

# For some reason the accession numbers have ".1" appended to them here 
# but not in the files defining the bac120 marker genes
trimmed_ids = [x.split('.')[0] for x in gtdb_genome_metadata.index.tolist()]
gtdb_genome_metadata['trimmed_accession'] = trimmed_ids
gtdb_genome_metadata['ncbi_taxid_int'] = gtdb_genome_metadata.ncbi_taxid.astype(int)
gtdb_genome_metadata = gtdb_genome_metadata.reset_index().set_index('trimmed_accession')

# Count up the COG categories among the bac120
cat_counts = bac120_meta_df.COG.value_counts()

# Add descriptive annotations by COG category
cog_df = pd.read_csv('data/COG-fun-20.csv', index_col=0)
descs = cog_df.loc[cat_counts.index].description
cat_counts = pd.DataFrame(
    dict(count=cat_counts, description=descs))

cat_counts.to_csv('data/gtdb/r207/bac120_COG_counts.csv')

# Grab the data with which we calculate correlations
pairwise_cols = bac120_nosc_df.columns[1:-1]
mat = bac120_nosc_df[pairwise_cols].values

# Need to flatten the matrix to permute rows and cols
# correlations will not change if we only shuffle one axis
print('Calculating and saving correlations for permuted data...')
permuted = np.random.permutation(mat.flatten()).reshape(mat.shape)
permuted_df = pd.DataFrame(permuted, columns=pairwise_cols)
permuted_corr = pg.pairwise_corr(permuted_df, method='pearson')
permuted_corr.to_csv('data/gtdb/r207/bac120_nosc_permuted_corr.csv')

# last column is genome_avg_NOSC -- 
# calculate the pairwise correlations of columns except GC
print('Calculating and saving correlations for bac120 NOSC values...')
pairwise_cols = bac120_nosc_df.columns[:-1]
nosc_corr = pg.pairwise_corr(
    bac120_nosc_df, columns=pairwise_cols,
    padjust='fdr_bh')

# Now calculate the partial correlations, controlling for genome_avg_NOSC
print('Calculating and saving partial correlations for bac120 NOSC values...')
nosc_corr_controlled = pg.pairwise_corr(
    bac120_nosc_df, columns=pairwise_cols, covar='genome_avg_NOSC',
    padjust='fdr_bh')

f = lambda row: ','.join(sorted(row))
def _save_corr_df(corr_df, fname):
    # Add descriptions of the two columns
    corr_df['X_desc'] = bac120_meta_df.loc[corr_df.X].Description.values
    corr_df['Y_desc'] = bac120_meta_df.loc[corr_df.Y].Description.values

    # Add COG categories of the two cols
    corr_df['X_COG'] = bac120_meta_df.loc[corr_df.X].COG.values
    corr_df['Y_COG'] = bac120_meta_df.loc[corr_df.Y].COG.values

    # Pair of COG categories represented    
    corr_df['COG_pair'] = list(map(f, corr_df['X_COG,Y_COG'.split(',')].values))
    
    # Save
    corr_df.to_csv(fname, index=False)

my_fname = 'data/gtdb/r207/bac120_nosc_corr.csv'
_save_corr_df(nosc_corr, my_fname)

my_fname = 'data/gtdb/r207/bac120_nosc_corr_controlled_for_genome_nosc.csv'
_save_corr_df(nosc_corr_controlled, my_fname)