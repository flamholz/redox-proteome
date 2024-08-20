import numpy as np
import pandas as pd
import seaborn as sns

from matplotlib import pyplot as plt
from scipy.stats import pearsonr
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import SparsePCA, PCA
from sklearn import linear_model
from tqdm import tqdm


"""
This script attempts to reconstruct proteomic $Z_{C}-\lambda$ co-variation
with sparse regression. 

As a reminder, the nominal oxidation state of C atoms in all proteins (the
"proteome") is a C-weighted average of the $Z_C$ values of individual proteins: 

$Z_{C,P} = \frac{ \sum_i  Z_{C,i} \cdot N_{C,i} \cdot \eta_i }{\sum_i N_{C,i} \cdot \eta_i }$

where $\eta_i$ is the relative level of protein $i$. 

By defining little 

$z_j = \frac{Z_{C,j} \cdot N_{C,j} \cdot \eta_j }{\sum_i N_{C,i} \cdot \eta_i }$

we notice that $Z_{C,P} = \sum_j z_j$. That is, in any condition $Z_{C,P}$ is
a linear function of the $z_j$ values. Therefore, if we regress
condition-dependent $Z_C,P$ values against $z_j$ we should be able to
perfectly reconstruct $Z_C,P$ changes. 

Moreover, since the expression levels $\eta_j$ are correlated with each other, e.g.,
ribosomal proteins are expressed at similar levels &mdash; we won't need all the
little $z_j$ values. Here we use sparse regression (the lasso) to ask how many
"basis proteins" are needed to predict 100% of variation in $Z_{C,P}$ as a means
of understanding the intrinsic dimensionality of the trend. We then ask whether
these basis proteins represent a wide or narrow diversity of biological
functions using the COG taxonomy. 
"""

# Helper functions
def _regress_and_count_nz_cogs(A_mat, b_resp, alpha, cds_df, protein_id='b_number'):
    """Perform sparse regression and count the COG categories of the proteins with nonzero coefficients.

    Args:
        A_mat (pd.DataFrame): Matrix of predictors.
        b_resp (pd.Series): Vector of responses.
        alpha (float): Regularization parameter.
        cds_df (pd.DataFrame): Dataframe with COG categories for each protein.
        protein_id (str): Name of the column in cds_df with the protein IDs.

    Returns:
        tuple: Tuple of (R^2, list of proteins with nonzero coefficients,
            pandas Series of COG categories and their counts).
    """
    reg = linear_model.Lasso(alpha=alpha, max_iter=100000, fit_intercept=False, positive=True)
    reg.fit(A_mat, b_resp)

    r2 = reg.score(A_mat, b_resp)
    
    nz_prots = A_mat.columns[np.where(reg.coef_ != 0)].values
    nz_prot_meta = cds_df.set_index(protein_id).loc[nz_prots]

    cats = nz_prot_meta.primary_COG_category.value_counts()
    return r2, nz_prots, cats
    
def _make_mat_Ce_normed(long_df, protein_id='b_number',
                        index_cols=('dataset', 'condition', 'growth_rate_hr')):
    """Convert long-form expression data to matrices of Z_C,P contributions z_j.

    See notes at the top for definition of z_j. 

    Args:
        long_df (pd.DataFrame): Long-form expression data with columns
            'b_number,dataset,condition,growth_rate_hr,Ce_total,NC_total'.
    
    Returns:
        mat_Ce_normd (pd.DataFrame): Matrix of z_j values for each protein j. 
            Rows are proteins, columns are conditions.
    """
    # Mask out entries with missing values
    mask = np.logical_or(long_df.Ce_total.notnull(), long_df.NC_total.notnull())
    
    # Calculate each proteins' z_j value in each condition. 
    # Note: Ce_total = N_C*copies_per_cell*Z_C
    # and NC_total = N_C*copies_per_cell
    mat_Ce_tot = long_df[mask].pivot_table(index=protein_id, 
                                           values='Ce_total', columns=index_cols)
    mat_NC_tot = long_df[mask].pivot_table(index=protein_id, values='NC_total', columns=index_cols)

    # Normalize by total protein C content
    mat_Ce_normd = mat_Ce_tot / mat_NC_tot.sum()
    return mat_Ce_normd.replace({np.NaN: 0})


def _do_sparse_regression(A_mat, b_resp, alpha, max_iter=100000):
    """Do sparse regression with Lasso regularization.

    Args:
        A_mat (np.ndarray): Matrix of predictors.
        b_resp (np.ndarray): Vector of responses.
        alpha (float): Regularization parameter.
        max_iter (int): Maximum number of iterations for Lasso regression.

    Returns:
        dict: Dictionary of results with keys 'number_nonzero', 'r2', and 'alpha'.
    """
    # Requiring positive contributions since proteins contributions to Z_C,P are strictly positive.
    # Not fitting an intercept since the chemical relationship has 0 intercept.
    reg = linear_model.Lasso(alpha=alpha, max_iter=max_iter,
                             fit_intercept=False, positive=True)
    reg.fit(A_mat, b_resp)

    # Get number of nonzero coefficients and R^2
    nz_coeff = reg.coef_[reg.coef_ != 0]
    r2 = reg.score(A_mat, b_resp)
    return dict(number_nonzero=nz_coeff.size, r2=r2, alpha=alpha)

print('Loading data...')

# Load all the expression data
all_exp_df = pd.read_csv('data/proteomes/all_protein_measurements.csv')

# All the yeast data
mask = all_exp_df.organism_key == 'yeast'
yeast_exp_df = all_exp_df[mask]

# All the cyanobacterial data
mask = all_exp_df.organism_key == 'PCC6803'
cyano_exp_df = all_exp_df[mask]

# All the E. coli data
mask = all_exp_df.organism_key == 'coli'
coli_exp_df = all_exp_df[mask]

# Focus on Schmidt dataset for E. coli analysis. 
mask = all_exp_df.dataset == 'schmidt_2016'
schmidt_df = all_exp_df[mask]

# Separate chemostat and batch (nonstress) data
mask = np.logical_and(schmidt_df.growth_mode == 'chemostat', schmidt_df.stress == False)
schmidt_chemo_df = schmidt_df[mask]

mask = np.logical_and(schmidt_df.growth_mode == 'batch', schmidt_df.stress == False)
schmidt_batch_df = schmidt_df[mask]

# grab all the non-stress conditions
mask = schmidt_df.stress == False
schmidt_nonstress_df = schmidt_df[mask]

# Read in condition-wise Z_C,P values
coli_NOSC_data = pd.read_csv('data/proteomes/Coli/Chure_proteome_NOSC.csv')
yeast_NOSC_data = pd.read_csv('data/proteomes/Scer/Xia_proteome_NOSC.csv')
cyano_NOSC_data = pd.read_csv('data/proteomes/Synechocystis/Zavrel_proteome_NOSC.csv')

# Read in coli genome information
all_CDS_df = pd.read_csv('data/genomes/all_ref_prot_NOSC.csv')
coli_CDS_df = all_CDS_df[all_CDS_df.organism == 'coli']
yeast_CDS_df = all_CDS_df[all_CDS_df.organism == 'yeast']
cyano_CDS_df = all_CDS_df[all_CDS_df.organism == 'PCC6803']

print('Data loaded. Now performing sparse regression...')
print('E. coli analysis...')
# Consider fitting just the chemostat conds, just the batch conds, and both together.
mat_schmidt_chemo = _make_mat_Ce_normed(schmidt_chemo_df)
mat_schmidt_batch = _make_mat_Ce_normed(schmidt_batch_df)
mat_schmidt_nonstress = _make_mat_Ce_normed(schmidt_nonstress_df)
mat_coli_all = _make_mat_Ce_normed(coli_exp_df)

# Perform and plot predictive accuracy of sparse regression for different levels of sparsity.
# Response variable is the whole proteome Z_C,P for the appropriate conditions
coli_NOSC_reindexed = coli_NOSC_data.set_index('dataset,condition,growth_rate_hr'.split(','))
b_chemo = coli_NOSC_reindexed.loc[mat_schmidt_chemo.columns].proteome_NOSC
b_batch = coli_NOSC_reindexed.loc[mat_schmidt_batch.columns].proteome_NOSC
b_nonstress = coli_NOSC_reindexed.loc[mat_schmidt_nonstress.columns].proteome_NOSC
b_coli_all = coli_NOSC_reindexed.loc[mat_coli_all.columns].proteome_NOSC

# Regressed against individual contributions of proteins in the matrices above. 
A_chemo = mat_schmidt_chemo.T
A_batch = mat_schmidt_batch.T
A_nonstress = mat_schmidt_nonstress.T
A_coli_all = mat_coli_all.T
coli_A_mats = [(A_chemo, b_chemo, 'chemostat'),
               (A_batch, b_batch, 'batch'),
               (A_nonstress, b_nonstress, 'both'),
               (A_coli_all, b_coli_all, 'all')]

# Test different regularization parameters. Larger alpha => sparser models.
n_alphas = 50
# Add 1e-8 to the end of the list since it was selected for the final analysis.
alphas = list(np.logspace(-6, -9, n_alphas)) + [1e-8]
res_ds = []

for alpha in tqdm(alphas):
    for A_mat, b_resp, cond in coli_A_mats:
        d = _do_sparse_regression(A_mat, b_resp, alpha)
        d['conditions_included'] = cond
        d['organism_key'] = 'coli'
        res_ds.append(d)

# Make and save a dataframe with the summary of this analysis.
coli_reg_df = pd.DataFrame(res_ds)
coli_reg_df.to_csv('output/sparse_reg/Schmidt_lasso_regression.csv', index=False)


print('Yeast analysis...')
mat_yeast_all = _make_mat_Ce_normed(yeast_exp_df, protein_id='majority_protein_ids')

# Perform and plot predictive accuracy of sparse regression for different levels of sparsity.
# Response variable is the whole proteome Z_C,P for the appropriate conditions
yeast_NOSC_reindexed = yeast_NOSC_data.set_index('growth_rate_hr')
b_yeast_all = yeast_NOSC_reindexed.loc[mat_yeast_all.columns.levels[2]].proteome_NOSC

# Regressed against individual contributions of proteins in the matrices above. 
A_yeast_all = mat_yeast_all.T

# Test different regularization parameters. Larger alpha => sparser models.
res_ds = []
for alpha in tqdm(alphas):
    d = _do_sparse_regression(A_yeast_all, b_yeast_all, alpha)
    d['conditions_included'] = 'chemostat'
    d['organism_key'] = 'yeast'
    res_ds.append(d)

# Make and save a dataframe with the summary of this analysis.
yeast_reg_df = pd.DataFrame(res_ds)
yeast_reg_df.to_csv('output/sparse_reg/Xia_lasso_regression.csv', index=False)

print('Cyanobacterial analysis...')
mat_cyano_all = _make_mat_Ce_normed(cyano_exp_df, protein_id='majority_protein_ids')

# Perform and plot predictive accuracy of sparse regression for different levels of sparsity.
# Response variable is the whole proteome Z_C,P for the appropriate conditions
cyano_NOSC_reindexed = cyano_NOSC_data.set_index('growth_rate_hr')
b_cyano_all = cyano_NOSC_reindexed.loc[mat_cyano_all.columns.levels[2]].proteome_NOSC
b_cyano_all = b_cyano_all[b_cyano_all.notnull()].copy()

# Regressed against individual contributions of proteins in the matrices above. 
A_cyano_all = mat_cyano_all.T

# Test different regularization parameters. Larger alpha => sparser models.
res_ds = []
for alpha in tqdm(alphas):
    d = _do_sparse_regression(A_cyano_all, b_cyano_all, alpha)
    d['conditions_included'] = 'chemostat'
    d['organism_key'] = 'PCC6803'
    res_ds.append(d)

# Make and save a dataframe with the summary of this analysis.
cyano_reg_df = pd.DataFrame(res_ds)
cyano_reg_df.to_csv('output/sparse_reg/Zavrel_lasso_regression.csv', index=False)

print('Saving a single concatenated dataframe...')
# Merge into a single dataframe for plots
long_reg_df = pd.concat([coli_reg_df, yeast_reg_df, cyano_reg_df])
long_reg_df.to_csv('output/sparse_reg/all_lasso_regression.csv', index=False)

print('Analysis of proteins selected by sparse regression at alpha = 1e-8...')
# This is done on the basis of their primary COG categories, which are high-level functional description
COG_cats = pd.read_csv('data/COG-fun-20.csv').set_index('category')

alpha = 1e-8
print('alpha = {0}'.format(alpha))

# Append organism name to each tuple. 
# Omitting yeast from the COG analysis because the COG DB is for prokaryotes only.
tmp_coli_mats = [(*c, 'coli') for c in coli_A_mats]
all_mats = tmp_coli_mats + [(A_cyano_all, b_cyano_all, 'chemostat', 'PCC6803')]

cat_counts_template = pd.Series(dict((i, 0) for i in COG_cats.index))
cols = []
index_names = []

cds_dfs = dict(coli=coli_CDS_df, yeast=yeast_CDS_df, PCC6803=cyano_CDS_df)
protein_ids = dict(coli='b_number', yeast='primary_accession', PCC6803='primary_accession')

# Loop over all the organisms and conditions,
# perform sparse regression, and count the number
# of proteins in each COG category.
for A_mat, b_resp, cond, organism in all_mats: 
    print(organism, cond)
    p_id = protein_ids[organism]
    cds_df = cds_dfs[organism]
    r2, nz_prots, cats = _regress_and_count_nz_cogs(A_mat, b_resp, alpha, cds_df, protein_id=p_id)
    print('{0} {1}: r2 = {2:.2f}'.format(organism, cond, r2))
    
    cat_counts_column = cat_counts_template.copy()
    cat_counts_column[cats.index] = cats
    cols.append(cat_counts_column)
    index_names.append('{0}_{1}'.format(organism, cond))

print('Saving COG counts...')
cat_counts_df = pd.DataFrame(cols, index=index_names)
cat_counts_df.to_csv('output/sparse_reg/lasso_regression_COG_counts_alpha={0:.2g}.csv'.format(alpha))

print('Saving long-form COG counts...')
long_cat_counts_df = cat_counts_df.reset_index().melt(
    id_vars='index', var_name='COG_category', value_name='number_proteins').rename({'index': 'organism_condition'}, axis=1)
long_cat_counts_df.to_csv('output/sparse_reg/long_lasso_regression_COG_counts_alpha={0:.2g}.csv'.format(alpha), index=False)