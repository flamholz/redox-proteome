import cobra
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm

from matplotlib import pyplot as plt

# This script will use the iML1515 model to determine the maximum
# growth rate of E. coli under different conditions.
# Elad Noor recommended iML1515 as a good model to use for this purpose.

# Load the model -- assumes we run from project root.
coli_model = cobra.io.read_sbml_model("models/iML1515.xml")

# Gurobi is default, appears to be slower? 
coli_model.solver = 'glpk'

# Try running a simple FBA to make sure it all works 
opt = coli_model.slim_optimize()
print('opt growth rate:', opt)


# We will iterate over pairs of (C source, electron acceptor) to
# determine maximum model growth rates for iML1515.

# Set up a medium without C or electron acceptors
my_medium = dict(coli_model.medium)
my_medium['EX_glc__D_e'] = 0 
my_medium['EX_o2_e'] = 0

# List of electron acceptors to test
e_acceptors = {
    'NO3': 'EX_no3_e',
    'O2': 'EX_o2_e',
    #'fumarate': 'EX_fum_e',
    'TMAO': 'EX_tmao_e',
    'DMSO': 'EX_dmso_e',
    'fermentation': None,
}


# Get the C exchanges from the model
c_exchanges = {}
for e in coli_model.exchanges:
    for m in e.metabolites.keys():
        if m.elements.get('C', 0) > 0:
            c_exchanges[e.id] = m.name


# Manual list of C sources to try with some categorization
# TODO: should put this in an external file.
c_exchanges2test = [
    dict(name='glucose', type='sugar', ex_id='EX_glc__D_e'),
    dict(name='acetate', type='organic acid', ex_id='EX_ac_e'),
    dict(name='glycerol', type='sugar alcohol', ex_id='EX_glyc_e'),
    dict(name='succinate', type='organic acid', ex_id='EX_succ_e'),
    dict(name='fumarate', type='organic acid', ex_id='EX_fum_e'),
    dict(name='pyruvate', type='organic acid', ex_id='EX_pyr_e'),
    dict(name='lactate', type='organic acid', ex_id='EX_lac__D_e'),
    dict(name='ethanol', type='alcohol', ex_id='EX_etoh_e'),
    dict(name='alanine', type='amino acid', ex_id='EX_ala__L_e'),
    dict(name='glycine', type='amino acid', ex_id='EX_gly_e'),
    dict(name='serine', type='amino acid', ex_id='EX_ser__L_e'),
    dict(name='threonine', type='amino acid', ex_id='EX_thr__L_e'),
    dict(name='valine', type='amino acid', ex_id='EX_val__L_e'),
    dict(name='leucine', type='amino acid', ex_id='EX_leu__L_e'),
    dict(name='isoleucine', type='amino acid', ex_id='EX_ile__L_e'),
    dict(name='aspartate', type='amino acid', ex_id='EX_asp__L_e'),
    dict(name='glutamate', type='amino acid', ex_id='EX_glu__L_e'),
    dict(name='asparagine', type='amino acid', ex_id='EX_asn__L_e'),
    dict(name='glutamine', type='amino acid', ex_id='EX_gln__L_e'),
    dict(name='cysteine', type='amino acid', ex_id='EX_cys__L_e'),
    dict(name='methionine', type='amino acid', ex_id='EX_met__L_e'),
    dict(name='lysine', type='amino acid', ex_id='EX_lys__L_e'),
    dict(name='arginine', type='amino acid', ex_id='EX_arg__L_e'),
    dict(name='histidine', type='amino acid', ex_id='EX_his__L_e'),
    dict(name='phenylalanine', type='amino acid', ex_id='EX_phe__L_e'),
    dict(name='tyrosine', type='amino acid', ex_id='EX_tyr__L_e'),
    dict(name='tryptophan', type='amino acid', ex_id='EX_trp__L_e'),
    dict(name='cytosine', type='nucleobase', ex_id='EX_cytd_e'),
    dict(name='uracil', type='nucleobase', ex_id='EX_ura_e'),
    dict(name='adenine', type='nucleobase', ex_id='EX_adn_e'),
    dict(name='guanine', type='nucleobase', ex_id='EX_gua_e'),
    dict(name='thymine', type='nucleobase', ex_id='EX_thym_e'),
    dict(name='uridine', type='nucleoside', ex_id='EX_uri_e'),
    dict(name='cytidine', type='nucleoside', ex_id='EX_cytd_e'),
    dict(name='adenosine', type='nucleoside', ex_id='EX_adn_e'),
    dict(name='guanosine', type='nucleoside', ex_id='EX_gua_e'),
    dict(name='thymidine', type='nucleoside', ex_id='EX_thym_e'),
    dict(name='inosine', type='nucleoside', ex_id='EX_inost_e'),
    dict(name='xanthosine', type='nucleoside', ex_id='EX_xan_e'),
    dict(name='sucrose', type='sugar', ex_id='EX_sucr_e'),
    dict(name='trehalose', type='sugar', ex_id='EX_tre_e'),
    dict(name='maltose', type='sugar', ex_id='EX_malt_e'),
    dict(name='gluconate', type='sugar acid', ex_id='EX_gal_e'),
    dict(name='glucuronate', type='sugar acid', ex_id='EX_glcur_e'),
    dict(name='galacturonate', type='sugar acid', ex_id='EX_galur_e'),
    dict(name='glucarate', type='sugar acid', ex_id='EX_glcr_e'),
    dict(name='rhamnose', type='sugar', ex_id='EX_rmn_e'),
    dict(name='arabinose', type='sugar', ex_id='EX_arab__L_e'),
    dict(name='xylose', type='sugar', ex_id='EX_xyl__D_e'),
    dict(name='fucose', type='sugar', ex_id='EX_fuc__L_e'),
    dict(name='mannose', type='sugar', ex_id='EX_man_e'),
    dict(name='fructose', type='sugar', ex_id='EX_fru_e'),
    dict(name='sorbitol', type='sugar alcohol', ex_id='EX_sbt__D_e'),
    dict(name='hexadecenoate', type='fatty acid', ex_id='EX_hdcea_e'),
    dict(name='tetradecenoate', type='fatty acid', ex_id='EX_ttdcea_e'),
    dict(name='tetradecanoate', type='fatty acid', ex_id='EX_ttdca_e'),
    dict(name='octadecanoate', type='fatty acid', ex_id='EX_ocdca_e'),
    dict(name='octadecenoate', type='fatty acid', ex_id='EX_ocdcea_e'),
    dict(name='hexanoate', type='fatty acid', ex_id='EX_hxa_e'),
    dict(name='decanoate', type='fatty acid', ex_id='EX_dca_e'),
    dict(name='octanoate', type='fatty acid', ex_id='EX_octa_e'),
    dict(name='hexadecanoate', type='fatty acid', ex_id='EX_hdca_e'),
    dict(name='dodecanoate', type='fatty acid', ex_id='EX_ddca_e'),
    dict(name='3-hydroxypropanoate', type='organic acid', ex_id='EX_3hpp_e'),
    dict(name='phenylacealdehyde', type='aromatic', ex_id='EX_pacald_e'),
    dict(name='phenylpropanoate', type='aromatic', ex_id='EX_pppn_e'),
    dict(name='dopamine', type='aromatic', ex_id='EX_dopa_e'),
    dict(name='tyramine', type='aromatic', ex_id='EX_tym_e'),
]


# Check that all C sources are in the model
for c in c_exchanges2test:
    if c['ex_id'] not in c_exchanges:
        print(c['name'], 'not found')

# Run FBA for all combinations of C and e- acceptors
growth_data = []

print('Calculating growth rates for pairs of C sources and e- acceptors')
for i, (e, e_ex) in tqdm(enumerate(e_acceptors.items()), desc='e- acceptor', position=0):
    for j, c_ex in tqdm(enumerate(c_exchanges2test), desc='C source', position=1, leave=False):
        tmp_medium = dict(my_medium)
        tmp_medium[c_ex['ex_id']] = 10.0

        # None signals fermentation (no e- acceptor)
        if e_ex is not None:
            tmp_medium[e_ex] = 1000.0

        tmp_model = coli_model.copy()
        tmp_model.medium = tmp_medium

        # Calculate the model's growth rate and save
        opt = tmp_model.optimize()
        mu = opt.objective_value

        mu = tmp_model.slim_optimize()
        growth_data.append(dict(
            e_acceptor=e,
            c_source=c_ex['name'],
            c_source_type=c_ex['type'],
            growth_rate_hr=mu,
            maintenance=opt.fluxes['ATPM'],
        ))

# Convert to a dataframe
long_growth_df = pd.DataFrame(growth_data)

# Binarize growth 
long_growth_df['grows'] = long_growth_df.growth_rate_hr > 0.01
growth_mat_df = long_growth_df.pivot_table(
    index='c_source_type,c_source'.split(','), 
    columns='e_acceptor', values='growth_rate_hr').reset_index()
growth_mat_df = growth_mat_df.fillna(0) # NA means infeasible growth

# Drop C sources with no growth at all
no_grow = (growth_mat_df == 0).all(axis=1)
growth_mat_df = growth_mat_df.drop(no_grow.index[no_grow]).sort_values('c_source_type').set_index(
    'c_source_type,c_source'.split(','))

# Save the growth matrix
growth_mat_df.to_csv('../output/iML1515_growth_rate_matrix.csv')

# Make a binary version of the growth matrix for plotting
binary_growth_mat_df = growth_mat_df > 0.001
binary_growth_mat_df.to_csv('../output/iML1515_binary_growth_matrix.csv')

# Maintenance never exceeds the minimum
# -- see that 6.86 is the only value
print('Maintenance ATP requirement for growth:')
print(long_growth_df.maintenance.value_counts())

# The maintenance ATP requirement is given in 
# coli_model.reactions.ATPM
# notice that its LB and UB are not equal
print('Model maintenance ATP requirement:')
print(coli_model.reactions.ATPM)
print('LB:', coli_model.reactions.ATPM.lower_bound)
print('UB:', coli_model.reactions.ATPM.upper_bound)