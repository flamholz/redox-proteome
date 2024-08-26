import pandas as pd
import os

# Metabolic modes of E. coli
print('Making Table S1... growth capabilities')

# Make a dataframe describing the subsequent sheets in this file. 
desc_dict = {
    'Sheet': ['E. coli observations', 
                       'E. coli iML1515 model',
                       'Fastest growth rates'],
    'Description': ['Carbon sources and electron acceptors that E. coli lab strains have been observed to grow on.',
                    'Predicted growth rates (/hr) from the E. coli iML1515 model. Assays growth on various pairs of carbon sources and electron acceptors.',
                    'Fastest observed growth rates for microbial heterotrophs and autotrophs.']
}
desc_df = pd.DataFrame(desc_dict)

# Data
coli_growth_df = pd.read_excel('data/physiology/coli_growth_conds.xlsx')

# FBA
growth_mat_df = pd.read_csv('output/iML1515_growth_rate_matrix.csv')

# Fastest growth rates
max_lambda_df = pd.read_csv('data/physiology/fastest_growers.csv')

# Only output the ones we are plotting, drop the "to_plot" column.
mask = max_lambda_df.to_plot
max_lambda_df = max_lambda_df[mask].copy().drop('to_plot', axis=1)

# Make a single excel file with all the data
writer = pd.ExcelWriter('si_tables/SuppTable1_Growth.xlsx', engine="openpyxl")

desc_df.to_excel(writer, sheet_name='Table descriptions', index=False)
coli_growth_df.to_excel(writer, sheet_name='E. coli observations', index=False)
growth_mat_df.to_excel(writer, sheet_name='E. coli iML1515 model', index=False)
max_lambda_df.to_excel(writer, sheet_name='Fastest growth rates', index=False)

writer.close()

# Amino acid properties
print('Making Table S2... amino acid properties')
aa_nosc_df = pd.read_csv('data/aa_nosc.csv')
aa_nosc_df.to_excel('si_tables/SuppTable2_AA_properties.xlsx', index=False)

# Load the lipid data -- doing this before the slower operations, despite it being table S6.
print('Making Table S6... lipid data')
lipids_by_cond_df = pd.read_csv('data/lipids/Coli/Marr1962_JBac_long.csv')
lipids_by_cond_df.rename(columns={'NOSC': 'Z_C'}, inplace=True)
total_lipids_marr = pd.read_csv('data/lipids/Coli/Marr1962_total_lipids_NOSC.csv')
total_lipids_marr.rename(columns={'lipid_NOSC': 'lipid_Z_C'}, inplace=True)

desc_dict = {
    'Sheet': ['E. coli lipid measurements',
              'E. coli total lipid Z_C'],
    'Description': ['Lipid composition of E. coli at different growth temperatures from Marr & Ingraham, 1962',
                    'Inferred total lipid Z_C values for E. coli under different growth conditions. Growth rates inferred as described in the Methods.']
}
desc_df = pd.DataFrame(desc_dict)

# Make a single excel file with all the data
writer = pd.ExcelWriter('si_tables/SuppTable6_Lipids.xlsx', engine="openpyxl")

desc_df.to_excel(writer, sheet_name='Table descriptions', index=False)
lipids_by_cond_df.to_excel(writer, sheet_name='E. coli lipid measurements', index=False)
total_lipids_marr.to_excel(writer, sheet_name='E. coli total lipid Z_C', index=False)

writer.close()

# Load GTDB bac120 data
print('Making Table S3... GTDB ZC values')

# Make an excel of the raw ZC values
pd.read_csv('output/gtdb/r207/bac120_nosc_vals_wide_compressed.csv'
    ).to_excel('si_tables/SuppTable3_GTDB_bac120_ZC.xlsx',
               sheet_name='bac120_ZC_values', index=False)

# Read the correlation matrix for bac120 -- raw correlations first
print('Making Table S4... GTDB ZC correlations')

desc_dict = {
    'Sheet': ['bac120 ZC correlations', 
              'bac120 partial correlations'],
    'Description': ['Correlations between Z_C values for bac120 genes across GTDB r207 representative genomes.',
                    'Partial correlations between Z_C values for bac120 genes, controlling for mean genome Z_C.']
}
desc_df = pd.DataFrame(desc_dict)

nosc_corr_df = pd.read_csv('output/gtdb/r207/bac120_nosc_corr.csv')
col_renames = {'p-unc': 'uncorrected p-value',
               'p-corr': 'corrected p-value',
               'p-adjust': 'p-value correction'}
nosc_corr_df.rename(columns=col_renames, inplace=True)

nosc_corr_controlled_df = pd.read_csv('output/gtdb/r207/bac120_nosc_corr_controlled_for_genome_nosc.csv')
nosc_corr_controlled_df.rename(columns=col_renames, inplace=True)

# Make a single excel file with all the data
writer = pd.ExcelWriter('si_tables/SuppTable4_GTDB_correlations.xlsx', engine="openpyxl")

desc_df.to_excel(writer, sheet_name='Table descriptions', index=False)
nosc_corr_df.to_excel(writer, sheet_name='bac120 ZC correlations', index=False)
nosc_corr_controlled_df.to_excel(writer, sheet_name='bac120 partial correlations', index=False)

writer.close()

# Load the growth conditions and proteome ZC
# Omitting the raw proteomic and genomic tables for now due to size limits.
print('Making Table S2... proteome Z_C')
desc_dict = {
    'Sheet': ['E. coli conditions', 
              'S. cerevisiae conditions', 
              'Syn. PCC 6803 conditions',
              #'Coding sequence Z_C', 
              #'All protein measurements'
              ],
    'Description': [
        'Expression-weighted proteome Z_C values for E. coli K-12 MG1655 under different growth conditions.',
        'Expression-weighted proteome Z_C values for S. cerevisiae S288C under different growth conditions.',
        'Expression-weighted proteome Z_C values for Syn. PCC 6803 under different growth conditions.',
        #'Z_C values for all coding sequences in all genomes considered.',
        #'Protein-level measurements for all organisms and conditions.'
        ]
}
desc_df = pd.DataFrame(desc_dict)

coding_seq_zc = pd.read_csv('data/genomes/all_ref_prot_NOSC.csv')
coding_seq_zc.rename(columns={'NOSC': 'Z_C'}, inplace=True)
coding_seq_zc.drop(['Ce', 'eC_ratio'], axis=1, inplace=True, errors='ignore')
coding_seq_zc.Z_C = coding_seq_zc.Z_C.round(6)
coding_seq_zc.fraction_transmembrane = coding_seq_zc.fraction_transmembrane.round(6)
coding_seq_zc.fraction_transmembrane_C = coding_seq_zc.fraction_transmembrane_C.round(6)

# Single files per-organism of proteome level Z_C at different growth rates.
coli_data = pd.read_csv('data/proteomes/Coli/Chure_proteome_NOSC.csv')
yeast_data = pd.read_csv('data/proteomes/Scer/Xia_proteome_NOSC.csv')
cyano_data = pd.read_csv('data/proteomes/Synechocystis/Zavrel_proteome_NOSC.csv')

# Reorder columns from above dfs to have the same order in the unified file. 
for df in [coli_data, yeast_data, cyano_data]:
    df.rename(columns={'proteome_NOSC': 'proteome_Z_C'}, inplace=True)
    df.drop(['Ce_total', 'NC_total'], axis=1, inplace=True, errors='ignore')
    # Round Z_C values to 6 decimal places
    df.proteome_Z_C = df.proteome_Z_C.round(6)

# Make column order consistent across all dfs
yeast_col_order = ['dataset', 'strain', 'condition', 'growth_mode', 'growth_rate_hr', 'proteome_Z_C']
yeast_data = yeast_data[yeast_col_order].copy()

cyano_col_order = ['dataset', 'strain', 'condition', 'growth_mode', 'red_light_intensity_uE_m_s', 'growth_rate_hr', 'proteome_Z_C']
cyano_data = cyano_data[cyano_col_order].copy()
# fill missing values with "not measured"
cyano_data['proteome_Z_C'].fillna('not measured', inplace=True)

# Unified file of the protein-level measurements.
all_expression_data = pd.read_csv('data/proteomes/all_protein_measurements.csv').drop('Unnamed: 0', axis=1)
all_expression_data.rename(columns={'NOSC': 'Z_C', 'majority_protein_ids': 'protein_id'}, inplace=True)
cols2use = ['protein_id', 'b_number', 'dataset', 'organism_key', 'species', 'strain',
            'condition', 'growth_rate_hr', 'growth_mode', 'stress', 'NC_per', 'Z_C',
            'copies_per_cell', 'fg_per_cell'] 
all_expression_data = all_expression_data[cols2use].copy()
all_expression_data.Z_C = all_expression_data.Z_C.round(6)
all_expression_data.copies_per_cell = all_expression_data.copies_per_cell.round(6)
all_expression_data.fg_per_cell = all_expression_data.fg_per_cell.round(6)
all_expression_data.growth_rate_hr = all_expression_data.growth_rate_hr.round(4)

# Make a single excel file with all the data
writer = pd.ExcelWriter('si_tables/SuppTable5_Proteomes.xlsx', engine="openpyxl")

desc_df.to_excel(writer, sheet_name='Table descriptions', index=False)
coli_data.to_excel(writer, sheet_name='E. coli conditions', index=False)
yeast_data.to_excel(writer, sheet_name='S. cerevisiae conditions', index=False)
cyano_data.to_excel(writer, sheet_name='Syn. PCC 6803 conditions', index=False)
#coding_seq_zc.to_excel(writer, sheet_name='Coding sequence Z_C', index=False)
#all_expression_data.to_excel(writer, sheet_name='All protein measurements', index=False)

writer.close()