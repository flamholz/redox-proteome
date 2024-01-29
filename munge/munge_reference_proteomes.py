import numpy as np
import pandas as pd 
import uniprot_util
import sys

from Bio.SeqUtils import molecular_weight
from Bio import SeqIO
from collections import Counter
from os import path
from tqdm import tqdm

from seq_util import calc_protein_nosc_no_raise

# Make it possible to import modules from this directory
from pathlib import Path
from os import path
dir_path = Path(__file__).parent
parent_dir = path.dirname(path.abspath(dir_path))
sys.path.append(str(parent_dir))

"""
# Preprocessing reference proteomes (coding sequences)
Reference proteomes from UniProt give the amino acid coding sequences of protein coding genes in many genomes. Here I am parsing the XML reference proteomes used in various mass spec proteomics studies (E. coli, yeast, cyanobacteria, etc.) to gather sequence information (sequence, carbon count, nominal oxidation state of carbon) and metadata (IDs, functional tags). 

See uniprot_util.py for parsing code. 

Note that we are parsing the E. coli b numbers (gene identifiers) from the reference proteome XML since the E. coli data from the Bellevue & Chure paper uses these identifiers. 
"""

# List of the reference proteomes to parse
ref_prot_metadata = pd.read_csv('data/genomes/reference_proteomes.csv', index_col=0)

# Load amino acid NOSC table -- used to calculate protein NOSC values
aa_nosc_df = pd.read_csv('data/aa_nosc.csv').set_index('letter_code')

print('Parsing reference proteomes...')
cds_dfs = {}
isoform_fname = 'data/genomes/uniprot_sprot_varsplic.fasta'

# Iterate over the reference proteomes and extract coding sequences and metadata
for idx, row in tqdm(ref_prot_metadata.iterrows()):
    print(idx)
    fpath = row.ref_proteome_fname
    # Only gets the dominant isoform sequences & metadata
    my_df = uniprot_util.uniprot_xml2df(fpath, extract_b_number=row.extract_b_num)
    
    # Adds entries and sequences for the secondary isoforms
    my_df = uniprot_util.add_isoforms2df(my_df, isoform_fname)
    
    # Load KEGG pathway mappings for genes
    kegg_mapping_fname = row.KEGG_pathway_mapping_fname
    kegg_pathways_fname = row.KEGG_pathways_fname
    uniprot_util.add_KEGGpathways2df(my_df, kegg_mapping_fname, kegg_pathways_fname)
    
    # Add NOSC + species information.
    # Important to add NOSC information after adding the isoforms so that they are calc'd 
    # from the secondary isoform sequences and not the dominant form.
    new_cols = my_df.aa_seq.apply(calc_protein_nosc_no_raise, args=(aa_nosc_df,)).apply(pd.Series)
    new_cols.columns = 'Ce,NC'.split(',')
    my_df = my_df.merge(new_cols, left_index=True, right_index=True)
    my_df['NOSC'] = my_df.Ce / my_df.NC
    my_df['eC_ratio'] = 4 - my_df.NOSC
    my_df['organism'] = idx
    
    # Make sure we have unique b-numbers for E. coli
    if row.extract_b_num:
        count_bs = pd.Series(Counter(my_df.b_number.values))
        mask = count_bs > 1
        if mask.any():
            print('Found duplicate b-numbers:')
            print(count_bs[mask])
            print('Dropping these')
            
            # Dropping the duplicates because there is just one duplicate b number 
            # and it is found in none of the E. coli expression datasets anyway.
            todrop = my_df.b_number.isin(count_bs[mask].index.values)
            my_df = my_df.drop(my_df.loc[todrop].index, axis=0)
            
    # Store in memory for later cells if needed
    cds_dfs[idx] = my_df
    
    # Save as CSV
    print('Saving reference proteome for {} to CSV'.format(idx))
    out_fname = row.CDS_NOSC_csv_fname
    my_df.to_csv(out_fname, index=False)


# Load COG IDs and functional categories, convert to CSV for later use.
print('Loading COG functional categories...')
cog_ids = pd.read_csv('data/COG-20.def.tab.txt', sep='\t', encoding='cp1252', header=None)
cog_ids.columns = 'COG_ID,categories,name,gene,pathway,PubMed_IDs,PDB_ID'.split(',')
cog_ids = cog_ids.set_index('COG_ID')

cog_func = pd.read_csv('data/COG-fun-20.tab.txt', sep='\t', header=None)
cog_func.columns = 'category,color,description'.split(',')
cog_func = cog_func.set_index('category')

cog_func.to_csv('data/COG-fun-20.csv', index=True)
cog_ids.to_csv('data/COG-20.def.csv', index=True)


cols = ['aa_seq', 'num_aas', 'NC', 'Ce', 'NOSC', 'mw_daltons',
       'transmembrane_aas', 'transmembrane_Cs', 'fraction_transmembrane',
       'fraction_transmembrane_C', 'primary_accession', 'accessions',
       'gene_name', 'description', 'locus_tags', 'GO_terms', 'COG_IDs',
       'KEGG_IDs', 'isoform_accessions', 'KEGG_path_IDs',
       'KEGG_pathways', 'eC_ratio', 'organism']

# Concate the dataframes into a larger one 
# Need to reset indices so they are unique keys. 
long_nosc_df = pd.concat([df.reset_index()[cols] for key, df in cds_dfs.items()]).reset_index().drop(
    'index',axis=1).set_index('primary_accession')

# For E. coli, add b_number back in for later cross-referencing
coli_tmp = cds_dfs['coli'].set_index('primary_accession')
long_nosc_df['b_number'] = None
long_nosc_df.loc[coli_tmp.index, 'b_number'] = coli_tmp.b_number
long_nosc_df = long_nosc_df.reset_index()

# Add columns for COG categories.
# Using the first listed COG ID to retrieve categorial information.
cog_class_dict = dict(primary_COG_category=[], secondary_COG_category=[])
for idx, row in tqdm(long_nosc_df.iterrows()):
    COGs = row.COG_IDs.split(',')
    prim, sec = '', ''
    for c in COGs:
        if c in cog_ids.index:
            cats = cog_ids.loc[c].categories
            prim = cats[0]
            if len(cats) > 1:
                sec = cats[1]
            break
    cog_class_dict['primary_COG_category'].append(prim)
    cog_class_dict['secondary_COG_category'].append(sec)
for l, col in cog_class_dict.items():
    long_nosc_df[l] = col

print('Saving long ZC table to CSV...')
long_nosc_df.infer_objects()
long_nosc_df.to_csv('data/genomes/all_ref_prot_NOSC.csv', index=False)
long_nosc_df.head()