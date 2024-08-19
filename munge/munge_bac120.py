import numpy as np
import pandas as pd
import glob

from tqdm import tqdm
from Bio import SeqIO
from os import path
from seq_util import *

"""
Calculate Z_C values for the GTDB bac120 marker genes.

This script will take quite a while to run. Perhaps hours. 

Could take advantage of parallelization here, but it's not worth the effort.
"""

# Load all the fasta filenames of the GTDB bac120 sequences
# NOTE: these are easily downloaded from GTDB on their website
print('Loading bac120 sequences...')
# Change to the directory where the sequences are stored
dirname = '../kofamscan/bac120_marker_genes_reps_r207/faa/'
fnames = glob.glob(dirname + '*.faa')

# Parse all the data from 
bac120_nosc_data = dict(gene_id=[], accession=[], NOSC=[])
for fpath in tqdm(fnames):
    # parse the gene identifier (PFAM, TIGERFAM) from the filename
    p, f = path.split(fpath)    
    head, tail = path.splitext(f)
    gene_identified = head
    for record in SeqIO.parse(fpath, 'fasta'):
        bac120_nosc_data['gene_id'].append(gene_identified)
        bac120_nosc_data['accession'].append(record.id)
        try:
            Ce, NC = calc_protein_nosc(record.seq)
            bac120_nosc_data['NOSC'].append(Ce/NC)
        except ValueError:
            # This sequence contains a non-specific amino acid
            # TODO: could calculate the NOSC with some error range
            # presuming either random aminos or some null distribution
            bac120_nosc_data['NOSC'].append(np.NaN)

# Convert to a dataframe
bac120_nosc_df = pd.DataFrame(bac120_nosc_data)

# Save DF as a CSV
print('Saving bac120 NOSC values to files...')
bac120_nosc_df.to_csv('data/gtdb/r207/bac120_nosc_vals.csv')

# Pivot so that we have rows per genome of ZC values
bac120_nosc_mat = bac120_nosc_df.pivot(index='accession', columns='gene_id', values='NOSC')
bac120_nosc_mat.to_csv('data/gtdb/r207/bac120_nosc_vals_wide.csv')
bac120_nosc_mat.to_csv('data/gtdb/r207/bac120_nosc_vals_wide_compressed.csv',
                       float_format='%.6f')