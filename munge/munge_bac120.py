import argparse
import glob
import numpy as np
import os
import pandas as pd

from Bio import SeqIO
from multiprocessing import Pool
from os import path
from seq_util import *
from tqdm import tqdm

"""
Calculate Z_C values for the GTDB bac120 marker genes.

We are considering 120 marker genes across 60k genomes.

A single file takes about 90s to process, so in serial this
would take about 3 hrs. With 4 cores, it takes about 1 hr.
"""

def _argparser():
    """Generates a parser for command-line arguments.
    
    Returns:
        argparse.ArgumentParser: parser for command-line arguments.
    """
    parser = argparse.ArgumentParser(
        prog='calc_nosc.py',
        description='Calculates Z_C values for protein sequences in FASTA files.',
        epilog='Text at the bottom of help')

    default_indir = '../kofamscan/bac120_marker_genes_reps_r207/faa/'
    parser.add_argument('-i', '--in', dest='input', default=default_indir,
                        help='input directory where ')
    parser.add_argument('-o', '--outdir', dest='outdir', default='output/gtdb/r207/',
                        help='output directory')
    parser.add_argument('-n', '--num_processes', dest='num_processes',
                        type=int, default=1,
                        help='number of processes to spawn')
    return parser


def _do_args():
    """Parses command-line arguments.
    
    Returns:
        tuple: (list of input FASTA filenames,
                output directory,
                number of processors)
    """
    p = _argparser()
    args = p.parse_args()
    
    fnames = [args.input]
    if path.isdir(args.input):
        # load all the FASTA amino acid
        g = path.join(args.input, '*.faa')
        fnames = glob.glob(g)
    
    if not path.isdir(args.outdir):
        os.mkdir(args.outdir)
    
    return fnames, args.outdir, args.num_processes


def _do_single_fasta(idx, fpath):
    bac120_nosc_data = dict(gene_id=[], accession=[], NOSC=[])

    # parse the gene identifier (PFAM, TIGERFAM) from the filename
    _, fname = path.split(fpath)  
    print('Processing file {0}: {1}...'.format(idx, fname))  
    head, _ = path.splitext(fname)
    gene_identified = head
    for record in tqdm(SeqIO.parse(fpath, 'fasta')):
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
    return bac120_nosc_data

def do_main():
    # Parse the command-line arguments
    fnames, outdir, num_processes = _do_args()

    # Looping over fasta files of the GTDB bac120 sequences
    # NOTE: these are downloaded from GTDB on their website

    dirname = '../kofamscan/bac120_marker_genes_reps_r207/faa/'
    fnames = glob.glob(dirname + 'TIGR0002*.faa')
    print('Found {0} files to process...'.format(len(fnames)))
    print('Output directory: {0}'.format(outdir))

    # Load the sequences and calculate the Z_C values, running each file in parallel
    print('Starting {0} processes to calculate NOSC values...'.format(num_processes))
    with Pool(num_processes) as p:
        deferred = p.starmap_async(
            _do_single_fasta,
            [(idx, f) for idx,f in enumerate(fnames)])
        # Wait for all the processes to finish, get the results
        results = deferred.get()

    # Combine the results
    bac120_nosc_data = dict(gene_id=[], accession=[], NOSC=[])
    for res in results:
        for k, v in res.items():
            bac120_nosc_data[k].extend(v)

    # Convert to a dataframe
    bac120_nosc_df = pd.DataFrame(bac120_nosc_data)

    # Save DF as a CSV in the output directory
    print('Saving bac120 NOSC values to files...')
    out_fname = path.join(outdir, 'bac120_nosc_vals.csv')
    bac120_nosc_df.to_csv(out_fname, index=False)

    # Pivot so that we have rows per genome of ZC values
    bac120_nosc_mat = bac120_nosc_df.pivot(index='accession', columns='gene_id', values='NOSC')
    out_fname = path.join(outdir, 'bac120_nosc_vals_wide.csv')
    bac120_nosc_mat.to_csv(out_fname)

    # Save a version with less precision for submission -- smaller file.
    out_fname = path.join(outdir, 'bac120_nosc_vals_wide_compressed.csv')
    bac120_nosc_mat.to_csv(out_fname, float_format='%.6f')

if __name__ == '__main__':
    do_main()