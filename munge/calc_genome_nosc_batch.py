import argparse
import glob
import os
import pandas as pd
import seaborn as sns

from Bio import SeqIO
from multiprocessing import Pool
from os import path
from seq_util import *
from tqdm import tqdm

__author__ = 'Avi Flamholz'


"""
Uses multiprocessing to calculate the NOSC values for all the protein sequences in a set of FASTA files.

Written for use on Phillips lab server, which has 48 cores. 
"""


def _argparser():
    """Generates a parser for command-line arguments.
    
    Returns:
        argparse.ArgumentParser: parser for command-line arguments.
    """
    parser = argparse.ArgumentParser(
        prog='calc_genome_nosc_batch.py',
        description='Calculates Z_C values for protein sequences in FASTA files.',
        epilog='Text at the bottom of help')

    parser.add_argument('-i', '--in', dest='input',
                        help='input directory or filename')
    parser.add_argument('-o', '--outdir', dest='outdir',
                        help='output directory')
    parser.add_argument('-n', '--num_processes', dest='num_processes',
                        type=int, default=1,
                        help='number of processes to spawn to use')
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
        # load all the FASTA files in the directory
        g1 = path.join(args.input, '*.fa')
        g2 = path.join(args.input, '*.faa')
        g3 = path.join(args.input, '*.fasta')
        fnames = glob.glob(g1) + glob.glob(g2) + glob.glob(g3)
    
    if not path.isdir(args.outdir):
        os.mkdir(args.outdir)
    
    return fnames, args.outdir, args.num_processors


def _get_genome_accession(fname):
    """Parses the genome accession from the filename.
    
    Args:
        fname (str): path to the input FASTA file.
    
    Returns:
        str: the genome accession.
    """
    _, f = path.split(fname)    
    head, _ = path.splitext(f)
    return head.split('.')[0]


def _get_out_fname(fname, out_dir):
    """Generates the output filename for a given input FASTA file.

    Args:
        fname (str): path to the input FASTA file.
        out_dir (str): path to the output directory.
    
    Returns:
        str: path to the output CSV file.
    """
    out_fname = _get_genome_accession(fname) + '.nosc.csv'
    return path.join(out_dir, out_fname)


def _do_single_fasta(fname):
    """
    Calculates the NOSC values for all the protein sequences in a single fasta. 
    
    Args:  
        fname (str): path to the fasta file.

    Returns:
        pandas.DataFrame: dataframe with the NOSC values and metadata.
    """
    bac120_nosc_data = dict(gene_id=[], genome_accession=[],
                            Ce=[], NC=[], NOSC=[])
    accession_id = _get_genome_accession(fname)
    for record in SeqIO.parse(fname, 'fasta'):
        bac120_nosc_data['genome_accession'].append(accession_id)
        bac120_nosc_data['gene_id'].append(record.id)
        try:
            Ce, NC = calc_protein_nosc(record.seq)
            bac120_nosc_data['Ce'].append(Ce)
            bac120_nosc_data['NC'].append(NC)
            bac120_nosc_data['NOSC'].append(Ce/NC)
        except ValueError:
            # This sequence contains a non-specific amino acid
            # TODO: could calculate the NOSC with some error range
            # presuming either random aminos or some null distribution 
            bac120_nosc_data['Ce'].append(None)
            bac120_nosc_data['NC'].append(None)
            bac120_nosc_data['NOSC'].append(None)

    return fname, pd.DataFrame(bac120_nosc_data)


def _calc_genome_nosc(genome_df):
    """Calculates the NOSC values for a single genome.

    Args:
        genome_df (pandas.DataFrame): dataframe with the NOSC values for a single genome.

    Returns:
        float: the NOSC value for the genome.
    """
    return genome_df['Ce'].sum() / genome_df['NC'].sum()


def do_main():
    """
    Spawns a process for each input FASTA file, calculates the NOSC values for all the
    protein sequences in the file, and saves the results to a CSV file.
    """
    in_fnames, outdir, n_proc = _do_args()
    genome_averages = dict(genome_accession=[], genome_avg_NOSC=[])

    print("Using {0} workers to process FASTA files".format(n_proc))
    with Pool(n_proc) as p:
        deferred = p.starmap_async(_do_single_fasta, [(fname,) for fname in in_fnames])
        # Wait for all the processes to finish, get the results
        results = deferred.get()

    for my_fname, res_df in results:
        out_fname = path.join(_get_out_fname(my_fname, outdir))
        res_df.to_csv(out_fname, index=False)
        
        # Calculate the carbon-weighted genome average NOSC value
        genome_averages['genome_accession'].append(_get_genome_accession(my_fname))
        genome_averages['genome_avg_NOSC'].append(_calc_genome_nosc(res_df))

    # Save the genome averages to a CSV file
    genome_averages_df = pd.DataFrame(genome_averages)
    genome_averages_df.to_csv(path.join(outdir, 'genome_averages.csv'), index=False)
    

if __name__ == '__main__':
    do_main()