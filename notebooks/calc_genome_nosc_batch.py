import argparse
import glob
import os
import pandas as pd
import seaborn as sns

from tqdm import tqdm
from Bio import SeqIO
from os import path
from seq_util import *


__author__ = 'Avi Flamholz'


def _argparser():
    """Generates a parser for command-line arguments.
    
    Returns:
        argparse.ArgumentParser: parser for command-line arguments.
    """
    parser = argparse.ArgumentParser(
        prog='calc_nosc.py',
        description='Calculates Z_C values for protein sequences in FASTA files.',
        epilog='Text at the bottom of help')

    parser.add_argument('-i', '--in', dest='input',
                        help='input directory or filename')
    parser.add_argument('-o', '--outdir', dest='outdir',
                        help='output directory')
    return parser


def _do_args():
    """Parses command-line arguments.
    
    Returns:
        tuple: (list of input FASTA filenames, output directory)
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
    
    return fnames, args.outdir


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

    return pd.DataFrame(bac120_nosc_data)


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
    Loops over all the FASTA files in the input directory and calculates
    NOSC values for each protein sequence. Saves the results to a CSV file.
    """
    in_fnames, outdir = _do_args()

    genome_averages = dict(genome_accession=[], genome_avg_NOSC=[])
    for fname in tqdm(in_fnames):
        df = _do_single_fasta(fname)
        out_fname = path.join(_get_out_fname(fname, outdir))
        df.to_csv(out_fname, index=False)

        # Calculate the carbon-weighted genome average NOSC value
        genome_averages['genome_accession'].append(_get_genome_accession(fname))
        genome_averages['genome_avg_NOSC'].append(_calc_genome_nosc(df))

    # Save the genome averages to a CSV file
    genome_averages_df = pd.DataFrame(genome_averages)
    genome_averages_df.to_csv(path.join(outdir, 'genome_averages.csv'), index=False)
    

if __name__ == '__main__':
    do_main()