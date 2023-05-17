import argparse
import glob
import os
import numpy as np
import pandas as pd

from Bio import SeqIO
from multiprocessing import Pool
from os import path
from sklearn.mixture import BayesianGaussianMixture

__author__ = 'Avi Flamholz'


"""
Uses multiprocessing to fit the NOSC distributions of all the protein sequences to a guassian mixture model.

Written for use on Phillips lab server, which has 48 cores. 
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

    parser.add_argument('-i', '--in', dest='input',
                        help='input directory or filename')
    parser.add_argument('-o', '--outdir', dest='outdir',
                        help='output directory')
    parser.add_argument('-m', '--max_infiles', dest='max_infiles',
                        type=int, default=None,
                        help='maximum number of files to process')
    parser.add_argument('-n', '--num_processors', dest='num_processors',
                        type=int, default=1,
                        help='number of processors to use')
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
        # load all the CSV files in the directory
        g1 = path.join(args.input, '*.nosc.csv')
        fnames = glob.glob(g1)

    # Truncate the list of input files if requested
    if args.max_infiles is not None:
        fnames = fnames[:args.max_infiles]
    
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
    return f.strip('.nosc.csv') 


def _do_single_csv(fname):
    """
    Load the distribution of NOSC values and fit a Gaussian mixture model to it.
    
    Fits a two-component model on the assumption that there is one component for the soluble proteins
    and another for the membrane proteins.

    Args:  
        fname (str): path to the fasta file.

    Returns:
        A dictionary with the following keys:
            genome_accession (str): the genome accession.
            genome_mean (float): the mean NOSC value for the genome.
            soluble_protein_mean (float): the mean NOSC value for the soluble proteins.
            soluble_protein_var (float): the variance of the NOSC values for the soluble proteins.
            membrane_protein_mean (float): the mean NOSC value for the membrane proteins.
            membrane_protein_var (float): the variance of the NOSC values for the membrane proteins.
            aic_score (float): the AIC score for the GMM fit.
            bic_score (float): the BIC score for the GMM fit.
    """
    my_df = pd.read_csv(fname)
    gmm = BayesianGaussianMixture(n_components=2, init_params='k-means++', max_iter=10000)
    mask = my_df.NOSC.notnull()
    finite_vals = my_df[mask].NOSC.values.reshape(-1, 1)
    gmm.fit(finite_vals)

    # Calculate the mean NOSC value for the genome
    genome_mean = my_df['Ce'].sum() / my_df['NC'].sum()

    # Calculate the mean and variance of each component
    # Note that argsort sorts in ascending order
    component_means = gmm.means_.flatten()
    idxs = np.argsort(component_means)
    component_means = component_means[idxs]
    component_vars = gmm.covariances_.flatten()[idxs]
    component_weights = gmm.weights_.flatten()[idxs]
    
    return dict(genome_accession=_get_genome_accession(fname), genome_mean=genome_mean,
                soluble_protein_mean=component_means[1], soluble_protein_var=component_vars[1], soluble_protein_weight=component_weights[1],
                membrane_protein_mean=component_means[0], membrane_protein_var=component_vars[0], membrane_protein_weight=component_weights[0],
                log_likelihood_score=gmm.score(finite_vals), converged=gmm.converged_)


def do_main():
    """
    Spawns a process for each input FASTA file, calculates the NOSC values for all the
    protein sequences in the file, and saves the results to a CSV file.
    """
    in_fnames, outdir, n_proc = _do_args()
    fit_data = [] # list of dictionaries

    print("Using {0} workers to process FASTA files".format(n_proc))
    with Pool(n_proc) as p:
        deferred = p.starmap_async(_do_single_csv, [(fname,) for fname in in_fnames])
        # Wait for all the processes to finish, get the results
        results = deferred.get()

    genome_fit_df = pd.DataFrame(results)
    genome_fit_df.to_csv(path.join(outdir, 'genome_nosc_fits.csv'), index=False)
    

if __name__ == '__main__':
    do_main()