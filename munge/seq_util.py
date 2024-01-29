import numpy as np
import pandas as pd

from os import path
from Bio import SeqIO
from Bio.SeqUtils import molecular_weight

__author__ = 'Avi Flamholz'


"""
This module contains functions for calculating chemical properties of protein sequences.
"""

cwd = path.dirname(__file__)
_DEFAULT_AA_DF = pd.read_csv(path.join(cwd, '../data/aa_nosc.csv')).set_index('letter_code')


def calc_protein_nosc(seq, aa_df=None):
    """Calculates the Z_C of the protein.
    
    i.e. the nominal oxidation state of C, or NOSC.
    
    Args:
        seq: the sequence as a string or Seq object. 
        aa_df: a dataframe containing the amino acid Z_C values in the appropriate format. 
            Defaults to the value statically loaded above. 
    
    Returns:
        A two-tuple (Ce_tot, NC_tot) where Ce is the formal number of valence electrons
        on C atoms in the polypeptide and NC is the number of C atoms. Z_C = Ce_tot/NC_tot

    Raises:
        ValueError: if the sequence contains unknown or non-specific amino acids.
    """
    if aa_df is None:
        aa_df = _DEFAULT_AA_DF
    aa_chars = set(aa_df.index.tolist())
    seq_list = list(seq.rstrip('*'))
    valid = set(seq_list).issubset(aa_chars)
    if not valid:
        invalid_aas = set(seq_list).difference(aa_chars)
        msg = 'Sequence contains unknown or non-specific amino acids "{0}"'.format(
            ','.join(invalid_aas))
        raise ValueError(msg)
        
    # NOSC of a protein is the C-weighted average of 
    # constituent amino acids.
    seqx = aa_df.loc[seq_list]
    nc_tot = seqx.NC.sum()
    ce_tot = (seqx.NOSC*seqx.NC).sum()
    return ce_tot, nc_tot


def calc_protein_nosc_no_raise(seq, aa_df=None):
    """Calculates the Z_C of the protein without raising any errors.
    
    i.e. the nominal oxidation state of C, or NOSC.
    
    Args:
        seq: the sequence as a string or Seq object. 
        aa_df: a dataframe containing the amino acid Z_C values in the appropriate format. 
            Defaults to the value statically loaded above. 
    
    Returns:
        A two-tuple (Ce_tot, NC_tot) where Ce is the formal number of valence electrons
        on C atoms in the polypeptide and NC is the number of C atoms. Z_C = Ce_tot/NC_tot
    """
    try: 
        return calc_protein_nosc(seq, aa_df)
    except ValueError:
        return np.nan, np.nan