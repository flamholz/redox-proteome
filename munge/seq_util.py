import numpy as np
import pandas as pd

from os import path
from Bio import SeqIO
from Bio.SeqUtils import molecular_weight
from rdkit import Chem
from rdkit.Chem import rdmolops

from collections import Counter

__author__ = 'Avi Flamholz'


"""
This module contains functions for calculating chemical properties of protein sequences.
"""

cwd = path.dirname(__file__)
_DEFAULT_AA_DF = pd.read_csv(path.join(cwd, '../data/aa_nosc.csv')).set_index('letter_code')


def _count_atoms(mol): 
    """Counts the number of atoms in a molecule.
    
    Returns:
        A Counter object with the atom symbols as keys and the counts as values.
    """
    return Counter([a.GetSymbol() for a in mol.GetAtoms()])


def calc_nosc_from_smiles(smiles_str):
    """Calculates the Z_C of a molecule from its SMILES string.
    
    Args:
        smiles_str: a SMILES string representing the molecule.
    
    Returns:
        The formal C redox state of the molecule calculated by the formula:
            Z_C = 4 - (1/N_C)*(-q + 4 N_C + N_H - 3 N_N - 2 N_O + 5 N_P - 2 N_S)
        where q is the formal charge of the molecule, N_C is the number of C atoms, etc.
    """
    # Read the molecule and add hydrogens
    mol = Chem.AddHs(Chem.MolFromSmiles(smiles_str))
    atom_counts = _count_atoms(mol)
    q = rdmolops.GetFormalCharge(mol)

    # Get atom counts
    N_C = atom_counts.pop('C', 0)
    N_H = atom_counts.pop('H', 0)
    N_N = atom_counts.pop('N', 0)
    N_O = atom_counts.pop('O', 0)
    N_P = atom_counts.pop('P', 0)
    N_S = atom_counts.pop('S', 0)

    # Only works for organic molecules with these heteroatoms
    assert len(atom_counts) == 0, 'Unknown atom types in molecule'
    return 4 - (1/N_C)*(-q + 4*N_C + N_H - 3*N_N - 2*N_O + 5*N_P - 2*N_S)


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