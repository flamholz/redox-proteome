import numpy as np
import pandas as pd
import re

from Bio import SeqIO
from Bio.SeqUtils import molecular_weight
from functools import reduce
from operator import concat


def calc_protein_nosc(seq, aa_nosc_df):
    aa_chars = set(aa_nosc_df.index.tolist())
    seq_list = list(seq.rstrip('*'))
    valid = set(seq_list).issubset(aa_chars)
    if not valid:
        return np.NaN, np.NaN
        
    # NOSC of a protein is the C-weighted average of 
    # constituent amino acids.
    seqx = aa_nosc_df.loc[seq_list]
    nc_tot = seqx.NC.sum()
    ce_tot = (seqx.NOSC*seqx.NC).sum()
    return ce_tot, nc_tot


#### DEPRECATED ####
# TODO: DELETE THIS now that we are using ref proteomes.

def calc_proteome_nosc(fname, aa_nosc_df):
    nosc_data = dict(locus_tag=[], aaseq=[], NOSC=[], NC=[], Ce_tot=[], MW=[], naa=[], gene_name=[], gene_id=[],
                     inner_membrane=[], outer_membrane=[])
    
    with open(fname) as infile:
        for record in SeqIO.parse(infile, format='gb'):
            for feature in record.features:
                if feature.type == 'CDS' and 'translation' in feature.qualifiers:
                    aaseq = feature.qualifiers['translation'][0]
                    nosc_data['locus_tag'].append(feature.qualifiers['locus_tag'][0])
                    nosc_data['aaseq'].append(aaseq)
                    nosc_data['naa'].append(len(aaseq))
                    mw = molecular_weight(aaseq, seq_type='protein')
                    nosc_data['MW'].append(mw)
                    ce_tot, nc = calc_protein_nosc(aaseq, aa_nosc_df)
                    nosc_data['Ce_tot'].append(ce_tot)
                    nosc_data['NOSC'].append(ce_tot/nc)
                    nosc_data['NC'].append(nc)
                    nosc_data['gene_name'].append(feature.qualifiers.get('gene', [None])[0])
                    
                    # grab some annotations from the notes
                    gene_id, inner_membrane, outer_membrane = None, False, False
                    # are separated by semicolons by convention and often include GO annotations.
                    notes = feature.qualifiers.get('note', [''])
                    notes = reduce(concat, [n.split(';') for n in notes])
                    for note in notes:
                        if note.startswith('ORF_ID:'):
                            gene_id = note.split(':')[-1]
                        if note.find('GO:0019866 - organelle inner membrane') > 0:
                            inner_membrane = True
                        if note.find('GO:0009279 - cell outer membrane') > 0:
                            outer_membrane = True
                    nosc_data['gene_id'].append(gene_id)
                    nosc_data['inner_membrane'].append(inner_membrane)
                    nosc_data['outer_membrane'].append(outer_membrane)

    nosc_df = pd.DataFrame(nosc_data)
    nosc_df['NOSC'] = nosc_df.Ce_tot / nosc_df.NC
    nosc_df['eC_ratio'] =  4-nosc_df.NOSC
    return nosc_df

def calc_proteome_nosc_uniprot(fname, aa_nosc_df):
    """Uniprot data comes as FASTA with defined headers."""
    nosc_data = dict(locus_tag=[], aaseq=[], NC=[], Ce_tot=[], MW=[], naa=[], gene_name=[], gene_id=[])
    
    # gene name is demarcated by GN=
    pat = re.compile(' GN=(\S+)')

    with open(fname) as infile:
        for record in SeqIO.parse(infile, format='fasta'):
            desc = record.description
            
            # Format is "db|unique_id|entry_name etc...
            gid = desc.split('|')[1]
            aaseq = str(record.seq).strip('*')
            name_matches = pat.findall(desc)
            gene_name = None
            if name_matches:
                gene_name = name_matches[0]
                            
            try:
                mw = molecular_weight(aaseq, seq_type='protein')
                ce_tot, nc = calc_protein_nosc(aaseq, aa_nosc_df)
            except:
                mw, ce_tot, nc, nosc = None, None, None, None
            
            nosc_data['locus_tag'].append(record.id)
            nosc_data['gene_id'].append(gid)
            nosc_data['gene_name'].append(gene_name)
            nosc_data['aaseq'].append(aaseq)
            nosc_data['naa'].append(len(aaseq))
            nosc_data['MW'].append(mw)
            nosc_data['Ce_tot'].append(ce_tot)
            nosc_data['NC'].append(nc)
            
    nosc_df = pd.DataFrame(nosc_data)
    nosc_df['NOSC'] = nosc_df.Ce_tot / nosc_df.NC
    nosc_df['eC_ratio'] =  4-nosc_df.NOSC
    return nosc_df
    