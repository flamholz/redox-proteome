import gzip 
import pandas as pd
import re

from lxml import etree
from Bio import SeqIO


KEYS = ['aa_seq', 'num_aas', 'mw_daltons', 'transmembrane_aas', 'fraction_transmembrane',
        'primary_accession', 'accessions', 'gene_name', 'description', 'locus_tags',
        'GO_terms', 'COG_IDs', 'KEGG_IDs', 'isoform_accessions']
B_PAT = re.compile('^b\d+$')

def _open_helper(fname, gz):
    if gz:
        return gzip.open(fname)
    return open(fname)

def uniprot_xml2df(fname, gz=True, extract_b_number=False):
    handle = _open_helper(fname, gz)
    data_dict = dict([(k,[]) for k in KEYS])
    if extract_b_number:
        data_dict['b_number'] = []
    
    tree = etree.parse(handle)
    root = tree.getroot()
    nsmap = root.nsmap
    
    for entry in root.findall('entry', nsmap):
        # grab the protein sequence
        seq = entry.find('sequence', nsmap)
        data_dict['aa_seq'].append(seq.text)
        data_dict['mw_daltons'].append(float(seq.attrib['mass']))
        seq_len = len(seq.text)
        data_dict['num_aas'].append(seq_len)
        
        # all the protein accessions
        accessions = entry.findall('accession', nsmap)
        acc_txts = [a.text for a in accessions]
        # textual list of accessions
        data_dict['accessions'].append(','.join(acc_txts))
        # first accession is the main one... 
        # https://www.uniprot.org/help/accession_numbers
        primary_acc = None if not acc_txts else acc_txts[0]
        data_dict['primary_accession'].append(primary_acc)
        
        # all the gene names
        gene = entry.find('gene', nsmap)
        names_txt = [n.text for n in gene]
        # textual list of names
        data_dict['locus_tags'].append(','.join(names_txt))
        # first name is the main one... 
        primary_name = None if not names_txt else names_txt[0]
        data_dict['gene_name'].append(primary_name)
        
        # description
        primary_desc = entry.findall('protein/recommendedName/fullName', nsmap)
        if primary_desc:
            data_dict['description'].append(primary_desc[0].text)
        else:
            data_dict['description'].append(None)
        
        # count up the number of transmembrane amino acids
        tm_locations = entry.findall("feature[@type='transmembrane region']/location", nsmap)
        tm_len = 0
        tm_fail = False
        for tm_loc in tm_locations:
            begin = tm_loc.find('begin', nsmap)
            begin_pos = int(begin.attrib.get('position', -1))                
            end = tm_loc.find('end', nsmap)
            end_pos = int(end.attrib.get('position', -1))
            tm_len += end_pos - begin_pos
            
            # in some cases the boundaries of the TM section are not know.
            # would rather skip these examples. 
            if begin_pos < 0 or end_pos < 0:
                tm_fail = True
                tm_len = -1
                break
                
        # Negative tm_len is a sentinel that we failed to extract
        data_dict['transmembrane_aas'].append(tm_len)
        data_dict['fraction_transmembrane'].append(tm_len/seq_len)

        # extract kegg ID and go terms
        db_refs = entry.findall('dbReference', nsmap)
        kegg_ids = [r.attrib['id'] for r in db_refs if r.attrib['type'] == 'KEGG']
        data_dict['KEGG_IDs'].append(','.join(kegg_ids))
        
        go_ids = [r.attrib['id'] for r in db_refs if r.attrib['type'] == 'GO']
        data_dict['GO_terms'].append(','.join(go_ids))
        
        cog_ids = [r.attrib['id'] for r in db_refs if r.attrib['type'] == 'eggNOG']
        data_dict['COG_IDs'].append(','.join(cog_ids))
        
        # need to grab the IDs for the isoforms so we can fetch their sequences
        isoform_ids = []
        isoforms_elts = entry.findall('comment/isoform', nsmap)
        for iso in isoforms_elts:
            isoform_ids.append(iso.find('id', nsmap).text)
        data_dict['isoform_accessions'].append(','.join(isoform_ids))
        
        if extract_b_number:
            # E. coli specific code
            b = None
            for t in names_txt:
                if B_PAT.match(t):
                    # First one is usually the best one
                    b = t
                    break 
                    
            if b == None:
                # check KEGG IDs also
                bs = [k for k in kegg_ids if k.startswith('eco:b')]
                b = None if not bs else bs[0].strip('eco:')
            data_dict['b_number'].append(b)
            
    handle.close()
    return pd.DataFrame(data_dict)


def add_KEGGpathways2df(cds_df, kegg_mapping_fname, kegg_pathways_fname):
    kegg_mapping = pd.read_csv(kegg_mapping_fname, sep='\t', header=None)
    kegg_mapping.columns = 'KEGG_gene_ID,KEGG_path_ID'.split(',')
    kegg_mapping = kegg_mapping.set_index('KEGG_gene_ID')

    kegg_pathways = pd.read_csv(kegg_pathways_fname, sep='\t', header=None)
    kegg_pathways.columns = 'KEGG_path_ID,KEGG_pathway_description'.split(',')
    kegg_pathways = kegg_pathways.set_index('KEGG_path_ID')
    kegg_mapping['KEGG_pathway_description'] = kegg_pathways.loc[
        kegg_mapping.KEGG_path_ID].KEGG_pathway_description.values
    
    kpids = []
    kpdescs = []
    for idx, row in cds_df.iterrows():
        kids = row.KEGG_IDs.split(',')
        pids = []
        pdescs = []
        for kid in kids:
            try:
                ps = kegg_mapping.loc[[kid]].KEGG_path_ID.unique().tolist()
                ds = kegg_mapping.loc[[kid]].KEGG_pathway_description.unique().tolist()
                pids.extend(ps)
                pdescs.extend(ds)
            except KeyError:
                # Some links from refseq are not actually in the KEGG mapping.
                # This is especially so for E. coli with multiple gene naming schemes.
                continue
        
        kpids.append(','.join(pids))
        kpdescs.append(','.join(pdescs))
    cds_df['KEGG_path_IDs'] = kpids
    cds_df['KEGG_pathways'] = kpdescs
    return cds_df


def add_isoforms2df(cds_df, isoform_fname):
    # maps isoform ID to the primary accession
    relevant_isoforms = dict()
    for idx, row in cds_df.iterrows():
        for iso_id in row.isoform_accessions.split(','):
            if iso_id:
                relevant_isoforms[iso_id] = row.primary_accession 
    
    rows2add = []
    reindexed_cds = cds_df.set_index('primary_accession')
    for rec in SeqIO.parse(isoform_fname, 'fasta'):
        desc = rec.description
        iso_id = desc.split('|')[1]
        if iso_id in relevant_isoforms:
            primary_id = relevant_isoforms[iso_id]
            iso_row_dict = reindexed_cds.loc[primary_id].to_dict()
            iso_row_dict['primary_accession'] = iso_id
            seq_str = str(rec.seq)
            iso_row_dict['aa_seq'] = seq_str
            iso_row_dict['num_aas'] = len(seq_str)
            rows2add.append(iso_row_dict)
            
            # append isoform ID to b_number to avoid duplicate b-numbers
            if 'b_number' in iso_row_dict:
                iso_row_dict['b_number'] = '{0}-{1}'.format(
                    iso_row_dict['b_number'], iso_id)
    
    return cds_df.append(rows2add, ignore_index=True)
        