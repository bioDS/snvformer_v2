#!/usr/bin/env python3
from pyensembl import EnsemblRelease
import torch
import numpy as np
import pandas as pd
import biomart
import pickle
from os.path import exists

cache_dir = "/data/ukbb/cache/"
ukbb_data = "/data/ukbb/net_input/"
biogrid_file = "BIOGRID-ALL-4.4.213.tab3.txt"

def get_ensembl_mappings():
    cache_file = cache_dir + "ensemble_entrez_cache.pickle"
    if exists(cache_file):
        with open(cache_file, "rb") as f:
            (a, b) = pickle.load(f)
            return a, b
    server = biomart.BiomartServer('http://grch37.ensembl.org/biomart')
    # server = biomart.BiomartServer('http://ensembl.org/biomart')
    mart = server.datasets['hsapiens_gene_ensembl']

    # List the types of data we want
    attributes = ['ensembl_gene_id', 'entrezgene_id']

    # Get the mapping between the attributes
    response = mart.search({'attributes': attributes})

    ensembl_to_entrez = {None: None}
    entrez_to_ensemble = {None: None}
    for line in response.iter_lines():
        line = line.decode('utf-8')
        line = line.split('\t')
        # The entries are in the same order as in the `attributes` variable
        ensembl_gene = line[0]
        entrez_gene= line[1]

        if (entrez_gene != ''):
            entrez_to_ensemble[entrez_gene] = ensembl_gene
            ensembl_to_entrez[ensembl_gene] = entrez_gene

    with open(cache_file, "wb") as f:
        pickle.dump((ensembl_to_entrez, entrez_to_ensemble), f)
    return ensembl_to_entrez, entrez_to_ensemble

# def get_all_entrez_ids():
#     _, entrez_to_ensembl = get_ensembl_mappings()
#     all_entrez_ids = np.array([i for i in entrez_to_ensembl.keys()])
#     return(all_entrez_ids)

def get_all_entrez_ids():
    biogrid_info = pd.read_csv(ukbb_data + biogrid_file, sep='\t', low_memory=False)
    human_info = biogrid_info[(biogrid_info['Organism Name Interactor A'] == "Homo sapiens") & (biogrid_info['Organism Name Interactor B'] == "Homo sapiens")]

    entrez_pairs = human_info[['Entrez Gene Interactor A', "Entrez Gene Interactor B"]].astype(int)
    biogrid_entrez_all = np.unique(entrez_pairs.to_numpy().flatten())

    _, entrez_to_ensembl = get_ensembl_mappings()
    biomart_entrez_all = np.array([i for i in entrez_to_ensembl.keys()])
    biomart_entrez_all = np.delete(biomart_entrez_all, np.where(biomart_entrez_all == None))
    biomart_entrez_all = biomart_entrez_all.astype(int)

    all_ids = np.unique(np.concatenate((biogrid_entrez_all, biomart_entrez_all))).astype(str)
    return all_ids


class GeneIndex:
    def __init__(self, gene_string_list):
        all_ids = get_all_entrez_ids()
        string_to_ind = {'None': 0}
        ind_to_string = {0: 'None'}
        id_vec = torch.zeros(len(gene_string_list), dtype=torch.long)
        next_ind = 1
        for ind, gene_id in enumerate(all_ids):
            if gene_id is not None:
                if gene_id not in string_to_ind:
                    ind_to_string[next_ind] = gene_id
                    string_to_ind[gene_id] = next_ind
                    next_ind += 1
        for ind, gene_id in enumerate(gene_string_list):
            if gene_id is not None:
                if gene_id in string_to_ind:
                    id_vec[ind] = string_to_ind[gene_id]
                else:
                    print("error: gene id {} not found".format(gene_id))
        self.string_to_ind = string_to_ind
        self.ind_to_string = ind_to_string
        self.id_vec = id_vec
        self.num_genes = len(string_to_ind)


def ensembl_string_to_int(id_string):
    if id_string is None:
        return 0
    else:
        return int(id_string[4:])

def unpack_adjust_list(names_list):
    if names_list == []:
        return None
    else:
        return names_list[0]

# N.B. requires all entrez IDs to be in gene_index
def entrez_id_matrix_to_ind(entrez_mat, string_to_ind):
    entrez_ids_to_inds = {}
    # there is no entrez gene_id 0, so None -> 0 is fine.
    for entrez_string, our_ind in string_to_ind.items():
        if entrez_string != 'None' and entrez_string != '':
            entrez_id = int(entrez_string)
            entrez_ids_to_inds[entrez_id] = our_ind
    use_row = lambda i: entrez_mat[i,0] in entrez_ids_to_inds and entrez_mat[i,1] in entrez_ids_to_inds
    include_rows = np.array([use_row(i) for i in np.arange(entrez_mat.shape[0])])
    submat = entrez_mat[include_rows,:]
    get_ind = lambda x: entrez_ids_to_inds[x]
    vec_get_ind = np.vectorize(get_ind)
    ind_mat = vec_get_ind(submat)
    return ind_mat


def map_or_none(a_to_b_map, a):
    if (a in a_to_b_map):
        return(a_to_b_map[a])
    else:
        return(None)

def get_gene_ids_for_positions(chromosomes, positions):
    ensembl_to_entrez, entrez_to_ensemble = get_ensembl_mappings()
    data = EnsemblRelease(75) # 75 is the last to use GRCh37
    # torch.tensor can't contain strings
    ensembl_gene_names = np.array([unpack_adjust_list(data.gene_ids_at_locus(chrom.item(), loci.item())) for chrom, loci in zip(chromosomes, positions)])
    # convert ensembl IDs to entrez IDs
    entrez_gene_names = np.array([map_or_none(ensembl_to_entrez,i) for i in ensembl_gene_names])
    num_entrez = np.sum(entrez_gene_names != None)
    num_ensembl = np.sum(ensembl_gene_names != None)
    if (num_entrez != num_ensembl):
        print("warning: some ensembl gene IDs were not mapped to NCBI IDs ({} ensembl IDS, {} entrez IDs)".format(num_ensembl, num_entrez))
    return entrez_gene_names

# We don't want only the interactions between genes present in our gene_index, we want everything.
# This allows two-step interactions between things in the data set where the middle item is not present.
def get_all_gene_interactions(string_to_ind):
    '''Read gene interactions from the downloaded biogrid file.
    Returns an nx2 matrix (undirected, without duplicates).
    IDs in the matrix are Entrez gene IDs.'''
    #TODO distinguish different types of interaction

    biogrid_info = pd.read_csv(ukbb_data + biogrid_file, sep='\t', low_memory=False)
    human_info = biogrid_info[(biogrid_info['Organism Name Interactor A'] == "Homo sapiens") & (biogrid_info['Organism Name Interactor B'] == "Homo sapiens")]

    entrez_pairs = human_info[['Entrez Gene Interactor A', "Entrez Gene Interactor B"]].astype(int)
    entrez_mat = entrez_pairs.to_numpy()

    ind_mat = entrez_id_matrix_to_ind(entrez_mat, string_to_ind)
    return ind_mat
