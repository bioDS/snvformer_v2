#!/usr/bin/env python3
import torch
from torch import tensor, nn
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv # not necessarily the best/right architecture
import sys, os
from gene_information import GeneIndex

class GCN(nn.Module):
    def __init__(self, num_node_features, num_node_outputs, hidden_dim):
        super().__init__()
        self.conv1 = GCNConv(num_node_features, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, num_node_outputs)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = self.conv1(x, edge_index)
        x = nn.functional.relu(x)
        x = nn.functional.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        # return nn.functional.log_softmax(x, dim=1) # this is for classification of nodes
        return x

class GeneInteractionGraph(nn.Module):
    def __init__(self, gene_embed_dim, hidden_dim, ind_to_string, gene_interactions):
        '''gene_interactions is a tensor of [genes, 2] containing every pair of interacting genes'''
        super().__init__()
        self.gcn = GCN(gene_embed_dim, gene_embed_dim, hidden_dim)
        self.ind_to_string = ind_to_string
        gene_edge_index = tensor([[[a,b], [b,a]] for a,b in gene_interactions]).flatten(end_dim=1)
        gene_edge_index = gene_edge_index.t().contiguous()
        self.gene_edge_index = gene_edge_index
        self.num_genes = len(self.ind_to_string)
        self.gene_embed_dim = gene_embed_dim

        self.gene_embedding = nn.Embedding(len(self.ind_to_string), gene_embed_dim)

    def forward(self, gene_ind_vec):
        '''Runs a GCN with one node per gene, inputting the gene-embedding as the gene features.
        duplicates the embedded outputs in the shape of the snp-gene vector'''

        # contains an embedded entry for gene w/ index i in position i
        input_gene_embeddings = self.gene_embedding(torch.tensor(range(self.num_genes)).to(gene_ind_vec.device))
        gene_graph_data = Data(x=input_gene_embeddings, edge_index=self.gene_edge_index)
        output_gene_embeddings = self.gcn(gene_graph_data.to(gene_ind_vec.device))
        # output_vec = torch.zeros(gene_ind_vec.shape[0], gene_ind_vec.shape[1], self.gene_embed_dim).to(gene_ind_vec.device)
        # for count, gene_ind in enumerate(gene_ind_vec):
        #     output_vec[count,:] = output_gene_embeddings[gene_ind,:]
        return output_gene_embeddings


def main():
    edge_index = torch.tensor([[0, 1],
                            [1, 0],
                            [1, 2],
                            [2, 1]], dtype=torch.long)
    x = torch.tensor([[-1], [0], [1]], dtype=torch.float)

    data = Data(x=x, edge_index=edge_index.t().contiguous())

def gene_test():
    gene_embed_dim = 5
    gene_string_list = ['None', 'ENSG00000132906', 'ENSG00000090273', 'ENSG00000175707', 'ENSG00000131795', 'ENSG00000211451', 'ENSG00000198483', 'ENSG00000121848', 'ENSG00000174827', 'ENSG00000234283', 'ENSG00000143412', 'ENSG00000160685', 'ENSG00000163354', 'ENSG00000116580', 'ENSG00000184144', 'ENSG00000229258', 'ENSG00000227630', 'ENSG00000163793', 'ENSG00000163798', 'ENSG00000158019', 'ENSG00000115760', 'ENSG00000115977', 'ENSG00000116039', 'ENSG00000189223', 'ENSG00000115112', 'ENSG00000115947', 'ENSG00000081479', 'ENSG00000237380', 'ENSG00000224189', 'ENSG00000115252', 'ENSG00000021826', 'ENSG00000178568', 'ENSG00000115677', 'ENSG00000006607', 'ENSG00000132170', 'ENSG00000223552', 'ENSG00000088727', 'ENSG00000164091', 'ENSG00000010322', 'ENSG00000163939', 'ENSG00000163935', 'ENSG00000144747', 'ENSG00000163406', 'ENSG00000250934', 'ENSG00000250836', 'ENSG00000138246', 'ENSG00000114126', 'ENSG00000085276', 'ENSG00000163581', 'ENSG00000226350', 'ENSG00000188438', 'ENSG00000251087', 'ENSG00000249563', 'ENSG00000109667', 'ENSG00000071127', 'ENSG00000109684', 'ENSG00000249001', 'ENSG00000118777', 'ENSG00000246375', 'ENSG00000138642', 'ENSG00000138646', 'ENSG00000138640', 'ENSG00000138821', 'ENSG00000138735', 'ENSG00000185305', 'ENSG00000225940', 'ENSG00000251599', 'ENSG00000164199', 'ENSG00000113742', 'ENSG00000169220', 'ENSG00000131183', 'ENSG00000112699', 'ENSG00000124782', 'ENSG00000079691', 'ENSG00000124564', 'ENSG00000180573', 'ENSG00000214801', 'ENSG00000206341', 'ENSG00000206503', 'ENSG00000228688', 'ENSG00000237686', 'ENSG00000111912', 'ENSG00000233082', 'ENSG00000205356', 'ENSG00000198556', 'ENSG00000106436', 'ENSG00000253471', 'ENSG00000157168', 'ENSG00000156170', 'ENSG00000086062', 'ENSG00000233554', 'ENSG00000167103', 'ENSG00000107611', 'ENSG00000148584', 'ENSG00000165449', 'ENSG00000227877', 'ENSG00000165476', 'ENSG00000122376', 'ENSG00000068383', 'ENSG00000129965', 'ENSG00000133812', 'ENSG00000066382', 'ENSG00000134824', 'ENSG00000173153', 'ENSG00000168071', 'ENSG00000168065', 'ENSG00000197891', 'ENSG00000110076', 'ENSG00000168066', 'ENSG00000133895', 'ENSG00000149809', 'ENSG00000197847', 'ENSG00000173327', 'ENSG00000175467', 'ENSG00000175115', 'ENSG00000235718', 'ENSG00000118971', 'ENSG00000250931', 'ENSG00000179912', 'ENSG00000155980', 'ENSG00000074527', 'ENSG00000135100', 'ENSG00000175727', 'ENSG00000112787', 'ENSG00000165659', 'ENSG00000102595', 'ENSG00000100490', 'ENSG00000140057', 'ENSG00000080824', 'ENSG00000186031', 'ENSG00000159322', 'ENSG00000169410', 'ENSG00000169752', 'ENSG00000169758', 'ENSG00000140386', 'ENSG00000140443', 'ENSG00000140750', 'ENSG00000140718', 'ENSG00000102908', 'ENSG00000198373', 'ENSG00000204991', 'ENSG00000183914', 'ENSG00000178852', 'ENSG00000108924', 'ENSG00000154262', 'ENSG00000171105', 'ENSG00000173809', 'ENSG00000130203', 'ENSG00000101076', 'ENSG00000156284', 'ENSG00000230479', 'ENSG00000100344']
    gene_index = GeneIndex(gene_string_list)
    num_genes = len(gene_index.string_to_ind)
    gene_embedding = nn.Embedding(num_genes, gene_embed_dim)
    protein_x_raw = torch.tensor(range(num_genes))
    protein_x_embed = gene_embedding(protein_x_raw)
    gcn_model = GCN(gene_embed_dim, gene_embed_dim, 16)
    protein_interactions = [
        [2,5],
        [77,78],
        [2,6],
        [12,7],
        [32,33],
        [1,90],
        [15,30]
    ]
    protein_edge_index = tensor([[[a,b], [b,a]] for a,b in protein_interactions]).flatten(end_dim=1)
    protein_edge_index = protein_edge_index.t().contiguous()
    print(protein_edge_index)

    protein_graph_data = Data(x=protein_x_embed, edge_index=protein_edge_index)

    output = gcn_model(protein_graph_data)

    print(f'input embedded gene 0: {protein_x_embed[0]}')
    print(f'output embedded gene 0: {output[0]}')

if __name__ == "__main__":
    gene_test()
