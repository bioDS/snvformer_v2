import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import nvtx
from enum import IntEnum
from gene_information import get_gene_ids_for_positions, get_all_gene_interactions
from gene_graph import GeneInteractionGraph

class InputTypes(IntEnum):
    pad = 0
    snv = 1
    pheno = 2
    class_tok = 3

class GeneIds(nn.Module):
    def __init__(self):
        super(GeneIds, self).__init__()
        self._positions = None
        self._chromosomes = None
        self._id_vec = None

    # N.B. assumes positions is fixed.
    @property
    def gene_id_vec(self):
        if self._id_vec is None:
            self._gene_id_strings = get_gene_ids_for_positions(self._chromosomes, self._positions)
            gene_ind_count = 1 # leave zero for the Nones
            seen_ids = {}
            ind_to_string = {0: "None"}
            id_list = []
            for gi in self._gene_id_strings:
                if gi == None:
                    id_list.append(0)
                if gi != None:
                    if gi not in seen_ids:
                        seen_ids[gi] = gene_ind_count
                        ind_to_string[gene_ind_count] = gi
                        gene_ind_count += 1
                    id_list.append(seen_ids[gi])
            self._int_to_string = ind_to_string
            self._string_to_ind = seen_ids
            self._id_vec = torch.tensor(id_list, dtype=torch.int32).to(self._chromosomes.device)
        return self._id_vec

    def forward(self, chromosomes, positions):
        # we could check if posittions == self.positions, and recalculate if necessary
        if self._positions == None:
            self._positions = positions[0,:]
        if self._chromosomes == None:
            self._chromosomes = chromosomes[0,:]
        return self.gene_id_vec.to(chromosomes.device)

class LazyGeneEmbedding(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.embed_dim = embed_dim
        self.gene_ids = GeneIds()
        self._embedding = None
        self._num_gene_ids = None
        self._device = None

    @property
    def embedding(self):
        if self._embedding == None:
            self._embedding = nn.Embedding(self._num_gene_ids, self.embed_dim).to(self._device)
        return self._embedding

    def forward(self, chromosomes, positions):
        self._device = chromosomes.device
        gene_ids = self.gene_ids(chromosomes, positions)
        if self._num_gene_ids == None:
            self._num_gene_ids = len(gene_ids)
        return self.embedding(gene_ids).broadcast_to(positions.shape[0], -1, -1).to(positions.device)

class LazyGeneGraphEmbedding(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.embed_dim = embed_dim
        self.gene_ids = GeneIds()
        self._num_gene_ids = None
        self._interactions = None
        self._gene_graph = None
        self._num_genes_in_graph = None
        self._unique_ids = None
        self._device = None

    @property
    def gene_graph(self):
        if self._gene_graph == None:
            self._gene_graph = GeneInteractionGraph(self.embed_dim, self.embed_dim, self.gene_ids._int_to_string, self.interactions).to(self._device)
        return self._gene_graph

    # N.B. must be run **after** self.gene_ids is defined
    @property
    def interactions(self):
        if self._interactions == None:
            self._interactions = get_all_gene_interactions(self.gene_ids._string_to_ind)
        return self._interactions

    def forward(self, chromosomes, positions):
        self._device = chromosomes.device
        gene_ids = self.gene_ids(chromosomes, positions)
        if self._num_gene_ids == None:
            self._num_gene_ids = len(gene_ids)
        unique_ids = torch.unique(gene_ids).to(chromosomes.device)
        ggo_for_id = self.gene_graph(unique_ids)
        embedded_gene_vec = ggo_for_id[gene_ids].broadcast_to(positions.shape[0], -1, -1).to(positions.device)
        return embedded_gene_vec


# works out sin/cosine situation on-the-fly rather than storing P
class OTFPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        super(OTFPositionalEncoding, self).__init__()
        self.denoms = torch.pow(10000, torch.arange(0, d_model, 2) / d_model)
        self.d_model = d_model

    def forward(self, positions):
        denoms = self.denoms.to(positions.device)
        positions = positions.unsqueeze(2).broadcast_to((-1, -1, self.d_model))
        enc = torch.zeros_like(positions, dtype=torch.float32).to(positions.device)
        enc[:, :, 0::2] = torch.sin(positions[:, :, 0::2] / denoms)
        enc[:, :, 1::2] = torch.cos(positions[:, :, 1::2] / denoms)
        return enc

class ExplicitPositionalEncoding(nn.Module):
    """Positional encoding."""
    # def __init__(self, num_hiddens, dropout, max_len=1000):
    def __init__(self, d_model: int, dropout: float = 0.0, max_len: int = 5000):
        super(ExplicitPositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(dropout)
        # Create a long enough `P`
        self.P = torch.zeros((1, max_len, d_model))
        X = torch.arange(max_len, dtype=torch.float32).reshape(
            -1, 1) / torch.pow(10000, torch.arange(
            0, d_model, 2, dtype=torch.float32) / d_model)
        self.P[:, :, 0::2] = torch.sin(X)
        self.P[:, :, 1::2] = torch.cos(X)

    def forward(self, positions):
        # X_old = X
        # P = self.P.repeat(len(X), 1, 1)
        X = self.P[:, positions[0,:].cpu(), :].to(positions.device)
        return self.dropout(X)

def test_otf_encoding():
    d_model = 64
    max_len = 1000
    otf_enc = OTFPositionalEncoding(d_model=d_model, max_len=max_len)
    exp_enc = ExplicitPositionalEncoding(d_model=d_model, max_len=max_len, dropout = 0.0)

    positions = torch.arange(max_len).unsqueeze(0)
    otf_pos = otf_enc(positions)
    exp_pos = exp_enc(positions)
    assert(torch.equal(otf_pos, exp_pos))

class LearnablePosEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        super(LearnablePosEncoding, self).__init__()
        # initialise to sin/cos encoding
        denoms = torch.pow(10000, torch.arange(0, d_model, 2) / d_model)
        positions = torch.arange(max_len)
        positions = positions.unsqueeze(1).broadcast_to((-1, d_model))
        enc = torch.zeros_like(positions, dtype=torch.float32)
        enc[:, 0::2] = torch.sin(positions[:, 0::2] / denoms)
        enc[:, 1::2] = torch.cos(positions[:, 1::2] / denoms)
        enc = enc.unsqueeze(0)
        self.pos_embedding = nn.Parameter(enc)


    def forward(self, positions):
        enc = self.pos_embedding[:, :positions.shape[1], :]
        return enc

class Vectorizer(nn.Module):

    def __init__(self,
                 seq_len,
                 embed_dim,
                 num_snv_classes,
                 use_phenos,
                 position_encoding = "fixed",
                 snv_encoding = "embedding",
                 pos_combine = "add",
                 ignore_chrom = False,
                 ignore_class = False,
                 dropout = 0.2,
                 snv_embed_size = None,
                 gene_embed_size = None,
                 use_gene_graph = False,
                 chrom_embed_size = None,
                 pos_embed_size = None,
                 class_tok=1) -> None:
        super().__init__()
        # only works with encoding 5 for the moment (0,1,2)
        self.class_tok = 1
        self.dropout = nn.Dropout(dropout)
        self.total_embed_dim = embed_dim
        self.num_snv_classes = num_snv_classes
        self.use_phenos = use_phenos
        self.use_output_resnet = False
        self.snv_encoding = snv_encoding
        self.ignore_chrom = ignore_chrom
        self.ignore_class = ignore_class
        if gene_embed_size != None and gene_embed_size > 0:
            self.gene_embed_size = gene_embed_size
            if use_gene_graph:
                self.gene_embedding = LazyGeneGraphEmbedding(gene_embed_size)
            else:
                self.gene_embedding = LazyGeneEmbedding(gene_embed_size)
        else:
            self.gene_embed_size = 0
            self.gene_embedding = self._ignore
        if pos_embed_size != None:
            self.pos_embed_dim = pos_embed_size
        if self.ignore_chrom:
            if chrom_embed_size == None:
                chrom_embed_size = 0
            self.chrom_embedding = self._ignore
        else:
            if chrom_embed_size == None:
                chrom_embed_size = 4
        if self.ignore_class:
            self.class_embed_size = 0
        else:
            self.class_embed_size = len(InputTypes)
        if pos_combine == "add":
            self.pos_combine = "add"
            if pos_embed_size == None:
                self.pos_embed_dim = embed_dim
            if snv_embed_size == None:
                snv_embed_size = self.total_embed_dim - self.class_embed_size - chrom_embed_size - self.gene_embed_size
        elif pos_combine == "cat":
            self.pos_combine = "cat"
            if pos_embed_size == None:
                self.pos_embed_dim = embed_dim // 2
            if snv_embed_size == None:
                snv_embed_size = self.total_embed_dim - self.class_embed_size - chrom_embed_size - self.pos_embed_dim - self.gene_embed_size
        else:
            raise ValueError("pos_combine must be either 'add' or 'cat'")
        if self.snv_encoding == "one-hot":
            self._encode_snvs = self._encode_snvs_one_hot
            if not self.ignore_chrom:
                self.chrom_embedding = self._encode_chrom_one_hot
        elif self.snv_encoding == "embedding":
            self.snv_embedding = nn.Embedding(self.num_snv_classes, snv_embed_size)
            self._encode_snvs = self._encode_snvs_embedding
            if not self.ignore_chrom:
                self.chrom_embedding = nn.Embedding(23, chrom_embed_size)
        else:
            raise ValueError("snv_encoding must be either 'embedding' or 'one-hot'")
        if position_encoding == "fixed":
            self.position_enc = ExplicitPositionalEncoding(self.pos_embed_dim, max_len=(249222527 + 1))
        elif position_encoding == "otf":
            self.position_enc = OTFPositionalEncoding(self.pos_embed_dim, max_len=(249222527 + 1))
        elif position_encoding == "embedding":
            self.position_enc = LearnablePosEncoding(self.pos_embed_dim, max_len=seq_len)
        else:
            raise ValueError("position_encoding must be one of 'fixed', 'otf', or 'embedding'")
        self.snv_embed_size = snv_embed_size

    def _ignore(self, something, *args):
        return torch.tensor([]).to(something.device)

    def _encode_chrom_one_hot(self, chromosomes):
        one_hot_chrom = F.one_hot(chromosomes.long() - 1, num_classes=22) 
        return one_hot_chrom

    def _encode_snvs_embedding(self, positions, chromosomes, snvs):
        seq_len = snvs.shape[1]
        # one-hot snv embedding
        if self.ignore_class:
            snv_class = torch.tensor([]).to(snvs.device)
        else:
            snv_class = torch.broadcast_to(
                F.one_hot(torch.tensor([int(InputTypes.snv)]), num_classes = self.class_embed_size), 
                (seq_len, self.class_embed_size)
            ).unsqueeze(0).broadcast_to(snvs.shape[0], -1, -1).to(snvs.device)
        embed_snvs = self.snv_embedding(snvs.long())
        embed_chrom = self.chrom_embedding(chromosomes.long())
        embed_gene = self.gene_embedding(chromosomes, positions)
        # one-hot chromosome embedding
        # chromosome inds start at 1, we need to adjust them.
        # sin/cos position embedding
        pos_enc = self.position_enc(positions).broadcast_to(snvs.shape[0], -1, -1)
        # [one_hot_snvs][one_hot_crom][pad] .+ [pos_enc]
        if self.pos_combine == "add":
            snv_encoded_vec = torch.cat((snv_class, embed_snvs, embed_gene, embed_chrom), dim=2) + pos_enc
        else:
            snv_encoded_vec = torch.cat((snv_class, embed_snvs, embed_gene, embed_chrom, pos_enc), dim=2)
        return snv_encoded_vec

    def _encode_snvs_one_hot(self, positions, chromosomes, snvs):
        # snv embedding is: [one-hot type (class/pheno/snv)][one-hot 0/1/2][one-hot chrom][pad] .+ [position]
        seq_len = snvs.shape[1]
        # one-hot snv embedding
        if self.ignore_class:
            snv_class = torch.tensor([])
        else:
            snv_class = torch.broadcast_to(
                F.one_hot(torch.tensor([int(InputTypes.snv)]), num_classes = self.class_embed_size), 
                (seq_len, self.class_embed_size)
            ).unsqueeze(0).broadcast_to(snvs.shape[0], -1, -1).to(snvs.device)
        one_hot_snvs = F.one_hot(snvs.long(), num_classes=self.num_snv_classes)
        # one-hot chromosome embedding
        # chromosome inds start at 1, we need to adjust them.
        one_hot_chrom = F.one_hot(chromosomes.long() - 1, num_classes=22) 
        # sin/cos position embedding
        pos_enc = self.position_enc(positions)
        # [one_hot_snvs][one_hot_crom][pad] .+ [pos_enc]
        pad_size = self.total_embed_dim - (snv_class.shape[-1] + one_hot_chrom.shape[-1] + one_hot_snvs.shape[-1])
        if self.pos_combine == "add":
            snv_encoded_vec = F.pad(torch.cat((snv_class, one_hot_snvs, one_hot_chrom), dim=2), (0,pad_size)) + pos_enc
        else:
            snv_encoded_vec = F.pad(torch.cat((snv_class, one_hot_snvs, one_hot_chrom, pos_enc), dim=2), (0,pad_size))
        return snv_encoded_vec

    def _encode_class(self, batch_size, use_device):
        # class embedding is [one-hot type][padding]
        # workaround classes being generally excluded
        # if (self.class_embed_size == 0):
        #     class_embed_size = len(InputTypes)
        # else:
        #     class_embed_size = self.class_embed_size
        # class_vec = F.pad(
        #     F.one_hot(torch.tensor([int(InputTypes.class_tok)]), num_classes = class_embed_size),
        #     (0,self.total_embed_dim-class_embed_size)
        # )
        class_vec = F.pad(
            self.snv_embedding(torch.tensor([int(InputTypes.class_tok)], device=use_device)),
            (0,self.total_embed_dim-self.snv_embed_size)
        )
        class_vec = torch.unsqueeze(class_vec, 0)
        class_vec = class_vec.repeat(batch_size, 1, 1)
        return class_vec

    def _encode_phenos(self, use_phenos):
        # pheno_embedding is [one-hot type][one-hot pheno ind][pheno_val][padding]
        if len(self.use_phenos) == 0:
            return torch.tensor([])
        batch_size = len(use_phenos)

        if self.ignore_class:
            batch_pheno_class_vec = torch.tensor([])
        else:
            pheno_class_vec = torch.broadcast_to(
                F.one_hot(torch.tensor([int(InputTypes.pheno)]), num_classes = self.class_embed_size),
                (use_phenos.shape[1],self.class_embed_size)
            )
            batch_pheno_class_vec = pheno_class_vec.repeat(batch_size, 1, 1)
        #TODO: this is a good thing, you should p
        # pheno_ind_vec = F.one_hot(
        #     torch.arange(0,use_phenos.shape[1])
        # )
        # batch_pheno_ind_vec = pheno_ind_vec.repeat(batch_size, 1, 1)
        pheno_vecs = []
        if ("age" in use_phenos):
            age_vec = torch.tensor(use_phenos["age"].to_numpy(dtype=np.float32), dtype=torch.float32).unsqueeze(1).unsqueeze(1).expand(-1,-1,self.total_embed_dim)
            pheno_vecs.append(age_vec)
        if ("sex" in use_phenos):
            sex_vec = torch.tensor((use_phenos["sex"] == "Female").to_numpy(), dtype=torch.float32).unsqueeze(1).unsqueeze(1).expand(-1,-1,self.total_embed_dim)
            pheno_vecs.append(sex_vec)
        if ("bmi" in use_phenos):
            bmi_vec = torch.tensor(use_phenos["bmi"].to_numpy(dtype=np.float32), dtype=torch.float32).unsqueeze(1).unsqueeze(1).expand(-1,-1,self.total_embed_dim)
            pheno_vecs.append(bmi_vec)
        if ("urate" in use_phenos):
            urate_vec = torch.tensor(use_phenos["urate"].to_numpy(dtype=np.float32), dtype=torch.float32).unsqueeze(1).unsqueeze(1).expand(-1,-1,self.total_embed_dim)
            pheno_vecs.append(urate_vec)
        if ("height" in use_phenos):
            height_vec = torch.tensor(use_phenos["height"].to_numpy(dtype=np.float32), dtype=torch.float32).unsqueeze(1).unsqueeze(1).expand(-1,-1,self.total_embed_dim)
            pheno_vecs.append(height_vec)
        #batch_pheno_val_vec = torch.unsqueeze(torch.swapdims(torch.stack((age_vec, sex_vec, bmi_vec)), 0, 1), 2)
        # batch_pheno_val_vec = torch.unsqueeze(torch.swapdims(torch.stack(pheno_vecs), 0, 1), 2)
        batch_pheno_val_vec = torch.cat(pheno_vecs, dim=1)
        # pheno_vec = torch.cat((batch_pheno_class_vec, batch_pheno_ind_vec, batch_pheno_val_vec), dim=2)
        # padded_pheno_vec = F.pad(pheno_vec, (0,self.total_embed_dim - pheno_vec.shape[2]))
        # return padded_pheno_vec
        return batch_pheno_val_vec

    @nvtx.annotate("vectorizer-forward")
    def forward(self, phenos, positions, chromosomes, snvs, max_len):
        use_device = snvs.device
        snv_encoded_vec = self._encode_snvs(positions, chromosomes, snvs)
        class_vec = self._encode_class(len(phenos), use_device) # this one we keep, it's not the snv/tok/pheno 'class' vec we might ignore
        #use_pheno_vals = ['age','sex','bmi']
        use_phenos = phenos[self.use_phenos]
        phenos_vec = self._encode_phenos(use_phenos)
        combined_vec = torch.cat((class_vec.to(use_device),
                                phenos_vec.to(use_device),
                                snv_encoded_vec.to(use_device)), dim=1)
        return self.dropout(combined_vec)


def test_vectorizer():
    from streaming_dataset import StreamingDataset
    dataset = StreamingDataset("/data/ukbb/net_input/all_gwas.h5")
    phenos, positions, chromosomes, snvs = dataset[0:2]
    vec = Vectorizer(64, 3, use_phenos=['age','sex','bmi'])
    tmp2 = vec(phenos, positions, chromosomes, snvs)
    assert(tmp2.shape == (2, 13290, 64))