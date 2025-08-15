#!/usr/bin/env python
from process_snv_mat import get_tok_mat
from os.path import exists
import h5py
import argparse
from pandas_plink import read_plink1_bin
import pandas

# read a bed file and convert to to an hd5 file

parser = argparse.ArgumentParser(description='Read a bed file and convert to hdf5.')
parser.add_argument("--plink_base", default="all_gwas")
parser.add_argument("--plink_dir", default="/data/ukbb/net_input")
parser.add_argument("--pheno_file", default="/data/ukbb/net_input/phenos.csv")
parser.add_argument("--encoding", type=int, default="5")
parser.add_argument("--h5_file", default="/data/ukbb/net_input/all_gwas.h5")


args = parser.parse_args()
plink_base = args.plink_base
data_dir = args.plink_dir
pheno_file = args.pheno_file
encoding = args.encoding
h5_filename = args.h5_file

print("using data from:", data_dir)
bed_file = data_dir + '/' + plink_base + ".bed"
bim_file = data_dir + '/' + plink_base + ".bim"
fam_file = data_dir + '/' + plink_base + ".fam"
print("bed_file:", bed_file)
geno_tmp = read_plink1_bin(bed_file, bim_file, fam_file)
geno_tmp["sample"] = pandas.to_numeric(geno_tmp["sample"])
urate_tmp = pandas.read_csv(pheno_file)
withdrawn_ids = pandas.read_csv(data_dir + "/w12611_2023-08-21.csv",
                                header=None,
                                names=["ids"])

print("removing withdrawn ids")
usable_ids = list(set(urate_tmp.eid) - set(withdrawn_ids.ids))
phenos = urate_tmp[urate_tmp["eid"].isin(usable_ids)]
del urate_tmp
# avail_phenos = urate
geno = geno_tmp[geno_tmp["sample"].isin(usable_ids)]
del geno_tmp

tok_mat, tok_to_string, string_to_tok, num_toks = get_tok_mat(geno, encoding)
positions = geno.pos.values
chromosomes = geno.chrom.values.astype(int)

h5_f = h5py.File(h5_filename, 'w')
h5_tok_mat = h5_f.create_dataset("tok_mat",
                                    tok_mat.shape,
                                    dtype='uint8',
                                    chunks=(1, tok_mat.shape[1]))
h5_f["chromosomes"] = chromosomes
h5_f["positions"] = positions
h5_f["tok_to_string"] = [str(tok_to_string)]
h5_f["string_to_tok"] = [str(string_to_tok)]
h5_f["num_toks"] = [num_toks]
h5_f["male"] = phenos["sex"] == 'Male'

h5_tok_mat[:, :] = tok_mat
h5_f.close()
h5_pandas = pandas.HDFStore(h5_filename)
phenos.to_hdf(h5_pandas, "phenos")
h5_pandas.close()

def testing():
    from trainer import load_train_objs, get_default_argparser, collate_fn
    import torch
    from torch import nn
    import os
    import numpy as np
    from torch.utils.data import DataLoader
    from torch.utils.data.distributed import DistributedSampler
    from itertools import permutations, islice
    parser = get_default_argparser()
    args = parser.parse_args()
    args.h5_file = "/data/ukbb/net_input/tin_fixed_order.h5"
    model = nn.Linear(123,1)
    os.environ["LOCAL_RANK"] = "0"
    os.environ["RANK"] = "0"
    os.environ["WORLD_SIZE"] = "1"
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12356"
    torch.distributed.init_process_group(backend="nccl")
    dataset, mini_test_set, verify_set, optimizer, warmup_scheduler, main_scheduler = load_train_objs(args, model)

    data_dir="/data/ukbb/net_input"
    plink_base="tin_fixed_order"
    pheno_file="/data/ukbb/net_input/phenos.csv"
    bed_file = data_dir + '/' + plink_base + ".bed"
    bim_file = data_dir + '/' + plink_base + ".bim"
    fam_file = data_dir + '/' + plink_base + ".fam"
    geno_tmp = read_plink1_bin(bed_file, bim_file, fam_file)
    geno_tmp["sample"] = pandas.to_numeric(geno_tmp["sample"])
    urate_tmp = pandas.read_csv(pheno_file)
    withdrawn_ids = pandas.read_csv(data_dir + "/w12611_2023-08-21.csv",
                                    header=None,
                                    names=["ids"])
    print("removing withdrawn ids")
    usable_ids = list(set(urate_tmp.eid) - set(withdrawn_ids.ids))
    full_phenos = urate_tmp[urate_tmp["eid"].isin(usable_ids)]
    del urate_tmp
    # avail_phenos = urate
    geno = geno_tmp[geno_tmp["sample"].isin(usable_ids)]
    del geno_tmp
    tok_to_string = dataset.tok_to_string
    string_to_tok = dataset.string_to_tok

    ind=0
    test_group = dataset[ind]

    batch_size = 100
    shuffle=True
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        pin_memory=True,
        shuffle=False,
        sampler=DistributedSampler(dataset, shuffle=shuffle),
        collate_fn=collate_fn
    )

    test_group = nth(dataloader._get_iterator(), 2)

    def nth(iterable, n, default=None):
        "Returns the nth item or a default value"
        return next(islice(iterable, n, None), default)



    for ind, test_group in enumerate(dataloader):
        print(f"trying {ind}")
        phenos, positions, chromosomes, snvs = test_group
        eid = phenos.eid
        plink_seq = geno[geno.sample.isin(eid)]
        orig_pheno = full_phenos[full_phenos.eid.isin(eid)].copy()
        ind_pos = np.array([np.where(orig_pheno.eid == phenos.eid.iloc[i]) for i in range(batch_size)]).squeeze()
        assert(len(ind_pos) == batch_size)
        orig_pheno = orig_pheno.iloc[ind_pos,:]
        orig_pheno.loc[np.isnan(orig_pheno["urate"].to_numpy()),"urate"] = 0.0
        orig_pheno.loc[np.isnan(orig_pheno["age"].to_numpy()), "age"] = 0.0
        orig_pheno.loc[np.isnan(orig_pheno["bmi"].to_numpy()), "bmi"] = 0.0
        # orig_pheno.iloc[np.isnan(orig_pheno["sex"].to_numpy())].sex = 0.0
        orig_pheno.loc[np.isnan(orig_pheno["height"].to_numpy()), "height"] = 0.0
        # if np.isnan(orig_pheno["urate"]).any():
        #     orig_pheno.loc[:,"urate"] = 0.0
        # if np.isnan(orig_pheno["bmi"]).any():
        #     orig_pheno.loc[:,"bmi"] = 0.0
        # if np.isnan(orig_pheno["age"]).any():
        #     orig_pheno.loc[:,"age"] = 0.0
        # if np.isnan(orig_pheno["height"]).any():
        #     orig_pheno.loc[:,"height"] = 0.0
        assert(np.sum(phenos.to_numpy() != orig_pheno.to_numpy()) == 0)
        assert(np.sum(np.array(plink_seq.pos) != positions.numpy()) == 0)
        assert(np.sum(np.array(plink_seq.chrom, dtype=np.int32) != chromosomes.numpy()) == 0)
        f = lambda x: tok_to_string[x]
        f = np.vectorize(f)
        # enc_seq = np.array([tok_to_string[i] for i in snvs[0,].numpy()], dtype=np.float32)
        enc_seq = np.array(f(snvs), dtype=np.float32)

        plink_seq_v5 = plink_seq[ind_pos,:].to_numpy()
        assert(np.array_equal(enc_seq, plink_seq_v5, equal_nan=True))
        # diffpos = np.where(~np.array_equal(enc_seq, plink_seq_v5, equal_nan=True))
        # diffpos = np.where(plink_seq_v5 != enc_seq)
        # plink_seq_v5[diffpos]
        # enc_seq[diffpos]
