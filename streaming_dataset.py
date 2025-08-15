import torch
import h5py
import pandas
# import nvtx
import numpy as np
import torch.random
from torch.utils.data import Dataset
import ast

class StreamingDataset(Dataset):
    """
    Train/test split should be done by passing the appropriate set's indices to use_inds.

    Data Augmentation (if enabled):
        - Even indices iterate through non-gout cases
        - Odd indices produce randomly perturbed gout cases (or occaisionally untouched gout cases)
        - `augment_frac` is the fraction of the snv sequence to change.
    """
    def __init__(self,
                 h5_filename,
                 use_inds=None,
                 default_device='cpu',
                 augment_data=True,
                 augment_frac=0.15,
                 augment_mult=1.0,
                 ):
        h5_f = h5py.File(h5_filename, 'r')
        h5_dataset = h5_f['tok_mat']
        if use_inds is None:
            use_inds = np.arange(h5_dataset.shape[0])
        else:
            use_inds = np.sort(use_inds)
        self.phenos = pandas.read_hdf(h5_filename, key="phenos").iloc[use_inds,:]
        self.use_inds = use_inds
        self.h5_dataset = h5_dataset
        self.positions = torch.tensor(h5_f["positions"], device=default_device).broadcast_to(len(use_inds), -1)
        #self.gene_id_vec = torch.tensor(h5_f["gene_ids"], device=default_device).broadcast_to(len(use_inds), -1)
        self.chromosomes = torch.tensor(h5_f["chromosomes"], device=default_device).broadcast_to(len(use_inds), -1)
        self._full_tok_mat = None
        self.default_device = default_device
        self.augment_data = augment_data
        self.augmented_data_size = 0
        self.augment_frac = augment_frac
        self.string_to_tok = ast.literal_eval(h5_f["string_to_tok"][0].decode('UTF-8'))
        self.tok_to_string = ast.literal_eval(h5_f["tok_to_string"][0].decode('UTF-8'))
        self.class_tok = self.string_to_tok['cls']

        # find gout and non-gout cases
        # total size is 2*(number of non-gout cases)
        if self.augment_data:
            self.gout_inds = np.where(self.phenos["gout"] == True)[0]
            self.non_gout_inds = np.where(self.phenos["gout"] == False)[0]
            num_non_gout = len(self.non_gout_inds)
            num_gout = len(self.gout_inds)
            max_augmented_data_size = 2*num_non_gout
            self.augmented_data_size = min(max_augmented_data_size, round(2*(augment_mult*num_gout)))
            self.midpoint = (self.augmented_data_size // 2) - num_gout

    def _pheno_remove_nan(self, phenos):
        new_phenos = phenos.copy()
        if np.isnan(new_phenos["urate"]):
            new_phenos["urate"] = 0.0
        if np.isnan(new_phenos["bmi"]):
            new_phenos["bmi"] = 0.0
        if np.isnan(new_phenos["age"]):
            new_phenos["age"] = 0.0
        if np.isnan(new_phenos["height"]):
            new_phenos["height"] = 0.0
        return new_phenos

    @property
    def full_tok_mat(self):
        if self._full_tok_mat is None:
            self._full_tok_mat = torch.tensor(self.h5_dataset[self.use_inds, :],
                                              device=self.default_device)
        return self._full_tok_mat

    def __len__(self):
        if self.augment_data:
            return self.augmented_data_size
        else:
            return len(self.phenos)

    def _augmented_get_nongout(self, idx):
        """
        return non_gout case idx/2 as normal
        """
        nongout_idx = self.non_gout_inds[idx//2]
        return self._standard_getitem(nongout_idx)
        
    # @torch.compile
    def _swap_snvs(self, snvs):
        inds = np.random.choice(len(snvs), int(np.ceil(len(snvs) * self.augment_frac)), replace=False)
        new_values = torch.randint(0, 3, (len(inds),), dtype=torch.uint8)
        snvs[inds] = new_values
        return snvs

    def _mask_chunk(self, snvs):
        num_mask = int(np.ceil(len(snvs) * self.augment_frac))
        start_ind = np.random.randint(0, len(snvs)-num_mask)
        snvs[start_ind:start_ind+num_mask] = 0
        return snvs

    def _augmented_get_gout(self, idx):
        """
        for indices 0...[num_gout_cases-1], returns gout case directly
        fir indices [num_gout_cases]...[num_nongout*2], takes gout case (idx % num_gout_cases) and perturbs it as follows:
            The first half [num_gout_cases]...[(num_nongout*2 - num_gout_cases)/2] are perburbed as follows
                - 15% of snvs at random are masked
            The second half:
                - 15% of snvs at random are reassigned (to 0/1/2 with even probability)
            In both cases, phenotypes are perburbed as follows:
                - age has a 10% change of increasing or decreasing by 1 (even chance of increasing or decreasing)
                - sex stays the same
                - bmi += N(0,1)

        """
        num_gout = len(self.gout_inds)
        if idx//2 < len(self.gout_inds):
            return self._standard_getitem(self.gout_inds[idx//2])
        else:
            # get actual gout case to use
            use_gout_idx = idx % num_gout
            use_idx = self.gout_inds[use_gout_idx]
            phenos = self._pheno_remove_nan(self.phenos.iloc[use_idx,:].copy())
            positions = self.positions[use_idx,:].clone()
            chromosomes = self.chromosomes[use_idx,:].clone()
            snvs = torch.tensor(self.h5_dataset[self.use_inds[use_idx], :], device=self.default_device)
            if idx  < self.midpoint:
                # mask some snvs
                snvs = self._mask_chunk(snvs)
            else:
                # randomly modify some snvs
                snvs = self._swap_snvs(snvs)
                pass
            # phenotypes
            # 10% chance of changing age
            if torch.rand(1) < 0.1:
                if torch.randint(0, 2, (1,)) == 1:
                    phenos["age"] += 1
                else:
                    phenos["age"] -= 1
            # always tweak bmi
            phenos["bmi"] += torch.normal(torch.tensor([0.0]), torch.tensor([1.0]))
            return phenos, positions, chromosomes, snvs

    def _augmented_getitem(self, idx):
        """
        Indices are split mod 2 (even/odd)
            - 0 : non-gout.
            - 1 : gout.
        """
        if idx % 2 == 0:
            return self._augmented_get_nongout(idx)
        else:
            return self._augmented_get_gout(idx)

    def _standard_getitem(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        phenos = self._pheno_remove_nan(self.phenos.iloc[idx,:])
        positions = self.positions[idx,:].clone()
        chromosomes = self.chromosomes[idx,:].clone()
        snvs = torch.tensor(self.h5_dataset[self.use_inds[idx], :], device=self.default_device)
        return phenos, positions, chromosomes, snvs

    # @nvtx.annotate()
    def __getitem__(self, idx):
        if self.augment_data:
            return self._augmented_getitem(idx)
        else:
            return self._standard_getitem(idx)

def test_augmented_data():
    """
    only actually checks the gout/non-gout cases can be retrieved
    and are gout/non-gout as they should be
    """
    from trainer import get_train_test
    import argparse
    parser = argparse.ArgumentParser(description='simple distributed training job')
    parser.add_argument("--test-frac", type=float, default="0.3")
    parser.add_argument("--h5-file", default="/data/ukbb/net_input/all_gwas.h5")
    args = parser.parse_args()
    train_inds, test_inds, unused_inds = get_train_test(args)
    pretrain_inds = np.sort(np.concatenate((train_inds, unused_inds)))
    train_set = StreamingDataset(args.h5_file, use_inds=pretrain_inds, augment_data=True)
    # unmodified cases
    assert(train_set[0][0]["gout"] == False)
    assert(train_set[1][0]["gout"] == True)
    assert(train_set[540][0]["gout"] == False)
    assert(train_set[541][0]["gout"] == True)
    # non-gout limit
    assert(train_set[2*len(train_set.non_gout_inds)-2][0]["gout"] == False)
    # first and last block-zero'd gout cases
    assert(train_set[2*len(train_set.gout_inds)+1][0]["gout"] == True)
    midpoint = len(train_set.non_gout_inds) + (len(train_set.non_gout_inds) - len(train_set.gout_inds)) //2
    assert(train_set[2*midpoint-1][0]["gout"] == True)
    # first and last perturbed gout cases
    assert(train_set[2*midpoint+1][0]["gout"] == True)
    assert(train_set[len(train_set)-1][0]["gout"] == True)



# sanity test
def test_sanity():
    sd = StreamingDataset("/data/ukbb/net_input/all_gwas.h5")
    phenos, positions, chromosomes, snvs = sd[1:5]
    assert(phenos.shape == (4,7))
    assert(positions.shape == (4,13290))
    assert(chromosomes.shape == (4,13290))
    assert(snvs.shape == (4,13290))

def test_use_inds():
    sd = StreamingDataset("/data/ukbb/net_input/all_gwas.h5", use_inds=[1,5,20,35,125,64])
    phenos, positions, chromosomes, snvs = sd[0:5]
    assert(phenos.shape == (5,7))
    assert(positions.shape == (5,13290))
    assert(chromosomes.shape == (5,13290))
    assert(snvs.shape == (5,13290))
    assert(phenos.eid.iloc[0] == 1000022)
    assert(phenos.eid.iloc[1] == 1000063)
    count = 0
    for phenos, positions, chromosomes, snvs in sd:
        count += 1
    assert(count == 6)
