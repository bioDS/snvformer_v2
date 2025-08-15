from trainer import *
import os
import h5py

def get_train_test_sets(args):
    train_inds, test_inds, unused_inds = get_train_test(args)
    pretrain_inds = np.sort(np.concatenate((train_inds, unused_inds)))
    train_set = StreamingDataset(args.h5_file, use_inds=pretrain_inds, augment_data=False)
    test_set = StreamingDataset(args.h5_file, use_inds=test_inds, augment_data=False)
    return train_set, test_set

def get_distribution_stats(args):
    dataset_name = args.h5_file.split('/')[-1].split('.')[0]
    if not os.path.exists("distributions"):
        os.mkdir("distributions")
    cache_file = f"distributions/{dataset_name}_snv_distribution.h5"
    try:
        h5_f = h5py.File(cache_file, 'r')
        snv_distribution = torch.tensor(h5_f["snv_distribution"][:])
        snv_ind_counts = torch.tensor(h5_f["snv_ind_counts"][:])
        total_seqs = h5_f["total_seqs"][0]
        mean_train_loss = h5_f["mean_train_loss"][0]
        mean_test_loss = h5_f["mean_test_loss"][0]
        h5_f.close()
    except:
        max_snv_classes = 64
        train_set, test_set = get_train_test_sets(args)
        phenos, positions, chromosomes, snvs = train_set[0]
        seq_len = len(snvs)
        snv_ind_counts = torch.zeros(seq_len, max_snv_classes)
        inds = torch.arange(seq_len)
        for phenos, positions, chromosomes, snvs in train_set:
            snv_ind_counts[inds, snvs.long()] += 1
        total_seqs = float(len(train_set))
        snv_distribution = snv_ind_counts / total_seqs
        loss_fn = torch.nn.CrossEntropyLoss()
        total_train_loss = 0.0
        # training set loss of mean distribution
        for phenos, positions, chromosomes, snvs in train_set:
            total_train_loss += loss_fn(snv_distribution, snvs.long())
        mean_train_loss = total_train_loss/len(train_set)
        # test set loss of mean distribution
        total_test_loss = 0.0
        for phenos, positions, chromosomes, snvs in test_set:
            total_test_loss += loss_fn(snv_distribution, snvs.long())
        mean_test_loss = total_test_loss/len(test_set)

        h5_f = h5py.File(cache_file, 'w')
        h5_snv_dist = h5_f.create_dataset("snv_distribution",
                                          snv_distribution.shape,
                                          dtype='float32',
                                          chunks=(1, snv_distribution.shape[1]))
        h5_snv_dist[:] = snv_distribution
        h5_snv_counts = h5_f.create_dataset("snv_ind_counts",
                                          snv_ind_counts.shape,
                                          dtype='int64',
                                          chunks=(1, snv_ind_counts.shape[1]))
        h5_snv_counts[:] = snv_ind_counts
        h5_f["total_seqs"] = [total_seqs]
        h5_f["mean_train_loss"] = [mean_train_loss]
        h5_f["mean_test_loss"] = [mean_test_loss]
        h5_f.close()
    return snv_distribution, snv_ind_counts, total_seqs, mean_train_loss, mean_test_loss

def main(args):
    snv_distribution, snv_ind_counts, total_seqs, mean_train_loss, mean_test_loss = get_distribution_stats(args)
    print(f"mean train loss: {mean_train_loss}")
    print(f"mean test loss: {mean_test_loss}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='simple distributed training job')
    parser.add_argument("--h5-file", default="/data/ukbb/net_input/gwas_ldprune_320.h5")
    parser.add_argument("--test-frac", type=float, default="0.3")
    args = parser.parse_args()

    main(args)