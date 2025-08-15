from trainer import *
import os
import h5py
import seaborn as sns
import pandas
import matplotlib.pyplot as plt

def get_train_test_sets(args):
    train_inds, test_inds, unused_inds = get_train_test(args)
    pretrain_inds = np.sort(np.concatenate((train_inds, unused_inds)))
    train_set = StreamingDataset(args.h5_file, use_inds=pretrain_inds, augment_data=False)
    test_set = StreamingDataset(args.h5_file, use_inds=test_inds, augment_data=False)
    return train_set, test_set

def plot_correlation(args):
    dataset_name = args.h5_file.split('/')[-1].split('.')[0]
    if not os.path.exists("distributions"):
        os.mkdir("distributions")
    print("setting up data")
    cache_file = f"distributions/{dataset_name}_snv_correlation_distribution.h5"
    train_set, test_set = get_train_test_sets(args)

    full_mat = train_set.h5_dataset
    num_cols = args.num_col_samples
    num_rows = args.num_row_samples
    sample_cols = np.random.choice(full_mat.shape[1], num_cols, replace=False)
    sample_cols.sort()
    sample_rows = np.random.choice(full_mat.shape[0], num_rows, replace=False)
    sample_rows.sort()
    snv_mat = train_set.h5_dataset[sample_rows,:][:,sample_cols]

    print("finding correlations")
    snv_npmat = np.array(snv_mat)
    snv_corrmat = np.corrcoef(snv_npmat.transpose())
    print("summarising data")
    corr_sums = np.zeros(2*num_cols-1)
    corr_sums[len(corr_sums) - snv_corrmat.shape[1]:] = snv_corrmat[0,:]
    for i in range(1,snv_corrmat.shape[0]):
        corr_sums[len(corr_sums) - snv_corrmat.shape[1]-i:-i] += abs(snv_corrmat[i,:])
    
    mean_corr = corr_sums / num_rows
    # exclude the diagonal
    tmp5 = mean_corr[0:num_cols-1]
    tmp6 = mean_corr[num_cols:]
    no_middle = np.concatenate((mean_corr[0:num_cols-1],mean_corr[num_cols:]))

    col_diff_rounded = (np.concatenate((np.arange(-(num_cols-1),0), np.arange(1,num_cols))) // args.batch_size) * args.batch_size
    corr_df_rounded = pandas.DataFrame({"correlation": no_middle, "position": col_diff_rounded})
    print("plotting")
    plot = sns.lineplot(data=corr_df_rounded, x="position", y="correlation", estimator="mean")
    plot.get_figure().savefig(args.fig_file)
    plt.clf()

def main(args):
    plot_correlation(args)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='simple distributed training job')
    parser.add_argument("--h5-file", default="/data/ukbb/net_input/all_unimputed_combined.h5")
    parser.add_argument("--test-frac", type=float, default="0.3")
    parser.add_argument("--num-row-samples", type=int, default=10000)
    parser.add_argument("--num-col-samples", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=50)
    parser.add_argument("--fig-file", default="snv_correlation.svg")
    args = parser.parse_args()

    main(args)