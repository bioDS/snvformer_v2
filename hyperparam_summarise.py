from ray import tune, train
from ray.air import session
from ray.train import Checkpoint
from ray.tune.schedulers import ASHAScheduler
# from ray.util.sgd.torch import is_distributed_trainable
# from ray.util.sgd.torch import distributed_checkpoint_dir
import tempfile
import sys

from trainer import *
import os
from encoder import *
from hyena_encoder import get_model
from packaging import version
from contextlib import nullcontext

import vectorizer, trainer, encoder, gene_graph, gene_information
import importlib
for m in [vectorizer, trainer, encoder, gene_graph, gene_information]:
    importlib.reload(m)
from trainer import *
from encoder import *
import pickle
import glob
import matplotlib.pyplot as plt
import seaborn as sns
import pprint

from hyperparam_search import train_cifar

if __name__ == "__main__":
    sns.set_theme(style="whitegrid")
    pp = pprint.PrettyPrinter()
    path = "/home/kieran_elmes/gout_transformer_v2/ray_results/fault_tolerant_run/"

    trainable_with_resources = tune.with_resources(partial(train_cifar), {"cpu": 4, "gpu": 1})
    restored_tuner = tune.Tuner.restore(path, trainable = trainable_with_resources)

    result_grid = restored_tuner.get_results()
    num_results = len(result_grid)
    print("Number of results:", num_results)

    # print best loss for each trial
    results_df = result_grid.get_dataframe(
        filter_metric="loss", filter_mode="min"
    )
    with pandas.option_context('display.max_rows', None, 'display.max_columns', None):
        print(results_df.loc[:, ["training_iteration", "time_total_s", "loss"]])
    
    # print best result and best result's iter losses
    best_result = result_grid.get_best_result(metric="loss", mode="min", scope="all")
    best_result_metrics = best_result.metrics_dataframe
    print("best result config: ")
    pp.pprint(best_result.config)
    ## iter losses
    print(best_result_metrics.loc[:,['training_iteration', 'loss']])
    best_iter_num = np.argmin(best_result_metrics["loss"])
    print(f"best iter was {best_iter_num+1}, with loss {best_result_metrics.loc[best_iter_num-1,'loss']}") # names in table are +1, but paths start from 0.

    # best checkpoint
    best_checkpoint = best_result.get_best_checkpoint(metric="loss", mode="min")
    best_checkpoint_path = best_checkpoint.path + "/snapshot.pt"
    print(f"best checkpoint at: {best_checkpoint_path}")

    # plot test curve for best trial
    trial_test_curve_df = best_result_metrics.loc[:,["training_iteration", "loss"]]
    plt.clf()
    sns.lineplot(trial_test_curve_df, x="training_iteration", y="loss")
    plt.ylabel("Test Loss")
    plt.xlabel("Training Epoch")
    plt.savefig("figures/best_trial_test_curve.png")
    plt.savefig("figures/best_trial_test_curve.pdf")
    



    print("plotting metrics")
    #ax = None
    #for ind, result in enumerate(result_grid):
    #    label = f"trial {ind}" 
    #    if ax is None:
    #        ax = result.metrics_dataframe.plot("training_iteration", "loss")
    #    else:
    #        result.metrics_dataframe.plot("training_iteration", "loss", ax=ax)
    #ax.get_legend().remove()
    #ax.set_title("Mean Accuracy vs. Training Iteration for All Trials")
    #ax.set_ylabel("Mean Test Accuracy")
    #ax.set_yscale("log")
    #plt.savefig("figures/hyperparam_acc_vs_iter.pdf")
    #plt.savefig("figures/hyperparam_acc_vs_iter.png")


    plot_config_names = {
        "config/lr": "Learning Rate",
        "config/num_layers": "No. Layers",
        "config/batch_size":         "Batch Size",
        "config/num_heads":          "Number of Encoder Heads",
        "config/linformer_k":        "Linformer k",
        "config/no_gradscaler":      "No Gradscaler",
        "config/dropout":            "Dropout",
        "config/position_encoding":  "Position Encoding",
        "config/snv_encoding":       "SNV Encoding",
        "config/ignore_chrom":       "Chromosome Ignored",
        "config/ignore_class":       "Class Ignored",
        "config/pos_combine":        "Position Combination Method",
        "config/torch_sdp":          "Torch SDP",
        "config/ffn_scale":          "FFN Scale",
        "config/pos_embed_size":     "Position Embedding Size",
        "config/snv_embed_size":     "SNV Embedding Size",
        "config/chrom_embed_size":   "Chromosome Embedding Size",
        "config/gene_embed_size":    "Gene Embedding Size",
        "config/gene_embed_graph":   "Gene Embedding Graph",
        "config/tf_init":            "GPT Initialisation Used",
        "config/model_type":         "Output Type",
        # "config/no_gradscale":       "no gradscale",
    }

    for cf, name in plot_config_names.items():
        # plot variation in learning rate vs. best loss
        print(f"plotting {cf} vs. loss")
        plt.clf()
        plt.figure(figsize=(4,4))
        ax = plt.gca()
        if cf == "config/lr" or cf == "config/dropout":
            sns.scatterplot(results_df, x=cf, y="loss")
            plt.xscale("log")
        else:
            sns.boxplot(results_df, x=cf, y="loss")
        ax.set_xlabel(name)
        ax.set_ylabel("Log Loss")
        plt.tight_layout()
        figname = f'figures/{cf.replace("/", "_")}_vs_loss'
        plt.savefig(figname + ".pdf")
        plt.savefig(figname + ".svg")
        plt.savefig(figname + ".png")
        plt.close()