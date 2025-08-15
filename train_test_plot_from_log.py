import re
import sys
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import seaborn as sns
import torch
from sklearn import metrics

sns.set_theme(style="whitegrid")

filename = sys.argv[1]
fig_prefix = sys.argv[2]
train_test_fig_file = f"{fig_prefix}_tt_loss.pdf"
roc_fig_file = f"{fig_prefix}_roc.pdf"

if re.match(".*.pt", filename):
    snapshot = torch.load(filename, map_location="cpu")
    epochs_run = snapshot["EPOCHS_RUN"]
    program_args = snapshot["PROGRAM_ARGS"]
    tiny_test_losses        = snapshot["TINY_TEST_LOSSES"]
    train_losses            = np.array([i.detach().numpy() for i in snapshot["BATCH_LOSSES"]])
    epoch_mean_batch_losses = snapshot["EPOCH_MEAN_BATCH_LOSSES"]
    epoch_mini_test_losses  = snapshot["EPOCH_MINI_TEST_LOSSES"]
    try:
        if snapshot["TEST_SUMMARY"] != None:
            test_summary = snapshot["TEST_SUMMARY"]
        else:
            test_summary = None
    except KeyError:
        test_summary = None
    del snapshot
else:
    test_summary = None
    # filename = "output/linformer_learnedpos_classifier_v6.txt"
    log_file = open(filename, "r")
    # fig_file = "train_test_sample_plot.png"
    rx_train = re.compile("Batch Loss [0-9]+.[0-9]+")
    rx_tiny_test = re.compile("Tiny Test Loss [0-9]+.[0-9]+")
    rx_num = re.compile("[0-9]+.?[0-9]*")
    def get_num(substr, num_type="float"):
        num_match = re.search(rx_num, substr)
        if not num_match:
            raise Exception("no number in substring")
        if num_type == "float":
            return float(num_match.group(0))
        elif num_type == "int":
            return int(num_match.group(0))
        else:
            raise TypeError("num_type must be one of 'int', 'float'")
    def maybe_get_regex_num(line, regex, num_type="float"):
        rx_match = re.search(regex, line)
        if rx_match:
            num = get_num(rx_match.group(0), num_type)
            return num
        return None
    train_losses = []
    tiny_test_losses = []
    for line in log_file:
        tl = maybe_get_regex_num(line, rx_train)
        ttl = maybe_get_regex_num(line, rx_tiny_test)
        train_losses.append(tl)
        tiny_test_losses.append(ttl)
    log_file.close()

def clean(df, iter_name="iter"):
    new_df = df
    melted = pd.melt(new_df, id_vars=iter_name, value_vars=["train", "test"], value_name="loss", var_name="set")
    non_nan = ~(np.isnan(melted.loss))
    non_zero = ~(melted.loss == 0.0)
    valid = non_nan & non_zero
    melted_valid = melted[valid]
    return melted_valid
    

plt.clf()
fig, ax = plt.subplots(figsize=(10,5), dpi=500)
iter_batch_size = 1
if len(epoch_mean_batch_losses) >= 5:
    df = pd.DataFrame({"train":epoch_mean_batch_losses, "test":epoch_mini_test_losses, "iter":np.arange(len(epoch_mean_batch_losses))})
    df_clean = clean(df)
    lp = sns.lineplot(ax=ax, data=df_clean, x="iter", y="loss", hue="set")
else:
    df = pd.DataFrame({"train":train_losses, "test":tiny_test_losses, "minibatch":np.arange(len(train_losses))//iter_batch_size * iter_batch_size})
    df_clean = clean(df, "minibatch")
    lp = sns.lineplot(ax=ax, data=df_clean, x="minibatch", y="loss", hue="set")
min_test_loss = min(df.test)
min_train_loss = min(df.train)
final_test_loss = df.test.to_numpy()[-1]
# ax.set_yscale('log', basey=2)
ax.axhline(y=min_test_loss, color='grey')
ax.axhline(y=final_test_loss, color='grey')
ax.set_xlabel("Epoch")
ax.set_ylabel("Loss")
# plt.yscale("log", base=2)
ymin, ymax = ax.get_ylim()
ymin = 2**np.floor(np.log2(min(min_train_loss,min_test_loss)))
# ymin = 0.0
ax.set_ylim(ymin, ymax)
fig = lp.get_figure()
fig.set_size_inches(6,4)
fig.set_dpi(300)
fig.tight_layout()
fig.savefig(train_test_fig_file)

def plot_roc(preds, targets, roc_fig_file):
    if preds.shape[1] == 2:
        preds = preds[:,1]
    if preds.shape[-1] == 1 or len(preds.shape) == 1:
        plt.clf()
        fig, ax = plt.subplots(figsize=(10,5), dpi=500)
        metrics.RocCurveDisplay.from_predictions(
            targets.ravel(),
            preds.ravel(),
            color="orange",
            plot_chance_level=True,
        )
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.legend()
        plt.savefig(roc_fig_file)
        roc_auc = metrics.roc_auc_score(y_true=targets, y_score=preds)
        print(f"ROC AUC: {roc_auc}")

# ROC plot
if test_summary != None:
    if type(test_summary["ALL_PREDS"]) == list and len(test_summary["ALL_PREDS"]) == 2 and len(test_summary["ALL_PREDS"][1]) == 0:
        test_summary["ALL_PREDS"] = test_summary["ALL_PREDS"][0]
    if type(test_summary["ALL_TARGETS"]) == list and len(test_summary["ALL_TARGETS"]) == 2 and len(test_summary["ALL_TARGETS"][1]) == 0:
        test_summary["ALL_TARGETS"] = test_summary["ALL_TARGETS"][0]
    if type(test_summary["ALL_PREDS"]) == list and len(test_summary["ALL_PREDS"]) == 2 and len(test_summary["ALL_PREDS"][1]) > 0:
        # plot roc using both urate and gout predictions
        gout_preds  = np.array([i.float().numpy() for i in test_summary["ALL_PREDS"][0]])
        urate_preds = np.array([i.float().numpy() for i in test_summary["ALL_PREDS"][1]])
        gout_targets = np.array([i.numpy() for i in test_summary["ALL_TARGETS"][0]])
        plot_roc(gout_preds, gout_targets, roc_fig_file + "_gout_preds.pdf")
        plot_roc(urate_preds, gout_targets, roc_fig_file + "_urate_preds.pdf")
    else:
        # single plot only
        preds = np.array([i.float().numpy() for i in test_summary["ALL_PREDS"]])
        targets = np.array([i.numpy() for i in test_summary["ALL_TARGETS"]])
        plot_roc(preds, targets, roc_fig_file)