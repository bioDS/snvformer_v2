from trainer import *
from train_classifier import *
from vectorizer import *
import numpy as np

parser = get_default_argparser()
args = parser.parse_args()

interactive_setup()
ddp_setup()
args.pretrained_hyena = None
args.d_model = 128
model = get_model(args)
train_set, mini_test_set, verify_set, optimizer, warmup_scheduler, main_scheduler = load_train_objs(args, model)

def get_snv_diff(ds, use_inds):
    found_diff = []
    for i in np.random.choice(use_inds, 100, replace=False):
        for j in np.random.choice(use_inds, 100, replace=False):
            if (i != j):
                num_same = np.sum(ds[i,:] == ds[j,:])
                num_diff = np.sum(ds[i,:] != ds[j,:])
                found_diff.append(num_diff)
    seq_len = len(ds[0,:])
    frac_diff = [float(i)/seq_len for i in found_diff]
    return frac_diff

import seaborn as sns
frac_diff = get_snv_diff(train_set.h5_dataset, len(train_set))
fig = sns.displot(frac_diff, )
fig.set_xlabels("fraction of identical snv positions")
fig.savefig("snv_diff_all.png")

inds = np.where(train_set.phenos["gout"] == True)[0]
frac_diff = get_snv_diff(train_set.h5_dataset, inds)
fig = sns.displot(frac_diff, )
fig.set_xlabels("fraction of identical snv positions")
fig.savefig("snv_diff_gout.png")

inds = np.where(train_set.phenos["gout"] == False)[0]
frac_diff = get_snv_diff(train_set.h5_dataset, inds)
fig = sns.displot(frac_diff, )
fig.set_xlabels("fraction of identical snv positions")
fig.savefig("snv_diff_nongout.png")

# get loss of random and mean predictions
all_verify_gout = verify_set.phenos["gout"].to_numpy()
mean_gout = np.mean(all_verify_gout)
mean_preds = torch.tensor(np.repeat(mean_gout, len(all_verify_gout)), dtype=torch.float32)
random_preds = torch.tensor(np.random.rand(len(all_verify_gout)), dtype=torch.float32)
mid_preds    = torch.tensor(np.repeat(0.5, len(all_verify_gout)), dtype=torch.float32)

targets = torch.tensor(all_verify_gout, dtype=torch.float32)

mean_pred_loss = torch.nn.functional.binary_cross_entropy(mean_preds, targets)
random_pred_loss = torch.nn.functional.binary_cross_entropy(random_preds, targets)
mid_pred_loss = torch.nn.functional.binary_cross_entropy(mid_preds, targets)

print(f"mean pred loss: {mean_pred_loss}")
print(f"random pred loss: {random_pred_loss}")
print(f"mid pred loss: {mid_pred_loss}")