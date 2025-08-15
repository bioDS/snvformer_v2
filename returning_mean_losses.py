from trainer import *
import pandas as pd
from pandas import DataFrame
import os

def get_row_loss(phenos, mean_preds, loss_lists):
    urate_losses, height_losses, gout_losses = loss_lists
    mean_urate, mean_height, pred_gout = mean_preds
    urate_loss = torch.nn.functional.mse_loss(torch.tensor(mean_urate), torch.tensor(phenos["urate"]))
    height_loss = torch.nn.functional.mse_loss(torch.tensor(mean_height), torch.tensor(phenos["height"]))
    gout_loss = torch.nn.functional.cross_entropy(pred_gout, torch.tensor(phenos["gout"]).long())
    urate_losses.append(urate_loss)
    height_losses.append(height_loss)
    gout_losses.append(gout_loss)

def print_loss_lists(list):
    urate_losses, height_losses, gout_losses = list
    mean_urate_loss = np.mean(urate_losses)
    print(f"mean urate loss: {mean_urate_loss}")
    mean_height_loss = np.mean(height_losses)
    print(f"mean height loss: {mean_height_loss}")
    mean_gout_loss = np.mean(gout_losses)
    print(f"mean gout loss: {mean_gout_loss}")

def get_mean_phenos(phenos):
    mean_urate = np.mean(phenos["urate"])
    mean_height = np.mean(phenos["height"])
    mean_gout = np.mean(phenos["gout"])
    gout_pred = torch.tensor([1-mean_gout, mean_gout])
    return (mean_urate, mean_height, gout_pred)

def main(args):
    train_inds, test_inds, unused_inds = get_train_test(args)
    pretrain_inds = np.sort(np.concatenate((train_inds, unused_inds)))
    train_set_full = StreamingDataset(args.h5_file, use_inds=pretrain_inds, augment_data=False)
    train_set_balanced = StreamingDataset(args.h5_file, use_inds=train_inds, augment_data=True, augment_mult=1.0)
    test_set = StreamingDataset(args.h5_file, use_inds=test_inds, augment_frac=args.augment_frac, augment_data=False)

    # full_mean_preds = (full_mean_urate, full_mean_height, full_pred_gout)
    full_mean_preds = get_mean_phenos(train_set_full.phenos)
    full_mean_male_preds = get_mean_phenos(train_set_full.phenos[train_set_full.phenos["sex"] == "Male"])
    full_mean_female_preds = get_mean_phenos(train_set_full.phenos[train_set_full.phenos["sex"] == "Female"])
    full_mean_gout_preds = get_mean_phenos(train_set_full.phenos[train_set_full.phenos["gout"] == True])
    full_mean_nongout_preds = get_mean_phenos(train_set_full.phenos[train_set_full.phenos["gout"] == False])
    balanced_mean_preds = get_mean_phenos(train_set_balanced.phenos)
    balanced_mean_male_preds = get_mean_phenos(train_set_balanced.phenos[train_set_balanced.phenos["sex"] == "Male"])
    balanced_mean_female_preds = get_mean_phenos(train_set_balanced.phenos[train_set_balanced.phenos["sex"] == "Female"])
    balanced_mean_gout_preds = get_mean_phenos(train_set_balanced.phenos[train_set_balanced.phenos["gout"] == True])
    balanced_mean_nongout_preds = get_mean_phenos(train_set_balanced.phenos[train_set_balanced.phenos["gout"] == False])
    full_loss_lists = ([], [], [])
    full_male_loss_lists = ([], [], [])
    full_female_loss_lists = ([], [], [])
    full_gout_loss_lists = ([], [], [])
    full_nongout_loss_lists = ([], [], [])
    balanced_loss_lists = ([], [], [])
    balanced_male_loss_lists = ([], [], [])
    balanced_female_loss_lists = ([], [], [])
    balanced_gout_loss_lists = ([], [], [])
    balanced_nongout_loss_lists = ([], [], [])
    for phenos, positions, chromosomes, snvs in test_set:
        get_row_loss(phenos, full_mean_preds, full_loss_lists)
        get_row_loss(phenos, balanced_mean_preds, balanced_loss_lists)
        if phenos["sex"] == "Male":
            get_row_loss(phenos, balanced_mean_male_preds, balanced_male_loss_lists)
            get_row_loss(phenos, full_mean_male_preds, full_male_loss_lists)
        else:
            get_row_loss(phenos, balanced_mean_female_preds, balanced_female_loss_lists)
            get_row_loss(phenos, full_mean_female_preds, full_female_loss_lists)
        if phenos["gout"]:
            get_row_loss(phenos, balanced_mean_gout_preds, balanced_gout_loss_lists)
            get_row_loss(phenos, full_mean_gout_preds, full_gout_loss_lists)
        else:
            get_row_loss(phenos, balanced_mean_nongout_preds, balanced_nongout_loss_lists)
            get_row_loss(phenos, full_mean_nongout_preds, full_nongout_loss_lists)

    print("Full losses")
    print_loss_lists(full_loss_lists)
    print("full male losses")
    print_loss_lists(full_male_loss_lists)
    print("full female losses")
    print_loss_lists(full_female_loss_lists)
    print("full gout losses")
    print_loss_lists(full_gout_loss_lists)
    print("full non-gout losses")
    print_loss_lists(full_nongout_loss_lists)
    print("Balanced losses")
    print_loss_lists(balanced_loss_lists)
    print("balanced male losses")
    print_loss_lists(balanced_male_loss_lists)
    print("balanced female losses")
    print_loss_lists(balanced_female_loss_lists)
    print("Balanced gout losses")
    print_loss_lists(balanced_gout_loss_lists)
    print("Balanced non-gout losses")
    print_loss_lists(balanced_nongout_loss_lists)

if __name__ == "__main__":
    parser = get_default_argparser()
    parser.add_argument("--snapshot-path", default="/data/ukbb/v2_snapshots/classifier.pt")
    parser.add_argument("--urate-weight", type=float, default=1.0, help="urate loss is multiplied by this weight in training")
    parser.add_argument("--gout-weight", type=float, default=1.0, help="gout loss is multiplied by this weight in training")
    args = parser.parse_args()
    main(args)