#!/usr/bin/env python
from trainer import *
import os
from encoder import *
from hyena_encoder import get_model
from packaging import version
from contextlib import nullcontext
from train_classifier import *
from gene_information import *
import pandas as pd
import biomart
import matplotlib.pyplot as plt
import seaborn as sns

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Load pre-trained encoder and get most attended-to SNVs in test data')
    parser.add_argument("--model-snapshot", default="/data/ukbb/v2_snapshots/snvformer_hosearch_nopt_ldfull.pt")
    parser.add_argument("--output-file", default="test_sig_snvs.csv")
    args = vars(parser.parse_args())

    snapshot = torch.load(args["model_snapshot"], map_location="cpu")
    encoder = custom_encoder_from_args_dict(snapshot["PROGRAM_ARGS"], ["age", "sex", "bmi"])
    snapshot_args = snapshot["PROGRAM_ARGS"]
    model_type = snapshot_args["model_type"]
    use_phenos = ["age", "sex", "bmi"]

    if model_type == "hyena":
        encoder = get_model(snapshot_args)
    else:
        encoder = custom_encoder_from_args(snapshot_args, use_phenos)
    # if model_type == "hyena":
    #     classifier = prerun_get_classifier(encoder, SimpleHyenaClassifier, HyenaClassifierTrainer, snapshot_args)
    # elif model_type == "flat_output":
    #     classifier = prerun_get_classifier(encoder, FlatOutputClassifier, ClassifierTrainer, snapshot_args)
    # elif model_type == "dual_output":
    #     classifier = prerun_get_classifier(encoder, OldDualClassifier, CEOnlyClassifierTrainer, snapshot_args)
    # else:
    # tmp_classifier = prerun_get_classifier(encoder, SimpleClassifier, ClassifierTrainer, snapshot_args)

    # classifier = classifier.to("cpu")
    os.environ["LOCAL_RANK"] = "0"
    train_set, mini_test_set, verify_set, optimizer, warmup_scheduler, main_scheduler = load_train_objs(snapshot_args, None)

    # train_data = prepare_dataloader(train_set, snapshot_args["batch_size"])

    train_data = DataLoader(
        train_set,
        batch_size=snapshot_args["batch_size"],
        pin_memory=False,
        shuffle=False,
        collate_fn=collate_fn
    )
    
    # train_data = prepare_dataloader(train_set, snapshot_args["batch_size"])
    # mini_test_data = prepare_dataloader(mini_test_set, snapshot_args["batch_size"])
    # verify_data = prepare_dataloader(verify_set, snapshot_args["batch_size"])

    # use_devices = 'cpu'
    # if model_type == "hyena":
    #     trainer = HyenaClassifierTrainer(classifier, train_data, mini_test_data, verify_data, optimizer, warmup_scheduler, main_scheduler, snapshot_args["save_every"], snapshot_args["snapshot_path"], use_devices, args, snapshot_args["limit_steps"])
    # elif model_type == "dual_output":
    #     trainer = CEOnlyClassifierTrainer(classifier, train_data, mini_test_data, verify_data, optimizer, warmup_scheduler, main_scheduler, snapshot_args["save_every"], snapshot_args["snapshot_path"], use_devices, args, snapshot_args["limit_steps"], snapshot_args["gradual_length"], target='gout')
    # else:
    #     trainer = ClassifierTrainer(classifier, train_data, mini_test_data, verify_data, optimizer, warmup_scheduler, main_scheduler, snapshot_args["save_every"], snapshot_args["snapshot_path"], use_devices, args, snapshot_args["limit_steps"], snapshot_args["gradual_length"], target='gout')

    # now run the network once through the training data, collecting self-attention scores for each SNV

    # bc, batch = next(enumerate(train_loader))
    batch_iter, (phenos, positions, chromosomes, snvs) = next(enumerate(train_data))
    # ensembl_to_entrez, entrez_to_ensemble = get_ensembl_mappings()
    data = EnsemblRelease(75) # 75 is the last to use GRCh37
    # torch.tensor can't contain strings
    # ensembl_gene_names = np.array([unpack_adjust_list(data.gene_ids_at_locus(chrom.item(), loci.item())) for chrom, loci in zip(chromosomes[0,:], positions[0,:])])
    ensembl_gene_names = np.array([unpack_adjust_list(data.gene_names_at_locus(chrom.item(), loci.item())) for chrom, loci in zip(chromosomes[0,:], positions[0,:])])

    use_device = 4
    encoder = encoder.to(use_device)
    weights = None
    abs_weights = None
    for batch_iter, (phenos, positions, chromosomes, snvs) in enumerate(train_data):
        with torch.no_grad():
            source = (phenos, positions.to(use_device), chromosomes.to(use_device), snvs.to(use_device), 0)
            if weights == None:
                weights, abs_weights = encoder.get_attn(*source)
            else:
                a,b = encoder.get_attn(*source)
                weights += a
                abs_weights += b

    use_weights = weights[4:] # ignore class, age, sex, bmi
    use_abs_weights = abs_weights[4:]
    snv_results = pd.DataFrame({
        "weight": use_weights.cpu().numpy(),
        "abs_weight": use_abs_weights.cpu().numpy(),
        "snv_ind": np.arange(0,len(use_weights)),
        "gene_id": ensembl_gene_names
    })
    snv_results = snv_results.sort_values(by="abs_weight", ascending=False)
    # snv_results = snv_results.sort_values(by="gene_id")

    gene_weights = {}
    gene_abs_weights = {}
    for _, row in snv_results.iterrows():
        gid = row["gene_id"]
        if gid != None:
            # print(gid)
            if gid in gene_weights:
                gene_weights[gid] += row["weight"]
                gene_abs_weights[gid] += row["abs_weight"]
            else:
                gene_weights[gid] = row["weight"]
                gene_abs_weights[gid] = row["abs_weight"]

    gk = gene_weights.keys()
    gaw = gene_abs_weights.keys()
    assert(gk == gaw)
    gene_results = pd.DataFrame({
        "gene_id": gk,
        "weight": gene_weights.values(),
        "abs_weight": gene_abs_weights.values()
    })
    gene_results = gene_results.sort_values(by="abs_weight", ascending=False)

    quit()
    # normalise gene_results by number of SNVs present in gene in dataset
    non_nan_gene_names = [x for x in ensembl_gene_names if x != None]
    names, counts = np.unique(non_nan_gene_names, return_counts=True)
    gene_counts = pd.DataFrame({"gene_id": names, "snv_count": counts})
    cgr = gene_results.set_index("gene_id").join(gene_counts.set_index("gene_id"))
    cgr = cgr.assign(adj_abs_weight = lambda x: x["abs_weight"]/x["snv_count"])
    cgr = cgr.sort_values(by="adj_abs_weight", ascending=False)

    snv_results.to_csv(args["output_file"] + ".snv_results.csv.gz", compression="gzip")
    cgr.to_csv(args["output_file"], compression="gzip")

    # run manually and set ldcgr to the ldfull cgr, cgr to all_gwas cgr
    combined_cgr = pd.concat([cgr.assign(Dataset="GWMed"), ldcgr.assign(Dataset="LDMed")])
    plt.figure()
    ax = sns.kdeplot(data=combined_cgr, x="adj_abs_weight", hue="Dataset")
    ax.set_xlabel("Adjusted Absolute Weight")
    plt.savefig("figures/multi-adj_weight_dist.png")
    plt.savefig("figures/multi-adj_weight_dist.pdf")