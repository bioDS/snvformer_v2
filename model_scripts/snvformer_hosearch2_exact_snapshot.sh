#!/usr/bin/bash
# this version reuses the snapshot from the best iter of the best hyperparameter trial
use_snapshot="/home/kieran_elmes/gout_transformer_v2/ray_results/fault_tolerant_run/train_cifar_092bc_00095_95_batch_size=4,dropout=0.1284,ffn_scale=2,gene_embed_graph=True,gene_embed_size=8,ignore_chrom=False,igno_2023-10-08_04-27-48/checkpoint_000011/snapshot.pt"
save_path="/data/ukbb/v2_snapshots/hosearch_actual_best.pt"
cp "$use_snapshot" "$save_path"
torchrun \
  --master_addr="localhost" \
  --master_port="12623" \
   --nnodes=1 \
   --nproc_per_node=1 \
train_classifier.py \
    --encoder-type="linformer" \
	--total-epochs=0 \
	--num-layers=3 \
	--batch-size=4 \
	--embed-dim=56 \
	--num-heads=4 \
    --seq-len=65803  \
    --linformer-k=32 \
    --h5-file="/data/ukbb/net_input/genotyped_p1e-1.h5" \
    --test-frac=0.3 \
	--devices=0 \
    --no-load-encoder \
    --snapshot-path="$save_path" \
    --warmup-batches=500 \
    --warmup-min-lr=6.0514340987984604e-05 \
    --warmup-max-lr=6.0514340987984604e-05 \
    --main-scheduler-step-size=1000 \
    --main-scheduler-gamma=1.00 \
    --report-on-batch=1000 \
    --position-encoding="embedding" \
    --snv-encoding="embedding" \
    --pos-combine="add" \
    --torch-sdp \
    --ffn-scale=2 \
    --dropout=0.1284368165277642 \
    --snv-embed-size=16 \
    --pos-embed-size=56 \
    --gene-embed-size=8 \
    --chrom-embed-size=32 \
    --gene-embed-graph \
    --no-gradscaler \
    --tf-init \
    --ignore-class \
    --model-type="single_output" \
    --target="gout" \