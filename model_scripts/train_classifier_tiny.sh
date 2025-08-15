#!/usr/bin/bash
torchrun \
   --nnodes=1 \
   --nproc-per-node=4 \
   --max-restarts=0 \
   --rdzv-id=mlp_test \
  --master_addr="localhost" \
train_classifier.py \
	--total-epochs 20 \
	--num-layers=2 \
	--batch-size=1024 \
	--embed-dim=32 \
	--num-heads=2 \
    --seq-len=320 \
    --h5-file="/data/ukbb/net_input/gwas_ldprune_320.h5" \
    --test-frac=0.3 \
	--devices=0,1,2 \
    --encoder-snapshot=/data/ukbb/v2_snapshots/script_tiny_encoder.pt \
    --snapshot-path=/data/ukbb/v2_snapshots/script_tiny_classifier.pt \
    --warmup-batches=500 \
    --warmup-min-lr=1e-5 \
    --warmup-max-lr=1e-3 \
    --main-scheduler-step-size=1000 \
    --main-scheduler-gamma=0.9 \
    --report-on-batch=100 \
    # --augment-data 
