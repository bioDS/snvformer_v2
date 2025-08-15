#!/usr/bin/bash
# remove short epochs for final run
torchrun \
   --nnodes=1 \
   --nproc-per-node=4 \
   --max-restarts=0 \
   --rdzv-id=mlp_test \
  --master_addr="localhost" \
train_encoder.py \
	--total-epochs 2 \
	--num-layers=2 \
	--batch-size=512 \
	--embed-dim=32 \
	--num-heads=2 \
    --seq-len=320 \
    --h5-file="/data/ukbb/net_input/gwas_ldprune_320.h5" \
    --test-frac=0.3 \
	--devices=0,1,2 \
    --snapshot-path=/data/ukbb/v2_snapshots/script_tiny_encoder.pt \
    --warmup-batches=500 \
    --warmup-min-lr=1e-10 \
    --warmup-max-lr=1e-7 \
    --main-scheduler-step-size=1000 \
    --main-scheduler-gamma=0.95 \
    --report-on-batch=100 \
    --no-augment-data