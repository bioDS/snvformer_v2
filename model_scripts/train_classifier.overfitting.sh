#!/usr/bin/bash
# N.B. was previously run with flassh attention, not math
torchrun \
  --master_addr="localhost" \
  --master_port="12453" \
   --nnodes=1 \
   --nproc_per_node=3 \
train_classifier.py \
	--total-epochs 2 \
	--num-layers=8 \
	--batch-size=32 \
	--embed-dim=128 \
	--num-heads=8 \
    --seq-len=13290 \
    --h5-file="/data/ukbb/net_input/all_gwas.h5" \
    --test-frac=0.3 \
	--devices=0,1,2 \
    --encoder-snapshot=/data/ukbb/v2_snapshots/script_encoder.pt \
    --snapshot-path=/data/ukbb/v2_snapshots/script_classifier.pt \
    --warmup-batches=500 \
    --warmup-min-lr=1e-3 \
    --warmup-max-lr=1e-1 \
    --main-scheduler-step-size=1000 \
    --main-scheduler-gamma=0.9 \
    --report-on-batch=100 \