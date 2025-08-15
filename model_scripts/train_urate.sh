#!/usr/bin/bash
torchrun \
   --nnodes=1 \
   --nproc-per-node=4 \
   --max-restarts=0 \
    --master_addr="localhost" \
train_urate.py \
    --encoder-type="classic" \
	--total-epochs 5 \
	--num-layers=8 \
	--batch-size=1 \
	--embed-dim=128 \
	--num-heads=8 \
    --seq-len=13290 \
    --h5-file="/data/ukbb/net_input/all_gwas.h5" \
    --test-frac=0.3 \
	--devices=0,1,2 \
    --encoder-snapshot=/data/ukbb/v2_snapshots/script_encoder.pt \
    --snapshot-path=/data/ukbb/v2_snapshots/script_urate_predictor_all_gwas.pt \
    --warmup-batches=500 \
    --warmup-min-lr=1e-10 \
    --warmup-max-lr=1e-7 \
    --main-scheduler-step-size=1000 \
    --main-scheduler-gamma=0.9 \
    --report-on-batch=10