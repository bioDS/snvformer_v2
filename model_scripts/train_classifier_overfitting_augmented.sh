#!/usr/bin/bash
torchrun --standalone --nproc-per-node=4 --master_addr="localhost" train_classifier.py \
	--total-epochs 2 \
	--num-layers=8 \
	--batch-size=32 \
	--embed-dim=128 \
	--num-heads=8 \
    --seq-len=13290 \
    --h5-file="/data/ukbb/net_input/all_gwas.h5" \
    --test-frac=0.3 \
	--devices=0,1,2 \
    --encoder-snapshot=/data/ukbb/v2_snapshots/script_encoder_2epochs.pt \
    --snapshot-path=/data/ukbb/v2_snapshots/script_classifier.pt \
    --warmup-batches=500 \
    --warmup-min-lr=1e-3 \
    --warmup-max-lr=1e-1 \
    --main-scheduler-step-size=1000 \
    --main-scheduler-gamma=0.9 \
    --report-on-batch=100 \
