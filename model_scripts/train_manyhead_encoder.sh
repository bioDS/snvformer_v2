#!/usr/bin/bash
# remove short epochs for final run
# N.B. embed dim is divided by num_heads internally
# Batch size is per-gpu
torchrun --standalone --nproc_per_node=3 --master_addr="localhost" train_encoder.py \
    --encoder-type="classic" \
    --total-epochs 100 \
    --num-layers=8 \
    --batch-size=4 \
    --embed-dim=4096 \
    --num-heads=256 \
    --seq-len=13228 \
    --h5-file="/data/ukbb/net_input/ld_full_r00001_v6.h5" \
    --test-frac=0.3 \
    --devices=0,1,2 \
    --snapshot-path=/data/ukbb/v2_snapshots/script_encoder_manyheads_ld_full.pt \
    --warmup-batches=500 \
    --warmup-min-lr=1e-10 \
    --warmup-max-lr=1e-7 \
    --main-scheduler-step-size=1000 \
    --main-scheduler-gamma=0.9 \
    --report-on-batch=1 \
    --limit-steps=100 \
    --no-augment-data
