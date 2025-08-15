#!/usr/bin/bash
torchrun \
  --master_addr="localhost" \
    --nnodes=1 \
    --nproc_per_node=3 \
    train_encoder.py \
        --encoder-type="linformer" \
        --total-epochs=2 \
        --num-layers=8 \
        --batch-size=1 \
        --embed-dim=256 \
        --num-heads=8 \
        --seq-len=65803  \
        --linformer-k=1024 \
        --h5-file="/data/ukbb/net_input/genotyped_p1e-1_v2.h5" \
        --test-frac=0.3 \
        --devices=0,1,2 \
        --snapshot-path=/data/ukbb/v2_snapshots/script_linformer_encoder_ld_learnpos_bigk.pt \
        --warmup-batches=500 \
        --warmup-min-lr=1e-10 \
        --warmup-max-lr=1e-7 \
        --main-scheduler-step-size=1000 \
        --main-scheduler-gamma=0.95 \
        --report-on-batch=100 \
        --mask-frac=0.15 \
        --no-augment-data
