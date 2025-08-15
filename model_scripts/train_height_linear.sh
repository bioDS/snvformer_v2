#!/usr/bin/bash
torchrun \
   --nnodes=1 \
   --nproc-per-node=4 \
   --max-restarts=2 \
  --master_addr="localhost" \
train_height.py \
    --model-type="linear" \
    --total-epochs 5 \
    --seq-len=13228 \
    --h5-file="/data/ukbb/net_input/ld_full_r00001.h5" \
    --test-frac=0.3 \
    --devices=0,1,2 \
    --snapshot-path=/data/ukbb/v2_snapshots/script_height_predictor_linear.pt \
    --warmup-batches=500 \
    --warmup-min-lr=1e-10 \
    --warmup-max-lr=1e-7 \
    --main-scheduler-step-size=1000 \
    --main-scheduler-gamma=0.9 \
    --report-on-batch=10
