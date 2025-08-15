#!/usr/bin/bash
torchrun \
   --nnodes=1 \
   --nproc-per-node=4 \
   --max-restarts=2 \
  --master_addr="localhost" \
train_height.py \
    --encoder-type="classic" \
	--total-epochs 5 \
	--num-layers=8 \
	--batch-size=16 \
	--embed-dim=128 \
	--num-heads=8 \
    --seq-len=13228 \
    --h5-file="/data/ukbb/net_input/ld_full_r00001.h5" \
    --test-frac=0.3 \
	--devices=0,1,2 \
    --encoder-snapshot=/data/ukbb/v2_snapshots/script_encoder_ld_full.pt \
    --snapshot-path=/data/ukbb/v2_snapshots/script_height_predictor.pt \
    --warmup-batches=500 \
    --warmup-min-lr=1e-10 \
    --warmup-max-lr=1e-7 \
    --main-scheduler-step-size=1000 \
    --main-scheduler-gamma=0.95 \
    --report-on-batch=10 \
    --no-gradscaler \