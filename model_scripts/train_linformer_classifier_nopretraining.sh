#!/usr/bin/bash
torchrun \
  --master_addr="localhost" \
   --nnodes=1 \
   --nproc_per_node=3 \
train_classifier.py \
    --encoder-type="linformer" \
	--total-epochs 200 \
	--num-layers=8 \
	--batch-size=16 \
	--embed-dim=256 \
	--num-heads=8 \
    --seq-len=13228  \
    --linformer-k=256 \
    --h5-file="/data/ukbb/net_input/ld_full_r00001_v6.h5" \
    --test-frac=0.3 \
	--devices=0,1,2 \
    --encoder-snapshot=/data/ukbb/v2_snapshots/script_linformer_encoder_ld_learnpos.pt \
    --snapshot-path=/data/ukbb/v2_snapshots/script_linformer_classifier_ld_full_learnpos_nopretraining.pt \
    --warmup-batches=500 \
    --warmup-min-lr=1e-10 \
    --warmup-max-lr=1e-7 \
    --main-scheduler-step-size=1000 \
    --main-scheduler-gamma=0.95 \
    --report-on-batch=20 \
    --no-gradscaler \
    --no-load-encoder
    #--augment-data \
    #--augment-frac=0.15 \
    #--augment-mult=5.0 \
