#!/usr/bin/bash
torchrun \
    --master_addr="localhost" \
   --nnodes=1 \
   --nproc_per_node=3 \
train_classifier.py \
    --model-type="hyena" \
	--total-epochs 20 \
	--num-layers=2 \
	--batch-size=2 \
	--num-heads=8 \
    --d-model=128 \
    --embed-dim=128 \
    --seq-len=1048576 \
    --h5-file="/data/ukbb/net_input/all_unimputed_combined.h5" \
    --test-frac=0.3 \
	--devices=0,1,2 \
    --encoder-snapshot=/data/ukbb/v2_snapshots/script_hyena_all_unimputed_lengthscaled.pt \
    --snapshot-path=/data/ukbb/v2_snapshots/script_hyena_classifier_all_unimputed.pt \
    --warmup-batches=20000 \
    --warmup-min-lr=1e-10 \
    --warmup-max-lr=1e-7 \
    --main-scheduler-step-size=10000 \
    --main-scheduler-gamma=0.9 \
    --report-on-batch=500 \
    --augment-data \
    --augment-frac=0.15 \
    --augment-mult=5.0 \