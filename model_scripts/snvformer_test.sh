#!/usr/bin/bash
torchrun \
  --master_addr="localhost" \
  --master_port="12423" \
   --nnodes=1 \
   --nproc_per_node=3 \
train_classifier.py \
    --encoder-type="linformer" \
	--total-epochs=60 \
	--num-layers=2 \
	--batch-size=2 \
	--embed-dim=32 \
	--num-heads=2 \
    --seq-len=65803  \
    --linformer-k=64 \
    --h5-file="/data/ukbb/net_input/genotyped_p1e-1_v2.h5" \
    --test-frac=0.3 \
	--devices=0,1,2 \
    --snapshot-path=/data/ukbb/v2_snapshots/snvformer_test.pt \
    --warmup-batches=500 \
    --warmup-min-lr=1e-5 \
    --warmup-max-lr=1e-3 \
    --main-scheduler-step-size=1000 \
    --main-scheduler-gamma=0.95 \
    --report-on-batch=100 \
    --no-load-encoder \
    --model-type="simple" \
    --encoder-type="linformer" \
    # --no-gradscaler \
    # --position-encoding=fixed \