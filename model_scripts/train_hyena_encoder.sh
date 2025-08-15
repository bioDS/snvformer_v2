#!/usr/bin/bash
# remove short epochs for final run
# N.B. seq_len here is max, rather than actual. (i.e. we have padding).
# hyena doesn't work with multiple gpus.
torchrun \
  --master_addr="localhost" \
   --nnodes=1 \
    --nproc_per_node=3 \
hyena_encoder.py \
	--total-epochs=20 \
	--num-layers=8 \
	--batch-size=16 \
	--embed-dim=256 \
	--num-heads=8 \
    --seq-len=13228  \
    --linformer-k=256 \
    --h5-file="/data/ukbb/net_input/ld_full_r00001_v6.h5" \
    --test-frac=0.3 \
	--devices=0,1,2 \
    --snapshot-path=/data/ukbb/v2_snapshots/script_hyena_ld_lengthscaled_learnpos.pt \
    --warmup-batches=20000 \
    --warmup-min-lr=1e-10 \
    --warmup-max-lr=1e-7 \
    --main-scheduler-step-size=100000 \
    --main-scheduler-gamma=0.95 \
    --report-on-batch=500 \
    --gradual-length \
    --mask-frac=0.15 \
