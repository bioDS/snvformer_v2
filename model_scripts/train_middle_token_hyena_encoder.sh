#!/usr/bin/bash
# remove short epochs for final run
# N.B. seq_len here is max, rather than actual. (i.e. we have padding).
# hyena doesn't work with multiple gpus.
torchrun \
    --master_addr="localhost" \
    --standalone \
    --nnodes=1 \
    --nproc_per_node=3 \
hyena_middle_prediction_encoder.py \
	--total-epochs=3 \
	--num-layers=4 \
	--batch-size=4 \
	--num-heads=8 \
    --d-model=128 \
    --seq-len=32768 \
    --h5-file="/data/ukbb/net_input/all_unimputed_combined_v6.h5" \
    --test-frac=0.3 \
	--devices=0,1,2 \
    --snapshot-path=/data/ukbb/v2_snapshots/script_hyena_all_unimputed_middle_v6.pt \
    --warmup-batches=20000 \
    --warmup-min-lr=1e-10 \
    --warmup-max-lr=1e-7 \
    --main-scheduler-step-size=10000 \
    --main-scheduler-gamma=0.9 \
    --report-on-batch=10 \
    --step-progress \
    --view-size=32768 \
    --gradual-length \