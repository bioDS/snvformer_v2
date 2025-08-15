#!/usr/bin/bash
# remove short epochs for final run
torchrun --standalone --nproc_per_node=3 --master_addr="localhost" train_encoder.py \
    --encoder-type="classic" \
	--total-epochs=2 \
	--num-layers=8 \
	--batch-size=32 \
	--embed-dim=256 \
	--num-heads=8 \
    --seq-len=13228 \
    --h5-file="/data/ukbb/net_input/ld_full_r00001_v6.h5" \
    --test-frac=0.3 \
	--devices=0,1,2 \
    --snapshot-path=/data/ukbb/v2_snapshots/script_encoder_ld_full.pt \
    --warmup-batches=500 \
    --warmup-min-lr=1e-10 \
    --warmup-max-lr=1e-7 \
    --main-scheduler-step-size=1000 \
    --main-scheduler-gamma=0.95 \
    --report-on-batch=100 \
    --gradual-length \
    --no-augment-data