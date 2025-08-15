#!/usr/bin/bash
# remove short epochs for final run
# N.B. seq_len here is max, rather than actual. (i.e. we have padding).
# hyena doesn't work with multiple gpus.
torchrun \
  --master_addr="localhost" \
   --nnodes=1 \
   --nproc-per-node=4 \
   --max-restarts=0 \
   --rdzv-id=mlp_test \
hyena_encoder.py \
	--total-epochs=10 \
	--num-layers=2 \
	--batch-size=2 \
	--num-heads=8 \
    --d-model=128 \
    --seq-len=1048576 \
    --h5-file="/data/ukbb/net_input/all_unimputed_combined_v6.h5" \
    --test-frac=0.3 \
	--devices=0,1,2 \
    --snapshot-path=/data/ukbb/v2_snapshots/script_hyena_all_unimputed_v6.pt \
    --warmup-batches=500 \
    --warmup-min-lr=1e-10 \
    --warmup-max-lr=1e-7 \
    --main-scheduler-step-size=1000 \
    --main-scheduler-gamma=0.95 \
    --report-on-batch=500 \

    # --h5-file="/data/ukbb/net_input/gwas_ldprune_320.h5" \