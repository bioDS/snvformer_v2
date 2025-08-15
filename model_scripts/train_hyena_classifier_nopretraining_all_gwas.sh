#!/usr/bin/bash
torchrun \
  --master_addr="localhost" \
   --nnodes=1 \
   --nproc_per_node=3 \
train_classifier.py \
    --encoder-type="hyena" \
	--total-epochs 200 \
	--num-layers=8 \
	--batch-size=16 \
	--embed-dim=256 \
	--num-heads=8 \
    --seq-len=13290  \
    --h5-file="/data/ukbb/net_input/all_gwas_v6.h5" \
    --test-frac=0.3 \
	--devices=0,1,2 \
    --encoder-snapshot=/data/ukbb/v2_snapshots/script_hyena_encoder_ld_learnpos_all_gwas.pt \
    --snapshot-path=/data/ukbb/v2_snapshots/script_hyena_classifier_all_gwas_learnpos_nopretraining.pt \
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
