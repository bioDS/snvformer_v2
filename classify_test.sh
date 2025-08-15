#!/usr/bin/bash
# torchrun --standalone --nproc-per-node=2 classify_test.py \
OMP_NUM_THREADS=1 python classify_test.py \
	--num-layers=8 \
	--batch-size=16 \
	--embed-dim=128 \
	--num-heads=8 \
    --seq-len=13290 \
    --h5-file="/data/ukbb/net_input/all_gwas.h5" \
    --test-frac=0.3 \
	--devices=0 \
    --classifier-snapshot=/data/ukbb/v2_snapshots/script_classifier.pt \
