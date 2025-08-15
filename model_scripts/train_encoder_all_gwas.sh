#!/usr/bin/bash
# remove short epochs for final run
#!/usr/bin/bash
torchrun \
  --master_addr="localhost" \
  --master_port="12429" \
   --nnodes=1 \
   --nproc_per_node=3 \
train_encoder.py \
    --encoder-type="classic" \
	--total-epochs=2 \
	--num-layers=8 \
	--batch-size=1 \
	--embed-dim=128 \
	--num-heads=8 \
    --seq-len=13290 \
    --h5-file="/data/ukbb/net_input/all_gwas.h5" \
    --test-frac=0.3 \
	--devices=0,1,2 \
    --snapshot-path=/data/ukbb/v2_snapshots/script_encoder.pt \
    --warmup-batches=500 \
    --warmup-min-lr=1e-10 \
    --warmup-max-lr=1e-7 \
    --main-scheduler-step-size=1000 \
    --main-scheduler-gamma=0.95 \
    --report-on-batch=100 \
    --gradual-length \
    --no-augment-data