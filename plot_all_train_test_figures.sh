#!/usr/bin/bash

use_py="$HOME/mambaforge/envs/torch2/bin/python"
snap_dir="/data/ukbb/v2_snapshots"
log_dir="./output"
fig_dir="./figures"

function plot_pt() {
    $use_py train_test_plot_from_log.py "$snap_dir/$1.pt" "$fig_dir/$2"
}

function plot_txt() {
    $use_py train_test_plot_from_log.py "$log_dir/$1.txt" "$fig_dir/$2"
}

# plot_pt script_linformer_classifier_ld_full_learnpos_nopretraining linformer_ld_full_learnpos_classifier_no_pretraining
# plot_pt script_linformer_classifier_ld_full_learnpos linformer_ld_full_learnpos_classifier_pretrained
# plot_pt script_linformer_sex_classifier_ld_full_learnpos_nopretraining linformer_sex_classifier_no_pretraining
# plot_txt linformer_learnable_pos linformer_ld_full_learnpos_encoder_pretraining
# plot_pt script_mlp mlp

# # snvformer variations
# plot_pt snvformer_class_tok_out snvformer_class_tok_out
# plot_pt snvformer_currentstuff  snvformer_currentstuff
# plot_pt snvformer.different     snvformer.different
# plot_pt snvformer_encoder       snvformer_encoder
# plot_pt snvformer_gradscaler    snvformer_gradscaler
# plot_pt snvformer_no_pretrain   snvformer_no_pretrain
# plot_pt snvformer_otf           snvformer_otf
# plot_pt snvformer               snvformer
# plot_pt snvformer_simpleout     snvformer_simpleout
# plot_pt snvformer_testing       snvformer_testing
# plot_pt snvformer_test          snvformer_test

function plot_i() {
    i=$1
    ni=${i/$snap_dir/$fig_dir}
    of=${ni/.pt/}
    echo $i $of
    $use_py train_test_plot_from_log.py $i $of
}

N=8
(
for file in /data/ukbb/v2_snapshots/*.pt; do
   ((i=i%N)); ((i++==0)) && wait
    plot_i $file &
done
)