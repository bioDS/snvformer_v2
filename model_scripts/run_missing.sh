#!/usr/bin/bash
to_run="
    train_urate_linear.sh
    train_mlp.sh
    train_hyena_classifier_nopretraining_all_gwas.sh
    train_hyena_classifier_nopretraining.sh
    snvformer.sh
    train_classifier_tin_et_al.sh
    train_classifier.overfitting.sh
    train_classifier_overfitting_augmented.sh
    train_encoder_all_gwas.sh
    train_urate.sh
    train_sex_classifier.sh
    train_linformer.sh
    train_linformer_classifier_nopretraining.sh
    train_linformer_classifier.sh
    train_classifier.sh
    train_encoder.sh
    train_classifier_tiny.sh
    train_combo.sh
    train_encoder_tiny.sh
    train_height_linear.sh
    train_height_manyheads.sh
    train_height.sh
    train_hyena_encoder.sh
    train_hyena_classifier_nopretraining_all_gwas.sh
    train_hyena_classifier_nopretraining.sh
    train_hyena_classifier_pretrained_all_gwas.sh
    train_hyena_encoder_v2.sh
    train_hyena_encoder_all_unimputed_v6.sh
    train_hyena_encoder_v2_all_gwas.sh
    train_linformer_bigk.sh
    train_manyhead_encoder.sh
    train_middle_token_hyena_encoder.sh
    train_nexttoken_hyena_encoder.sh
"

cd /home/kieran_elmes/gout_transformer_v2

for script in $to_run; do
    echo bash "model_scripts/$script" | tee "output/$script.txt"
    
    bash "model_scripts/$script" | tee -a "output/$script.txt"
done