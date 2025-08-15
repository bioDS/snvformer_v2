function run_script() {
    script=$1
    echo "running $script"
    bash $script
}

# ho_search best run
# train_cifar_ae570_00017 TERMINATED 4 7 4 16 0.133063 otf False True cat 4 4 4 False True dual-output 7.12978e-05 True 16 15832.8 0.534001 │
run_script model_scripts/snvformer_hosearch_encoder.sh
run_script model_scripts/snvformer_hosearch.sh
run_script model_scripts/snvformer_hosearch_nopt.sh

# different encodings
run_script model_scripts/snvformer_hosearch_nopt_v1.sh
run_script model_scripts/snvformer_hosearch_nopt_v2.sh
run_script model_scripts/snvformer_hosearch_nopt_v3.sh
run_script model_scripts/snvformer_hosearch_nopt_v6.sh
# different inputs
run_script model_scripts/snvformer_hosearch_nopt_all_gwas.sh
run_script model_scripts/snvformer_hosearch_nopt_gwas_ldfull.sh
run_script model_scripts/snvformer_hosearch_nopt_gwas_ldprune320.sh
run_script model_scripts/snvformer_hosearch_nopt_tinetal.sh
# combo gout/urate output
run_script model_scripts/snvformer_hosearch_nopt_combo.sh


# different encodings
run_script model_scripts/snvformer_otf_nopt_encv1.sh
run_script model_scripts/snvformer_otf_nopt_encv2.sh
run_script model_scripts/snvformer_otf_nopt_encv3.sh
run_script model_scripts/snvformer_otf_nopt_encv6.sh

# snvformer nopt on different files
run_script model_scripts/snvformer_otf_nopt_tinetal.sh # w/ single output + otf
run_script model_scripts/snvformer_otf_nopt_all_gwas.sh # w/ single output + otf
run_script model_scripts/snvformer_otf_nopt_gwas_ldprune320.sh # w/ single output + otf
run_script model_scripts/snvformer_otf_nopt_ldfull.sh # w/ single output + otf

# urate combo
run_script model_scripts/snvformer_otf_urate_combo.sh

## encoders
#run_script model_scripts/snvformer_encoder.sh
#run_script model_scripts/snvformer_chrom_encoder.sh
run_script model_scripts/snvformer_embedpos_encoder.sh # w/ chrom_embedding + position embedding
run_script model_scripts/snvformer_embedpos_encoder_4xffn.sh # w/ chrom_embedding + position embedding

# TODO
# snvformer (otf) w/ data augmentation
run_script model_scripts/snvformer_dualout_otf_augmented.sh
# snvformer otf + deep (nopt)
run_script model_scripts/snvformer_dualout_otf_deep.sh
# snvformer otf + shallow (nopt)
run_script model_scripts/snvformer_dualout_otf_shallow.sh
# snvformer hyena (instead of linformer) nopt
run_script model_scripts/snvformer_dualout_otf_hyena.sh
# snvformer (otf) + gene encoding
run_script model_scripts/snvformer_dualout_otf_gene.sh
# snvformer (otf) + gene encoding (graph)
run_script model_scripts/snvformer_dualout_otf_genegraph.sh
# snvformer (otf) urate output only
# TODO
# snvformer (otf) + combined urate output
# TODO
#snvformer (otf) + double embed dim (but not linformer k)
run_script model_scripts/snvformer_dualout_otf_2xembed_nopt.sh
#snvformer (otf) + tf_init (nopt)
# combine w/ addition instead of concatenation (nopretrain)
run_script model_scripts/snvformer_addpos.sh

# 6-layer k=96, embedding=96 from paper
run_script model_scripts/snvformer_paper_encoder.sh
run_script model_scripts/snvformer_paper.sh

# worth checking first
#run_script model_scripts/snvformer_k96_nopt.sh # w/ dual_output + otf + k=96
run_script model_scripts/snvformer_k128_nopt.sh # w/ dual_output + otf + k=128
run_script model_scripts/snvformer_k256_nopt.sh # w/ dual_output + otf + k=256
#run_script model_scripts/snvformer_dualout_otf.sh # w/ dual output + otf
#run_script model_scripts/snvformer_dualout_gradscaler.sh # w/ dual output + gradscaler + otf
run_script model_scripts/snvformer_dualout_otf_scheduledlr.sh # w/ dual_output, otf, lr 1e-7 -> 1e-4, then *0.95 every 1,000
run_script model_scripts/snvformer_embedpos.sh # w/  + position embedding

# combine all promising differences vs. snvformer
# run_script model_scripts/snvformer_dualout_chrom_4xffn_pretrained_k96

# dual output versions
run_script model_scripts/snvformer_dualout_chrom.sh # w/ dual output + chrom embedding
run_script model_scripts/snvformer_dualout_4x_ffn.sh # w/ dual output + 4x ffn (w/o pretraining)

# N.B. all models use torch_sdp (encoder did not)
run_script model_scripts/snvformer.sh # original version, 1 epoch pre-training
run_script model_scripts/snvformer_no_pretrain.sh # w/o pretraining
# single output versions
run_script model_scripts/snvformer_simpleout.sh # w/ single output
run_script model_scripts/snvformer_otf.sh # w/ single output + otf
run_script model_scripts/snvformer_gradscaler.sh # w/ single output + gradscaler
run_script model_scripts/snvformer_chrom.sh # w/ single output + chrom embedding
run_script model_scripts/snvformer_4x_ffn.sh # w/ single output + 4x ffn (w/o pretraining)
run_script model_scripts/snvformer_4x_ffn_augmented.sh # w/ single output + 4x ffn (w/o pretraining) + 2x data augmentation

# w/ different encoders
run_script model_scripts/snvformer_embedpos_4x_ffn_pretrained.sh # w/ single output + otf + gradscaler + chrom embedding + 4x ffn (w/ pretraining)

# non-snvformer models
#run_script model_scripts/train_encoder_all_gwas.sh
#run_script model_scripts/train_encoder.sh
#run_script model_scripts/train_encoder_tin_et_al.sh
#run_script model_scripts/train_encoder_tiny.sh
#run_script model_scripts/train_hyena_encoder_all_unimputed_v6.sh
#run_script model_scripts/train_hyena_encoder.sh
#run_script model_scripts/train_hyena_encoder_v2_all_gwas.sh
#run_script model_scripts/train_hyena_encoder_v2.sh
#run_script model_scripts/train_manyhead_encoder.sh
#run_script model_scripts/train_middle_token_hyena_encoder.sh
#run_script model_scripts/train_classifier_overfitting_augmented.sh
#run_script model_scripts/train_classifier.overfitting.sh
#run_script model_scripts/train_classifier.sh
#run_script model_scripts/train_classifier_tin_et_al_class_tok.sh
#run_script model_scripts/train_classifier_tin_et_al_hyena.sh
#run_script model_scripts/train_classifier_tin_et_al_linformer.sh
#run_script model_scripts/train_classifier_tin_et_al.sh
#run_script model_scripts/train_classifier_tiny.sh
#run_script model_scripts/train_combo.sh
#run_script model_scripts/train_height_linear.sh
#run_script model_scripts/train_height_manyheads.sh
#run_script model_scripts/train_height.sh
#run_script model_scripts/train_hyena_classifier_nopretraining_all_gwas.sh
#run_script model_scripts/train_hyena_classifier_nopretraining.sh
#run_script model_scripts/train_hyena_classifier_pretrained_all_gwas.sh
#run_script model_scripts/train_linformer_bigk.sh
#run_script model_scripts/train_linformer_classifier_nopretraining.sh
#run_script model_scripts/train_linformer_classifier.sh
#run_script model_scripts/train_linformer.sh
#run_script model_scripts/train_mlp.sh
#run_script model_scripts/train_nexttoken_hyena_encoder.sh
#run_script model_scripts/train_sex_classifier.sh
#run_script model_scripts/train_urate_linear.sh
#run_script model_scripts/train_urate.sh