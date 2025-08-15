#!/usr/bin/bash
# julia ./custom_logic_get_params.jl  | tee logic_best_params.txt
julia ./custom_logic_get_params.jl --output-file="logic_model_grid_search_all_gwas.jld2" --h5-dataset="/data/ukbb/net_input/all_gwas.h5" | tee logic_best_params_all_gwas.txt
julia ./custom_logic_get_params.jl --output-file="logic_model_grid_search_gwas_ldprune_320.jld2" --h5-dataset="/data/ukbb/net_input/gwas_ldprune_320.h5" | tee logic_best_params_gwas_ldprune_320.txt
julia ./custom_logic_get_params.jl --output-file="logic_model_grid_search_tin_fixed_order.jld2" --h5-dataset="/data/ukbb/net_input/tin_fixed_order.h5" | tee logic_best_params_tin_fixed_order.txt
julia ./custom_logic_get_params.jl --output-file="logic_model_grid_search_genotyped_p1e-1.jld2" --h5-dataset="/data/ukbb/net_input/genotyped_p1e-1.h5" | tee logic_best_params_genotyped_p1e-1.txt
julia ./custom_logic_get_params.jl --output-file="logic_model_grid_search_ld_full_r00001.jld2" --h5-dataset="/data/ukbb/net_input/ld_full_r00001.h5" | tee logic_best_params_ld_full_r00001.txt

