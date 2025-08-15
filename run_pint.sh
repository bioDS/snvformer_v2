#!/usr/bin/bash
run_if_not_exists() {
    ds=$1
    depth=$2
    max_nz=$3
    use_hierarchy=$4
    use_pheno=$5
    use_geno=$6
    log_urate=$7
    rfile="pint-$ds-depth-$depth-max_nz-$max_nz-use_hier-$use_hierarchy-geno-$use_geno-pheno-$use_pheno-logurate-$log_urate.rds"
    if [ ! -f $rfile ]; then
        args="--results-file=$rfile --dataset=$ds --depth=$depth --max-nz-beta=$max_nz"
        if [ $use_hierarchy == "true" ]; then
            args="$args --use-hierarchy"
        fi
        if [ $use_pheno == "true" ]; then
            args="$args --use-age --use-sex --use-bmi"
        fi
        if [ $use_geno == "true" ]; then
            args="$args --use-geno"
        fi
        if [ $log_urate == "true" ]; then
            args="$args --log-urate"
        fi
        ./pint_urate.R $args | tee "${rfile}.log"
    else
        echo "file $rfile already exists"
    fi
}

# quick test
run_if_not_exists gwas_ldprune_320.h5 2 100 true true true false
run_if_not_exists gwas_ldprune_320.h5 2 100 true true true true
# pheno only
run_if_not_exists gwas_ldprune_320.h5 2 1000 false true false false
run_if_not_exists gwas_ldprune_320.h5 2 1000 false true false true
# geno only
run_if_not_exists gwas_ldprune_320.h5 2 1000 true false true false
run_if_not_exists all_gwas.h5 2 10000 true        false true false
run_if_not_exists ld_full_r00001.h5 2 10000 true  false true false
run_if_not_exists ld_full_r00001.h5 2 500 false   false true false
run_if_not_exists gwas_ldprune_320.h5 2 500 false false true false
run_if_not_exists all_gwas.h5 2 500 false         false true false
run_if_not_exists gwas_ldprune_320.h5 2 1000 true false true true
run_if_not_exists all_gwas.h5 2 10000 true        false true true
run_if_not_exists ld_full_r00001.h5 2 10000 true  false true true
run_if_not_exists ld_full_r00001.h5 2 500 false   false true true
run_if_not_exists gwas_ldprune_320.h5 2 500 false false true true
run_if_not_exists all_gwas.h5 2 500 false         false true true
# both
run_if_not_exists gwas_ldprune_320.h5 2 1000 true true true false
run_if_not_exists all_gwas.h5 2 10000 true        true true false
run_if_not_exists ld_full_r00001.h5 2 10000 true  true true false
run_if_not_exists ld_full_r00001.h5 2 500 false   true true false
run_if_not_exists gwas_ldprune_320.h5 2 500 false true true false
run_if_not_exists all_gwas.h5 2 500 false         true true false
run_if_not_exists gwas_ldprune_320.h5 2 1000 true true true true
run_if_not_exists all_gwas.h5 2 10000 true        true true true
run_if_not_exists ld_full_r00001.h5 2 10000 true  true true true
run_if_not_exists ld_full_r00001.h5 2 500 false   true true true
run_if_not_exists gwas_ldprune_320.h5 2 500 false true true true
run_if_not_exists all_gwas.h5 2 500 false         true true true

# pheno subsets
ds="gwas_ldprune_320.h5" # doesn't matter
depth=2
max_nz=1000
use_hierarchy=false
rfile="pint-$ds-depth-$depth-max_nz-$max_nz-use_hier-$use_hierarchy-geno-$use_geno-pheno"
args="--dataset=$ds --depth=$depth --max-nz-beta=$max_nz"
# commented out to avoid re-running
#./pint_urate.R $args --results-file=$rfile-age_only.rds --use-age | tee "${rfile}-age.log"
#./pint_urate.R $args --results-file=$rfile-sex_only.rds --use-sex | tee "${rfile}-sex.log"
#./pint_urate.R $args --results-file=$rfile-bmi_only.rds --use-bmi | tee "${rfile}-bmi.log"
#./pint_urate.R $args --results-file=$rfile-agesex_only.rds --use-age --use-sex | tee "${rfile}-agesex.log"
#./pint_urate.R $args --results-file=$rfile-agebmi_only.rds --use-age --use-bmi | tee "${rfile}-agebmi.log"
#./pint_urate.R $args --results-file=$rfile-bmisex_only.rds --use-bmi --use-sex | tee "${rfile}-bmisex.log"