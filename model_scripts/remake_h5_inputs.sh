#!/usr/bin/bash

data_dir=/data/ukbb/net_input

v1_files="genotyped_p1e-1"
v2_files="genotyped_p1e-1 all_gwas"
v3_files="genotyped_p1e-1 all_gwas"
v5_files="all_gwas gwas_ldprune_320 ld_full_r00001 genotyped_p1e-1 tin_fixed_order"
v6_files="ld_full_r00001 all_gwas genotyped_p1e-1 tin_fixed_order"

for file in $v1_files; do
    if [ ! -f $data_dir/${file}_v1.h5 ]; then
        python bed_to_hdf5.py --plink_base=${file} --encoding=1 --h5_file=$data_dir/${file}_v1.h5
    fi
done
for file in $v2_files; do
    if [ ! -f $data_dir/${file}_v2.h5 ]; then
        python bed_to_hdf5.py --plink_base=${file} --encoding=2 --h5_file=$data_dir/${file}_v2.h5
    fi
done
for file in $v3_files; do
    if [ ! -f $data_dir/${file}_v3.h5 ]; then
        python bed_to_hdf5.py --plink_base=${file} --encoding=3 --h5_file=$data_dir/${file}_v3.h5
    fi
done
for file in $v5_files; do
    if [ ! -f $data_dir/${file}.h5 ]; then
        python bed_to_hdf5.py --plink_base=${file} --encoding=5 --h5_file=$data_dir/${file}.h5
    fi
done
for file in $v6_files; do
    if [ ! -f $data_dir/${file}_v6.h5 ]; then
        python bed_to_hdf5.py --plink_base=${file} --encoding=6 --h5_file=$data_dir/${file}_v6.h5
    fi
done