#!/usr/bin/env Rscript
library(argparse)
library(ggplot2)
library(dplyr)
library(pROC)

source("pint_common_functions.R")

get_results <- function(effect_sets, use_inds, depth) {
    # Test matrix using all proposed interactions:
    main_effects <- effect_sets$main$effects
    intercept <- effect_sets$intercept
    if (is.nan(intercept)) {
        intercept <- 0.0
    }

    test_Y <- urate_vec[use_inds]

    test_wide_tok_mat <- wide_tok_mat[use_inds,]
    test_Z_main <- test_wide_tok_mat[,main_effects$i]
    test_Z <- test_Z_main
    beta <- main_effects$strength


    if (depth >= 2) {
        pair_effects <- effect_sets$pairwise$effects
        i = pair_effects$i
        j = pair_effects$j
        mul_cols <- function(a, b) {
            return(test_wide_tok_mat[,a]  * test_wide_tok_mat[,b])
        }
        test_Z_pair <- mapply(mul_cols, i[!is.na(i)], j[!is.na(j)])
        test_Z <- cbind(test_Z, test_Z_pair)
        beta <- c(beta, pair_effects$strength)
    }

    # TODO: untested
    if (depth >= 3) {
        trip_effects <- effect_sets$triple$effects
        i <- trip_effects$i
        j <- trip_effects$j
        k <- trip_effects$k
        mul_cols <- function(a, b, c) {
            return(test_wide_tok_mat[,a]  * test_wide_tok_mat[,b] * test_wide_tok_mat[,c])
        }
        test_Z_trip <- mapply(mul_cols, i, j, k)
        test_Z <- cbind(test_Z, test_Z_trip)
        beta <- c(beta, trip_effects$strength)
    }

    # print now so we see them all together at the end of the log
    pred_test_Y <- as.numeric(test_Z %*% beta + intercept)
    pred_test_err <- mean((pred_test_Y - test_Y)^2) # mean squared error

    # MSE of mean for comparison
    mean_err <- mean((test_Y - mean(train_urate_vec))^2)

    # ROC using urate to predict gout.
    use_gout <- gout_vec[use_inds]
    roc_score <- roc(use_gout, pred_test_Y)

    summary <- data.frame(
        mean_err = mean_err,
        pred_test_err = pred_test_err
    )
    results <- list(
        summary = summary,
        roc = roc_score
    )
    return(results)
}

parser <- ArgumentParser(description='Summarise results from a previous run of Pint')
parser$add_argument('--data-dir', default='/data/ukbb/net_input/', type="character", help='Directory containing HDF5 datasets')
# parser$add_argument('--dataset', default='all_gwas.h5', metavar='d', type="character", help='HDF5 dataset to load')
# parser$add_argument('--results-file', default="pint-all_gwas.h5-depth-2-max_nz-10000-use_hier-true-geno-true-pheno-true-logurate-false.rds", type="character", help='regression results are stored here')
parser$add_argument('--dataset', default='ld_full_r00001.h5', metavar='d', type="character", help='HDF5 dataset to load')
parser$add_argument('--results-file', default="pint-ld_full_r00001.h5-depth-2-max_nz-500-use_hier-false-geno-true-pheno-true-logurate-true.rds", type="character", help='regression results are stored here')
parser$add_argument('--pheno-csv', default='/data/ukbb/net_input/phenos.csv', metavar='p', type="character", help='phenos.csv file path')
args <- parser$parse_args()
results_file <- args$results_file
ds_path <- paste0(args$data_dir, args$dataset)

all_data <- get_data(ds_path, args$pheno_csv)
wide_tok_mat <- all_data$wide_tok_mat
wide_tok_mat <- all_data$wide_tok_mat
use_phenos <- all_data$use_phenos
urate_vec <- all_data$urate_vec
gout_vec <- all_data$gout_vec
train_inds <- all_data$train_inds
test_inds <- all_data$test_inds
train_urate_vec <- urate_vec[train_inds]

all_results <- readRDS(results_file)

# start with summary of single (main-only) run
# plot distribution of strengths
print("all/significant main effects only")
single_estimate <- all_results$outputs$single_pint_output$estimate_unbiased
main_effects <- single_estimate$main$effects
ggplot(data.frame(main_effects), aes(strength)) +
    geom_histogram(binwidth = 0.01)

length(main_effects$strength)
mean(abs(main_effects$strength))
sd_abs <- sd(abs(main_effects$strength))
sig1_main <- main_effects %>% filter(abs(strength) >= sd_abs)
nonsig_main <- main_effects %>% filter(abs(strength) < sd_abs)
mean_nonsig_effect <- mean(nonsig_main$strength)
sig1_intercept <- single_estimate$intercept + mean_nonsig_effect

sig1_effects <- list(
    effects=sig1_main
)
all_sig1 <- list(
    main=sig1_effects,
    intercept=sig1_intercept
)
print(all_sig1)

single_est_results <- get_results(single_estimate, test_inds, 1)
print(sprintf("single_est_results: %f", single_est_results$roc$auc))
all_sig1_results <- get_results(all_sig1, test_inds, 1)
print(sprintf("all_sig1_roc: %f", all_sig1_results$roc$auc))

# pairwise sig results
print("all/significant pairwise effects")
full_estimate <- all_results$outputs$pint_output$estimate_unbiased
main_effects <- full_estimate$main$effects
pair_effects <- full_estimate$pairwise$effects
combined_effect_strengths <- c(main_effects$strength, pair_effects$strength)
# ggplot(data.frame(list(strength=combined_effect_strengths)), aes(strength)) +
#     geom_histogram(binwidth = 0.01)
# ggplot(data.frame(list(strength=combined_effect_strengths)), aes(strength)) +
#     geom_density()
sd_abs <- sd(abs(combined_effect_strengths))
sig1_main <- main_effects %>% filter(abs(strength) >= sd_abs)
nonsig_main <- main_effects %>% filter(abs(strength) < sd_abs)
sig1_pair <- pair_effects %>% filter(abs(strength) >= sd_abs)
nonsig_pair <- pair_effects %>% filter(abs(strength) < sd_abs)
mean_nonsig_effect <- mean(nonsig_main$strength) + mean(nonsig_pair$strength)
sig1_intercept <- full_estimate$intercept + mean_nonsig_effect

all_sig1 <- list(
    main = list(effects = sig1_main),
    pairwise = list(effects = sig1_pair),
    intercept=sig1_intercept
)

if (length(sig1_pair$strength) == 0) {
    use_depth <- 1
} else {
    use_depth <- 2
}
print(all_sig1)

full_est_results <- get_results(full_estimate, test_inds, 2)
print(sprintf("full_est_results: %f", full_est_results$roc$auc))
full_sig1_results <- get_results(all_sig1, test_inds, use_depth)
print(sprintf("full_sig_results: %f", full_sig1_results$roc$auc))


genetic_sig_pair <- sig1_pair[c(3,5,8),]
gene_pair_sig1 <- list(
    main = list(effects = sig1_main),
    pairwise = list(effects = genetic_sig_pair),
    intercept=sig1_intercept
)

if (length(sig1_pair$strength) == 0) {
    use_depth <- 1
} else {
    use_depth <- 2
}
print(gene_pair_sig1)

full_est_results <- get_results(full_estimate, test_inds, 2)
print(sprintf("full_est_results: %f", full_est_results$roc$auc))
full_sig1_results <- get_results(gene_pair_sig1, test_inds, use_depth)
main_only_sig1_results <- get_results(gene_pair_sig1, test_inds, 1)
print(sprintf("full_sig_results: %f", full_sig1_results$roc$auc))
print(sprintf("full_sig_results: %f", main_only_sig1_results$roc$auc))


# report top 100 effects by predicted strength, use 1,000 for plots.
me <- full_estimate$main$effects
me$j <- NA
pe <- full_estimate$pairwise$effects
full_alle <- rbind(me,pe)
full_alle <- full_alle[order(-abs(full_alle$strength)),]

# use only top 1000 effects by strength
alle <- head(full_alle, n = 1000)

print("top 100 effects by strength")
print(head(alle, n = 100))

# get snvs only
library(stringr)
all_rs <- str_extract_all(c(alle$i, alle$j), regex("rs[0-9]+"))
tmp <- all_rs[sapply(all_rs, length) >= 1] # remove empty strings
tmp2 <- lapply(tmp, `[[`, 1) # first element of duplicated list
tmp3 <- tmp2[!is.na(tmp2)] # remove NA
tmp3 <- unique(tmp3)
library("rsnps")
ncbi_snps <- ncbi_snp_query(tmp3)
snp_genes <- ncbi_snps$gene
print(snp_genes)

# find cumulative abs effect strength for genes
ta <- alle
ta <- ta |> mutate(snp_id_i = str_extract(i, regex("rs[0-9]+"))) |>
    mutate(snp_id_j = str_extract(j, regex("rs[0-9]+")))

snpi_tibble <- ncbi_snps[ta[32,]$snp_id_i == ncbi_snps$query,]

get_gene <- function(snp_id) {
    # print(snp_id)
    if (snp_id %in% ncbi_snps$query) {
        snpi_tibble <- ncbi_snps[snp_id == ncbi_snps$query,][1,] # search string might have had duplicates
        return(snpi_tibble$gene)
    } else {
        return(NA)
    }
}

# ta2 <- ta |> mutate(snp_i_gene = get_gene(snp_id_i))
ta2 <- ta
ta2$snp_i_gene <- sapply(ta$snp_id_i, get_gene)
ta2$snp_j_gene <- sapply(ta$snp_id_j, get_gene)

found_genes <- unique(c(ta2$snp_i_gene, ta2$snp_j_gene))
found_genes <- found_genes[!is.na(found_genes)]
found_genes <- found_genes[sapply(found_genes, nchar) >= 1] # remove empty string

gi <- found_genes[1]
sum_strength_for_gene <- function(gi) {
    return(sum(ta2 |> filter(snp_i_gene == gi | snp_j_gene == gi) |> select(strength) |> sapply(abs)))
}
mean_strength_for_gene <- function(gi) {
    return(mean(ta2 |> filter(snp_i_gene == gi | snp_j_gene == gi) |> select(strength) |> sapply(abs)))
}
sum_strength_for_gene(gi)
gene_df <- data.frame(gene_name=found_genes)
gene_df$sum_strength <- sapply(gene_df$gene_name, sum_strength_for_gene)
gene_df$mean_strength <- sapply(gene_df$gene_name, mean_strength_for_gene)

# plot distribution of gene strength scores
sum_strength_dist <- ggplot(data.frame(list(strength=gene_df$sum_strength)), aes(strength)) +
    geom_density() + theme_bw()
ggsave(paste0("figures/", results_file, "_sum_strength_dist.pdf"), sum_strength_dist, width = 4, height = 4)

mean_strength_dist <- ggplot(data.frame(list(strength=gene_df$mean_strength)), aes(strength)) +
    geom_density() + theme_bw()
ggsave(paste0("figures/", results_file, "_mean_strength_dist.pdf"), mean_strength_dist, width = 4, height = 4)

print("saving results")
# save summary results
summary_results_file <- paste0(results_file, "_summary.rds")
saveRDS(
        list(gene_df=gene_df,
        all_effects=ta2,
        single_est_results=single_est_results,
        single_sig1_results=all_sig1_results,
        full_est_results=full_est_results, # main + pairwise
        full_sig1_results=full_sig1_results # main + pairwise
        ),
    file=summary_results_file
)

print("genes by sum(abs(effect)):")
print(gene_df[order(-gene_df$sum_strength),])

print("genes by mean(abs(effect)):")
print(gene_df[order(-gene_df$mean_strength),])

# plot rocs
single_roc  <-
    ggroc(single_est_results$roc) +
    theme_bw() +
    ggtitle(paste0('ROC Curve ', '(AUC = ', round(single_est_results$roc$auc, 3), ')')) +
    geom_segment(x = -1, y = 0, xend = 0, yend = 1, color = "#4c72b0")

full_roc  <-
    ggroc(full_est_results$roc) +
    theme_bw() +
    ggtitle(paste0('ROC Curve ', '(AUC = ', round(full_est_results$roc$auc, 3), ')')) +
    geom_segment(x = -1, y = 0, xend = 0, yend = 1, color = "#4c72b0")

ggsave(paste0("figures/", results_file, "_pint_single_roc.pdf"), single_roc, width = 4, height = 4)
ggsave(paste0("figures/", results_file, "_pint_full_roc.pdf"), full_roc, width = 4, height = 4)