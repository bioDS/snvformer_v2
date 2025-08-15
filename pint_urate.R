#!/usr/bin/env Rscript
library(rhdf5)
library(hdf5r)
library(argparse)
library(Pint)
library(dplyr)
library(pROC)
library(ggplot2)

source("pint_common_functions.R")

# TODO: add relevant phenotypes to wide_tok_mat as binary classes

parser <- ArgumentParser(description='Fit UKBB hdf5 dataset with Pint to predict urate from genotype')
parser$add_argument('--data-dir', default='/data/ukbb/net_input/', type="character", help='Directory containing HDF5 datasets')
parser$add_argument('--dataset', default='gwas_ldprune_320.h5', metavar='d', type="character", help='HDF5 dataset to load')
parser$add_argument('--pheno-csv', default='/data/ukbb/net_input/phenos.csv', metavar='p', type="character", help='phenos.csv file path')
parser$add_argument('--depth', default=2, type='integer', help='depth of interactions to search for, 1, 2, or 3')
parser$add_argument('--max-nz-beta', default=100, type='integer', help='maximum number of non-zero effects')
parser$add_argument('--results-file', default='pint_test_output.rds', type="character", help='regression results are stored here')
parser$add_argument('--use-bmi', default=FALSE, action="store_true")
parser$add_argument('--use-age', default=FALSE, action="store_true")
parser$add_argument('--use-sex', default=FALSE, action="store_true")
parser$add_argument('--use-geno', default=FALSE, action="store_true")
parser$add_argument('--use-hierarchy', default=FALSE, action="store_true")
parser$add_argument('--log-urate', default=FALSE, action="store_true")
parser$add_argument('--limit-rows', default=-1, type='integer', help='maximum number of samples to consider')
args <- parser$parse_args()
ds_path <- paste0(args$data_dir, args$dataset)

use_hierarchy <- args$use_hierarchy

all_data <- get_data(ds_path, args$pheno_csv, args$use_bmi, args$use_age, args$use_sex, args$limit_rows, args$use_geno)
wide_tok_mat <- all_data$wide_tok_mat
use_phenos <- all_data$use_phenos
urate_vec <- all_data$urate_vec
if (args$log_urate) {
    urate_vec <- log(urate_vec)
}
gout_vec <- all_data$gout_vec
train_inds <- all_data$train_inds
test_inds <- all_data$test_inds
train_mat <- wide_tok_mat[train_inds,]
train_urate_vec <- urate_vec[train_inds]

single_pint_output <- interaction_lasso(
    train_mat,
    train_urate_vec,
    depth=1,
    approximate_hierarchy=use_hierarchy,
    verbose=TRUE,
    num_threads=32,
    use_intercept=TRUE,
    max_nz_beta=args$max_nz_beta,
    estimate_unbiased=TRUE,
)

pint_output <- interaction_lasso(
    train_mat,
    train_urate_vec,
    depth=args$depth,
    approximate_hierarchy=use_hierarchy,
    verbose=TRUE,
    num_threads=10,
    use_intercept=TRUE,
    max_nz_beta=args$max_nz_beta,
    estimate_unbiased=TRUE,
)

get_results <- function(use_pint_output, use_inds, depth) {
    # Test matrix using all proposed interactions:
    main_effects <- use_pint_output$estimate_unbiased$main$effects
    intercept <- use_pint_output$estimate_unbiased$intercept

    test_Y <- urate_vec[use_inds]

    test_wide_tok_mat <- wide_tok_mat[use_inds,]
    test_Z_main <- test_wide_tok_mat[,main_effects$i]
    test_Z <- test_Z_main
    beta <- main_effects$strength


    if (depth >= 2) {
        pair_effects <- use_pint_output$estimate_unbiased$pairwise$effects
        i = pair_effects$i
        j = pair_effects$j
        mul_cols <- function(a, b) {
            return(test_wide_tok_mat[,a]  * test_wide_tok_mat[,b])
        }
        test_Z_pair <- mapply(mul_cols, i, j)
        test_Z <- cbind(test_Z, test_Z_pair)
        beta <- c(beta, pair_effects$strength)
    }

    # TODO: untested
    if (depth >= 3) {
        trip_effects <- use_pint_output$estimate_unbiased$triple$effects
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
        output = use_pint_output,
        roc = roc_score,
        train_inds=train_inds,
        test_inds=test_inds
    )
    return(results)
}

single_est_results  <- get_results(single_pint_output, test_inds, 1)
single_run_train_results <- get_results(single_pint_output, train_inds, 1)
full_run_test_results  <- get_results(pint_output, test_inds, args$depth)
full_run_train_results <- get_results(pint_output, train_inds, args$depth)

sprintf("single prediction err: %f (test), %f (train), gout auroc: %f (test)",
    single_est_results$summary$pred_test_err,
    single_run_train_results$summary$pred_test_err,
    single_est_results$roc$auc)
sprintf("full prediction err: %f (test), %f (train), gout auroc: %f (test)",
    full_run_test_results$summary$pred_test_err,
    full_run_train_results$summary$pred_test_err,
    full_run_test_results$roc$auc)

# MSE of mean for comparison
sprintf("mean-of-training-set test err: %f", single_est_results$summary$mean_err)

# true urate for comparison
test_urate_roc <- roc(gout_vec[test_inds], urate_vec[test_inds])
sprintf("auroc of true urate in test set: %f", test_urate_roc$auc)

summary <- data.frame(
    mean_err = single_run_train_results$summary$mean_err,
    full_pred_test_err = full_run_test_results$summary$pred_test_err,
    full_pred_train_err = full_run_train_results$summary$pred_test_err,
    single_pred_test_err = single_est_results$summary$pred_test_err,
    single_pred_train_err = single_run_train_results$summary$pred_test_err
)
outputs <- list(
    pint_output = pint_output,
    single_pint_output = single_pint_output
)
results <- list(
    summary = summary,
    single_run_test_results = single_est_results,
    single_run_train_results = single_run_train_results,
    full_run_test_results = full_run_test_results,
    full_run_train_results = full_run_train_results,
    test_urate_roc = test_urate_roc,
    outputs = outputs
)


# save results
saveRDS(results, args$results_file)

library(ggplot2)
ag <- readRDS("pint-all_gwas.h5-depth-2-max_nz-500-use_hier-false.rds")
agh <- readRDS("pint-all_gwas.h5-depth-2-max_nz-10000-use_hier-true.rds")
pds <- agh
# single_roc <- ggroc(pds$single_run_test_results$roc)
# full_roc <- ggroc(pds$full_run_test_results$roc)

single_roc  <-
    ggroc(pds$single_run_test_results$roc) +
    theme_bw() +
    ggtitle(paste0('ROC Curve ', '(AUC = ', round(pds$single_run_test_results$roc$auc, 3), ')')) +
    geom_segment(x = -1, y = 0, xend = 0, yend = 1, color = "#4c72b0")

full_roc  <-
    ggroc(pds$full_run_test_results$roc) +
    theme_bw() +
    ggtitle(paste0('ROC Curve ', '(AUC = ', round(pds$full_run_test_results$roc$auc, 3), ')')) +
    geom_segment(x = -1, y = 0, xend = 0, yend = 1, color = "#4c72b0")

ggsave("figures/pint_single_roc.pdf", single_roc, width = 4, height = 4)
ggsave("figures/pint_full_roc.pdf", full_roc, width = 4, height = 4)

