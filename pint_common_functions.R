# TODO: add relevant phenotypes to wide_tok_mat as binary classes
library(rhdf5)
library(dplyr)

get_data <- function(ds_path, phenos_csv, use_bmi=TRUE, use_age=TRUE, use_sex=TRUE, limit_rows=-1, use_geno=TRUE) {
    # h5ls(ds_path)
    num_toks <- h5read(ds_path, '/num_toks')
    positions <- h5read(ds_path, '/positions')
    if (use_geno) {
        tok_mat <- h5read(ds_path, '/tok_mat')
    } else {
        tok_mat <- matrix(, nrow=402167, ncol=0)
    }
    # b0_names <- h5read(ds_path, '/phenos/block0_items')
    # b1_names <- h5read(ds_path, '/phenos/block0_items')
    # assert(b0_names == "gout")
    # gout = h5read(ds_path, '/phenos/block0_values')
    b2_items <- h5read(ds_path, '/phenos/block2_items')
    b2_values <- h5read(ds_path, '/phenos/block2_values')
    use_eid <- b2_values[1, ]
    tts <- h5read(ds_path, '/tok_to_string') # note that 5: '0'
    # snv_names <- read_delim
    bim_file <- gsub(".h5", ".bim", ds_path)
    bim_contents <- read.table(bim_file)
    snv_names <- bim_contents[,2]

    # we have to read gout status from phenos.csv, R can't handle bitvectors
    phenos_csv <- read.csv(phenos_csv)
    use_phenos <- phenos_csv %>% filter(eid %in% use_eid)
    # we can't use cases without urate measurements
    full_include_rows <- !is.na(use_phenos$urate)
    use_phenos <- use_phenos[!is.na(use_phenos$urate),]

    # convert tok_mat to binary
    # We'll ignore cases other than 5/6/7 (a.k.a 0/1/2)
    if (use_geno) {
        tok_mat_0 <- tok_mat == 5
        rownames(tok_mat_0) <- paste0(snv_names, "-0")
        tok_mat_1 <- tok_mat == 6
        rownames(tok_mat_1) <- paste0(snv_names, "-1")
        tok_mat_2 <- tok_mat == 7
        rownames(tok_mat_2) <- paste0(snv_names, "-2")
        wide_tok_mat <- as.matrix(t(rbind(tok_mat_0, tok_mat_1, tok_mat_2))) # matrix was still transposed from input
        wide_tok_mat <- wide_tok_mat[full_include_rows, ]
        mode(wide_tok_mat) <- "integer"
        # colnames(wide_tok_mat) <- c(seq_len(ncol(wide_tok_mat)))
    } else {
        wide_tok_mat <- matrix(, nrow=nrow(use_phenos), ncol=0)
    }
    # add demographic information as binary classes to matrix
    ## Age
    num_age_classes <- 10
    min_age <- min(use_phenos$age)
    max_age <- max(use_phenos$age)
    age_step_size <- (max_age - min_age) / num_age_classes

    age_lim <- min_age
    age_mat <- c()
    for (step in 1:num_age_classes) {
        age_lim <- age_lim + age_step_size
        age_vec <- as.matrix(use_phenos$age >= age_lim)
        colnames(age_vec) <- paste0("age_gte_", age_lim)
        age_mat <- cbind(age_mat, age_vec)
    }
    ## Sex
    sex_vec <- as.matrix(use_phenos$sex == "Female")
    colnames(sex_vec) <- "Female"
    ## BMI
    num_bmi_classes <- 10
    use_phenos[is.na(use_phenos$bmi),]$bmi <- 0.0
    min_bmi <- min(use_phenos[use_phenos$bmi > 0,]$bmi)
    max_bmi <- max(use_phenos[use_phenos$bmi > 0,]$bmi)
    bmi_step_size <- (max_bmi - min_bmi) / num_bmi_classes

    bmi_lim <- min_bmi
    bmi_mat <- c()
    for (step in 1:num_bmi_classes) {
        bmi_lim <- bmi_lim + bmi_step_size
        bmi_vec <- as.matrix(use_phenos$bmi >= bmi_lim)
        colnames(bmi_vec) <- paste0("bmi_gte_", bmi_lim)
        bmi_mat <- cbind(bmi_mat, bmi_vec)
    }

    if (use_bmi) {
        wide_tok_mat <- cbind(bmi_mat, wide_tok_mat)
    }
    if (use_sex) {
        wide_tok_mat <- cbind(sex_vec, wide_tok_mat)
    }
    if (use_age) {
        wide_tok_mat <- cbind(age_mat, wide_tok_mat)
    }

    # Train/Test split
    ## make sure test set is balanced
    gout_inds <- which(use_phenos["gout"][[1]])
    non_gout_inds <- which(!use_phenos["gout"][[1]])
    final_gout_train_ind <- 0.7 * length(gout_inds)
    gout_train_inds <- gout_inds[1:final_gout_train_ind]
    nongout_train_inds <- non_gout_inds[1:final_gout_train_ind]
    gout_test_inds <- gout_inds[final_gout_train_ind:length(gout_inds)]
    nongout_test_inds <- non_gout_inds[final_gout_train_ind:length(gout_inds)]
    pretrain_inds <- non_gout_inds[length(gout_inds):length(non_gout_inds)]
    full_train_inds <- c(gout_train_inds, nongout_train_inds, pretrain_inds)
    test_inds <- c(gout_test_inds, nongout_test_inds)


    # full_include_row_indices <- which(full_include_rows)
    # final_train_ind <- ceiling(0.7 * length(full_include_row_indices))
    # full_train_inds <- c(1:final_train_ind)
    # test_inds <- c((final_train_ind+1):length(full_include_row_indices))

    if (limit_rows > 0) {
        limit_rows <- limit_rows
        train_inds <- full_train_inds[1:limit_rows]
    } else {
        train_inds <- full_train_inds
    }

    urate_vec <- as.matrix(use_phenos$urate)
    gout_vec <- as.matrix(use_phenos$gout)

    all_data <- list(
        wide_tok_mat=wide_tok_mat,
        use_phenos=use_phenos,
        urate_vec=urate_vec,
        gout_vec=gout_vec,
        train_inds=train_inds,
        test_inds=test_inds
    )
    return(all_data)
}