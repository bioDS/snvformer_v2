using JLD2
using DataLoaders
using DataFrames
using InvertedIndices

function pandas_hdf5_block(items, values)
    column_names = [filter(x -> x != '\0', cn) for cn in items]
    block_df = DataFrame(values', Symbol.(column_names)) # Transpose the data to have nDataPoints x nHeaders
    block_df
end

function pandas_hdf5_to_dataframe(phenos_hdf)
    all_dfs = []
    for block_no in ["0", "1", "2"] #TODO: 4 is different
        block_items_name = "block" * block_no * "_items"
        block_values_name = "block" * block_no * "_values"
        block_df = pandas_hdf5_block(phenos_hdf[block_items_name], phenos_hdf[block_values_name])
        push!(all_dfs, block_df)
    end
    hcat(all_dfs...)
end

function split_pheno(pheno, num_splits)
    min_pheno = minimum(pheno[(!).(isnan.(pheno))])
    max_pheno = maximum(pheno[(!).(isnan.(pheno))])
    div_size = (max_pheno - min_pheno) / num_splits
    all_vecs = zeros(Int32, length(pheno), num_splits)
    for vi in 1:num_splits
        all_vecs[1:end,vi] = pheno .>= div_size*vi
    end
    return transpose(all_vecs)
end

function combine_pheno_matrix(binary_tok_mat, phenos, male; num_pheno_splits=10, binarise_phenos=False)
    if binarise_phenos
        age = split_pheno(phenos[:,"age"], num_pheno_splits)
        bmi = split_pheno(phenos[:,"bmi"], num_pheno_splits)
    else
        age   = reshape(phenos[:,"age"], 1,:)
        bmi   = reshape(phenos[:,"bmi"], 1,:)
    end
    male = reshape(male, 1, :)

    prepend_pheno_mat = cat(male, age, bmi, binary_tok_mat, dims=1)
    return prepend_pheno_mat
end

function load_data(args)
    h5_dataset = args["h5-dataset"]
    f = jldopen(h5_dataset, "r")
    chromosomes = f["chromosomes"]
    positions = f["positions"]
    tok_to_string = f["tok_to_string"]
    string_to_tok = f["string_to_tok"]
    num_toks = f["num_toks"]
    tok_mat = f["tok_mat"]
    phenos_hdf = f["phenos"]
    phenos = pandas_hdf5_to_dataframe(phenos_hdf)
    male = Bool.(f["male"])
    close(f)
    train_inds, test_inds, verify_inds, unused_inds = get_train_test_verify(phenos)
    if args["use-pretrain"]
        train_inds = cat(train_inds, unused_inds, dims=1)
    end
    train_inds = shuffle(train_inds)
    # create full binary matrix
    tok_mat_0 = tok_mat .== 5
    tok_mat_1 = tok_mat .== 6
    tok_mat_2 = tok_mat .== 7
    binary_tok_mat = Float32.(vcat(tok_mat_0, tok_mat_1, tok_mat_2))
    if args["single-input-matrix"]
        train_mat = combine_pheno_matrix(binary_tok_mat[:,train_inds], phenos[train_inds,:], male[train_inds]; num_pheno_splits=args["num-pheno-splits"], binarise_phenos=args["binarise-phenos"])
        test_mat = combine_pheno_matrix(binary_tok_mat[:,test_inds], phenos[test_inds,:], male[test_inds]; num_pheno_splits=args["num-pheno-splits"], binarise_phenos=args["binarise-phenos"])
        verify_mat = combine_pheno_matrix(binary_tok_mat[:,verify_inds], phenos[verify_inds,:], male[verify_inds]; num_pheno_splits=args["num-pheno-splits"], binarise_phenos=args["binarise-phenos"])
        train_dl = DataLoader((train_mat, phenos.urate[train_inds], phenos.gout[train_inds]), args["batch-size"])
        test_dl = DataLoader((test_mat, phenos.urate[test_inds], phenos.gout[test_inds]), args["batch-size"])
        verify_dl = DataLoader((verify_mat, phenos.urate[verify_inds], phenos.gout[verify_inds]), args["batch-size"])
    else
        train_dl = DataLoader((binary_tok_mat[:,train_inds],
                                phenos.gout[train_inds],
                                phenos.urate[train_inds],
                                phenos.bmi[train_inds],
                                phenos.height[train_inds],
                                phenos.age[train_inds],
                                male[train_inds]),
                            args["batch-size"])
        test_dl = DataLoader((binary_tok_mat[:,test_inds],
                                phenos.gout[test_inds],
                                phenos.urate[test_inds],
                                phenos.bmi[test_inds],
                                phenos.height[test_inds],
                                phenos.age[test_inds],
                                male[test_inds]),
                            args["batch-size"])
        verify_dl = DataLoader((binary_tok_mat[:,verify_inds],
                                phenos.gout[verify_inds],
                                phenos.urate[verify_inds],
                                phenos.bmi[verify_inds],
                                phenos.height[verify_inds],
                                phenos.age[verify_inds],
                                male[verify_inds]),
                            args["batch-size"])
    end
    return train_dl, test_dl, verify_dl, tok_to_string
end

function get_train_test_verify(phenos)
    gout = phenos.gout
    gout_inds = findall(gout)
    non_gout_inds = findall(gout .== false)
    split_ind = Int(ceil((1.0 - args["test-frac"]) * length(gout_inds)))
    train_gout_inds = gout_inds[1:split_ind]
    test_gout_inds = gout_inds[split_ind+1:end]
    train_non_gout_inds = non_gout_inds[1:length(train_gout_inds)]
    test_non_gout_inds = non_gout_inds[length(train_gout_inds)+1:length(train_gout_inds)+length(test_gout_inds)]
    unused_non_gout_inds = non_gout_inds[length(train_gout_inds)+length(test_gout_inds)+1:end]
    train_inds = vcat(train_gout_inds, train_non_gout_inds)
    non_train_inds = vcat(test_gout_inds, test_non_gout_inds)
    # split non_train into 50% test, 50% verify
    test_inds = non_train_inds[1:6:end]
    verify_inds = non_train_inds[Not(1:6:end)]
    unused_inds = unused_non_gout_inds
    return train_inds, test_inds, verify_inds, unused_inds
end