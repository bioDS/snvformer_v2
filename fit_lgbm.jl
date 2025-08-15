include("data.jl")
using ArgParse
using MLJ
using DataFrames
# using DataFramesMeta
using ScientificTypes
using JLD2
import Libdl
#push!(Libdl.DLLOADPATH, "/home/kel63/src/LightGBM/")
# push!(Libdl.DL_LOAD_PATH, "/home/kel63/src/LightGBM/")
import LightGBM
import LightGBM.MLJInterface.LGBMClassifier
using LossFunctions
using Gadfly, Compose
import Cairo, Fontconfig
using Printf
using Plots
using StatisticalMeasuresBase
include("jl_common.jl")

s = ArgParseSettings()
@add_arg_table! s begin
    "--h5-datadir"
        help = "an option with an argument"
        default = "/data/ukbb/net_input/"
    "--h5-dataset"
        help = "an option with an argument"
        default = "gwas_ldprune_320"
    "--output-file-base", "-o"
        help = "save model and summary statistics here as jld2"
        default = "lgbm_output"
    "--test-frac"
        arg_type = Float64
        default=0.3
    "--batch-size"
        arg_type = Int
        default=16
    "--num-ands"
        arg_type = Int
        default=10
    "--num-linear"
        arg_type = Int
        default=20
    "--num-pheno-thresholds"
        arg_type = Int
        default = 10
    "--num-epochs"
        arg_type = Int
        default=100
    "--device"
        default="cpu"
    "--use-pretrain"
        default = false
    "--num-pheno-splits"
        arg_type = Int
        default = 10
    "--binarise-phenos"
        default = true 
    "--no-retrain"
        arg_type = Bool
        action = :store_false
        dest_name = "retrain"
    "--retrain"
        arg_type = Bool
        action = :store_true
    "--no-retest"
        arg_type = Bool
        action = :store_false
        dest_name = "retest"
    "--retest"
        arg_type = Bool
    "--grid-size"
        arg_type = Int
        default = 10
end
args = parse_args(ARGS, s)
args["single-input-matrix"] = true
num_epochs = args["num-epochs"]
num_and_layers = args["num-ands"]
num_lin_out = args["num-linear"]
num_thresholds = args["num-pheno-thresholds"]
output_suffix =  "_" * args["h5-dataset"]
output_file = args["output-file-base"] * output_suffix * ".jld2"

# load gout data


# override printed version of type, by default it's a long mess.
Base.show(io::IO, ::Type{StatisticalMeasuresBase.RobustMeasure{StatisticalMeasuresBase.FussyMeasure{StatisticalMeasuresBase.RobustMeasure{StatisticalMeasures._LogLossType{Float64}}, typeof(StatisticalMeasures.l2_check)}}}) = print(io, "Cross Entropy")
# run plots headless
ENV["GKSwstype"] = "100"

MLJ.default_resource(CPUThreads())


"""
if `machine_file` file exists and retrain=False, loads `machine_file`.
otherwise, trains a logistic machine and saves it there.
"""
function fit_lgbm(machine_file="lgbm_machine.jlso", retrain=true, grid_size=10, plot_file="lgbm_machine_hyperparam_search")
	if !isfile(machine_file) || retrain
        tg = coerce(combined_gout, OrderedFactor)
        mat_df = DataFrame(transpose(combined_mat), :auto)
        lgbm = LGBMClassifier(device_type="cpu", num_threads=32)
        num_leaves_range = range(lgbm, :num_leaves, lower=1, upper=10000, scale=:log10)
        min_data_in_leaf_range = range(lgbm, :min_data_in_leaf, lower=10, upper=10000, scale=:log10)
        max_depth_range = range(lgbm, :max_depth, lower=2, upper=20)
        hp_lgbm_model = TunedModel(model=lgbm,
            resampling=Holdout(fraction_train=train_frac),
            tuning=LatinHypercube(gens=2, popsize=120),
            n=grid_size,
            range=[num_leaves_range, min_data_in_leaf_range, max_depth_range],
            measure=cross_entropy)
        lgbm_machine = machine(hp_lgbm_model, mat_df, tg)
        MLJ.fit!(lgbm_machine, rows=1:length(tg))
		MLJ.save(machine_file, lgbm_machine)
    else
        lgbm_machine = machine(machine_file)
    end
    rep = report(lgbm_machine)
    rep.plotting.parameter_names[rep.plotting.parameter_names .== "num_leaves"] .= "Num Leaves"
    rep.plotting.parameter_names[rep.plotting.parameter_names .== "min_data_in_leaf"] .= "Min Data in Leaf"
    rep.plotting.parameter_names[rep.plotting.parameter_names .== "max_depth"] .= "Max Depth"
    lgbm_search_plot = Plots.plot(lgbm_machine)
    plot!(size=(800,800))
    # measurement = string(typeof(rep.best_history_entry.measure[1]))
    # typeof(rep.best_history_entry.measure[1])
    # print(lgbm_search_plot.subplots[1].attr[:xguide])
    savefig(lgbm_search_plot, "figures/" * plot_file * ".pdf")
    savefig(lgbm_search_plot, "figures/" * plot_file * ".png")
    return lgbm_machine
end

function eval_machine(lgbm_machine, eval_dat, args, retest=true, results_file="lgbm_verify_results.jld2")
    if retest || !isfile(results_file)
        mat, urate, gout = eval_dat.data.data
        mat_df = DataFrame(transpose(mat), :auto)
        tg = coerce(gout, OrderedFactor)
        yhat = predict(lgbm_machine, mat_df)
        lgbm_logloss = mean(LogLoss()(yhat, tg))
        lgbm_accuracy = Accuracy()(mode.(yhat), tg)
        save(results_file, Dict(
            "y" => tg,
            "yhat" => yhat,
            "logloss"  => lgbm_logloss,
            "accuracy" => lgbm_accuracy,
            "args" => args
            ))
    else
        results = JLD2.load(results_file)
        yhat = results["yhat"]
        tg = results["y"]
        lgbm_logloss = results["logloss"]
        lgbm_accuracy = results["accuracy"]
    end
    fpr, tpr, thresholds = roc_curve(yhat, tg)
    auroc = auc(yhat, tg)
    plot_roc("figures/lgbm_roc" * output_suffix, fpr, tpr, auroc)
    Printf.@printf("auroc: %f\n", auroc)
    Printf.@printf("accuracy: %f\n", lgbm_accuracy)
    # plot machine hyperparameter search
    # Plots.plot

    return (lgbm_logloss, lgbm_accuracy)
end

machine_file = "lgbm_machine" * output_suffix * ".jlso"
if (args["retrain"] || args["retest"]) || !isfile(machine_file)
    ld_args = copy(args)
    ld_args["h5-dataset"] = args["h5-datadir"] * args["h5-dataset"] * ".h5"
    train_dl, test_dl, verify_dl, tok_to_string = load_data(ld_args)
else
    verify_dl = nothing
end
if (args["retrain"]) || !isfile(machine_file)
    seq_len = size(train_dl.data.data[1])[1]

    train_mat, train_urate, train_gout = train_dl.data.data
    test_mat, test_urate, test_gout = test_dl.data.data
    # combine train/test set for TunedModel Holdout function
    combined_mat = cat(train_mat, test_mat, dims=2)
    combined_gout = cat(train_gout, test_gout, dims=1)
    train_frac = length(train_gout)/length(combined_gout)
end

lgbm_machine = fit_lgbm(machine_file, args["retrain"], args["grid-size"], "lgbm_hyperparam_search" * output_suffix)
eval_machine(lgbm_machine, verify_dl, args, args["retest"], "lgbm_verify_results" * output_suffix *".jld2")
