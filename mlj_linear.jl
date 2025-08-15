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
using LossFunctions
using Gadfly, Compose
import Cairo, Fontconfig
using Printf
using Plots
using StatisticalMeasuresBase
using MLJLinearModels
using MLJTuning
# using Infiltrator
include("jl_common.jl")

# if isdefined(Main, :Infiltrator)
#   Main.infiltrate(Plots, Base.@locals, "utils.jl", 106)
# end

# if isdefined(Main, :Infiltrator)
#   Main.infiltrate(@__MODULE__, Base.@locals, @__FILE__, @__LINE__)
# end

MLJ.@load LogisticClassifier pkg=MLJLinearModels

s = ArgParseSettings()
@add_arg_table! s begin
    "--h5-datadir"
        help = "an option with an argument"
        default = "/data/ukbb/net_input/"
    "--h5-dataset"
        help = "an option with an argument"
        default = "tin_fixed_order"
    "--output-file-base", "-o"
        help = "save model and summary statistics here as jld2"
        default = "linear_output"
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
        action = :store_true
    "--grid-size"
        arg_type = Int
        default = 64
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
function fit_logistic(machine_file="logistic_machine.jlso", retrain=true, grid_size=10, plot_file="logistic_machine_hyperparam_search")
	if !isfile(machine_file) || retrain
        tg = coerce(combined_gout, OrderedFactor)
        mat_df = DataFrame(transpose(combined_mat), :auto)
        logistic = LogisticClassifier(penalty=:en)
        lambda = range(logistic, :lambda, lower=1e-10, upper=10, scale=:log10)
        # penalty = range(logistic, :penalty, values=[:l2, :l1])
        # gamma = range(logistic, :gamma, lower=0.0, upper=1.0)
        gamma = range(logistic, :gamma, lower=1e-10, upper=10, scale=:log10)
        fit_intercept = range(logistic, :fit_intercept, values=[true, false])
        penalize_intercept = range(logistic, :penalize_intercept, values=[true, false])
        hp_logistic_model = TunedModel(model=logistic,
            resampling=Holdout(fraction_train=train_frac),
            tuning=LatinHypercube(gens=2, popsize=120),
            n=grid_size,
            range=[lambda, gamma, fit_intercept, penalize_intercept],
            measure=cross_entropy)
        logistic_machine = machine(hp_logistic_model, mat_df, tg)
        MLJ.fit!(logistic_machine, rows=1:length(tg))
		MLJ.save(machine_file, logistic_machine)
    else
        logistic_machine = machine(machine_file)
    end
    rep = report(logistic_machine)
    rep.plotting.parameter_names[rep.plotting.parameter_names .== "lambda"] .= "Lambda"
    rep.plotting.parameter_names[rep.plotting.parameter_names .== "gamma"] .= "Gamma"
    rep.plotting.parameter_names[rep.plotting.parameter_names .== "fit_intercept"] .= "Fit Intercept"
    rep.plotting.parameter_names[rep.plotting.parameter_names .== "penalize_intercept"] .= "Penalise Intercept"
    logistic_search_plot = Plots.plot(logistic_machine, dpi=300)
    plot!(size=(900,800))
    # measurement = string(typeof(rep.best_history_entry.measure[1]))
    # typeof(rep.best_history_entry.measure[1])
    # print(logistic_search_plot.subplots[1].attr[:xguide])
    savefig(logistic_search_plot, "figures/" * plot_file * ".pdf")
    savefig(logistic_search_plot, "figures/" * plot_file * ".png")
    return logistic_machine
end

function eval_machine(logistic_machine, eval_dat, args, retest=true, results_file="logistic_verify_results.jld2")
    if retest || !isfile(results_file)
        mat, urate, gout = eval_dat.data.data
        mat_df = DataFrame(transpose(mat), :auto)
        tg = coerce(gout, OrderedFactor)
        yhat = predict(logistic_machine, mat_df)
        logistic_logloss = mean(LogLoss()(yhat, tg))
        logistic_accuracy = Accuracy()(mode.(yhat), tg)
        save(results_file, Dict(
            "y" => tg,
            "yhat" => yhat,
            "logloss"  => logistic_logloss,
            "accuracy" => logistic_accuracy,
            "args" => args
            ))
    else
        results = JLD2.load(results_file)
        yhat = results["yhat"]
        tg = results["y"]
        logistic_logloss = results["logloss"]
        logistic_accuracy = results["accuracy"]
    end
    fpr, tpr, thresholds = roc_curve(yhat, tg)
    auroc = auc(yhat, tg)
    plot_roc("figures/logistic_roc" * output_suffix, fpr, tpr, auroc)
    Printf.@printf("auroc: %f\n", auroc)
    Printf.@printf("accuracy: %f\n", logistic_accuracy)

    return (logistic_logloss, logistic_accuracy)
end

machine_file = "logistic_machine" * output_suffix * ".jlso"
if (args["retrain"] || args["retest"]) || !isfile(machine_file)
    ld_args = copy(args)
    ld_args["h5-dataset"] = args["h5-datadir"] * args["h5-dataset"] * ".h5"
    train_dl, test_dl, verify_dl, tok_to_string = load_data(ld_args)
else
    train_dl = nothing
    test_dl = nothing
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

logistic_machine = fit_logistic(machine_file, args["retrain"], args["grid-size"], "logistic_hyperparam_search" * output_suffix)
logistic_logloss, logistic_accuracy = eval_machine(logistic_machine, verify_dl, args, args["retest"], "logistic_verify_results" * output_suffix *".jld2")
