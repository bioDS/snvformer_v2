using ArgParse
using JLD2
using Lux
using Printf
using CUDA
using LuxCUDA
using Hyperopt
using Distributed
using Plots
using Gadfly, Compose
import Cairo, Fontconfig
using Printf
using StatisticalMeasuresBase
# using ThreadPools
# using Base.Threads
# using FLoops
include("data.jl")
include("logic_models.jl")
include("MLJLogicModel.jl")
include("jl_common.jl")

# Distributed.addprocs(8)
# MLJ.default_resource(CPUProcesses())
MLJ.default_resource(CPU1())

# override printed version of type, by default it's a long mess.
Base.show(io::IO, ::Type{StatisticalMeasuresBase.RobustMeasure{StatisticalMeasuresBase.FussyMeasure{StatisticalMeasuresBase.RobustMeasure{StatisticalMeasures._LogLossType{Float64}}, typeof(StatisticalMeasures.l2_check)}}}) = print(io, "Cross Entropy")
# run plots headless
ENV["GKSwstype"] = "100"

function mlj_search(train_dl, test_dl, default_args, grid_size=10, save_file="tuned_logic_model_" * h5_base * ".jld2")
    # args = copy(default_args)
    X, y = dl_to_table(train_dl)
    tX, ty = dl_to_table(test_dl)
    combined_X = vcat(X, tX)
    combined_y = cat(y, ty, dims=1)
    train_frac = length(y)/length(combined_y)
    
    hyper_model = HPLogicModel(num_epochs=20, use_device=to_device, batch_size=32)
    num_linear = range(hyper_model, :num_linear, lower=1, upper=32)
    num_ands = range(hyper_model, :num_ands, lower=1, upper=32)
    num_pheno_thresholds = range(hyper_model, :num_pheno_thresholds, lower=1, upper=20)
    lr=range(hyper_model, :lr, lower=1e-7, upper=1e-1, scale=:log)

    tunable_logic_model = TunedModel(model=hyper_model,
        resampling=Holdout(fraction_train=train_frac),
        # tuning=Grid(goal=grid_size),
        tuning=LatinHypercube(gens=2, popsize=120),
        n=grid_size,
        range=[
            num_linear,
            num_ands,
            num_pheno_thresholds,
            lr
        ],
        measure=LogLoss(),
    )
    tunable_logic_machine = machine(tunable_logic_model, combined_X, combined_y)
    MLJ.fit!(tunable_logic_machine)
    
    of = jldopen(save_file, "w")
    of["tuned_logic_machine"] = tunable_logic_machine
    close(of)
    min_version_params = tunable_logic_machine.report[:fit].best_model
    iter_args = [i.model for i in tunable_logic_machine.report[:fit].history]
    iter_loss = [i.measurement for i in tunable_logic_machine.report[:fit].history]
    return min_version_params, iter_args, iter_loss
end

function grid_search(train_dl, test_dl, default_args)
    args = copy(default_args)
    # iter_args = []
    and_range = [1, 2, 4, 8, 16, 32]
    linear_range =  [1, 2, 4, 8, 16, 32]
    threshold_range = [5, 10, 20]
    total_tests = length(and_range)*length(linear_range)*length(threshold_range)
    iter_args = Vector(undef, total_tests)
    iter_best_loss = Vector(undef, total_tests)
    iter_final_loss = Vector(undef, total_tests)
    iter_best_epoch = Vector(undef, total_tests)
    # iter_loss = []
    # iter_ind = 1
    for (ai, num_ands) in enumerate(and_range)
       for (bi, num_linear) in enumerate(linear_range)
            for (ti, num_pheno_thresholds) in enumerate(threshold_range)
                iter_ind = ti + (bi-1)*length(threshold_range) + (ai-1)*length(threshold_range)*length(linear_range)
                args["num-ands"] = num_ands
                args["num-linear"] = num_linear
                args["num-pheno-thresholds"] = num_pheno_thresholds
                model = LuxPhenoComboModel_init(seq_len, args["num-linear"], args["num-ands"], args["num-pheno-thresholds"])
                ps, st = Lux.setup(rng, model) |> to_device
                opt_state = Optimisers.setup(Optimisers.AdamW(1e-4), ps)
                opt_state, ps, best_ps, epoch_train_losses, epoch_test_losses = dual_pheno_train(model, train_dl, test_dl, ps, st, opt_state, args["num-epochs"])
                # epoch_test_losses = [1.0]
                min_test_loss, min_epoch = findmin(filter(!isnan, epoch_test_losses))
                args["best_epoch"] = min_epoch
                # push!(iter_args, copy(args))
                # push!(iter_loss, min_test_loss)
                iter_args[iter_ind] = copy(args)
                iter_best_loss[iter_ind] = min_test_loss
                iter_final_loss[iter_ind] = epoch_test_losses[end]
                iter_best_epoch[iter_ind] = min_epoch
                @printf("iter %d, loss: %g at epoch %d\n", iter_ind, min_test_loss, min_epoch)
            end
        end
    end
    of = jldopen("logic_hyperparam_grid_details.jlso", "w")
    of["iter_args"] = iter_args
    of["iter_best_loss"] = iter_best_loss
    of["iter_final_loss"] = iter_final_loss
    of["iter_best_epoch"] = iter_best_epoch
    close(of)
    min_version_loss, ind = findmin(iter_best_loss)
    min_version_params = iter_args[ind]
    @printf("minimum loss was %g, at arg ind %d\n", min_version_loss, ind)
    println("minimum parameters were")
    println(min_version_params)
    return min_version_params, iter_args, iter_best_loss
end


# objective = function(resources::Real, params::AbstractVector)
#     # num_ands = round.(Int,exp2.(range(0,5))),
#     # num_linear = round.(Int,exp2.(range(0,5))),
#     # num_pheno_thresholds = range(1,20),
#     # lr = exp10.(LinRange(-1,-7,1000)),
#     # sampler=Hyperband(R=num_samples, η=3, inner=BOHB(dims=[Hyperopt.Continuous(), Hyperopt.Continuous(), Hyperopt.Continuous(), Hyperopt.Continuous()]))
#     num_ands, num_linear, num_pheno_thresholds, lr, state, default_args = params
#     num_ands = Int(num_ands)
#     num_linear = Int(num_linear)
#     num_pheno_thresholds = Int(num_pheno_thresholds)
#     # @show params
#     if !(state === nothing)
#         ps, st, opt_state, epochs_run, args = state
#         ps = ps |> to_device
#         st = st |> to_device
#     else
#         epochs_run = 0
#         args = copy(default_args)
#         args["num-ands"] = num_ands
#         args["num-linear"] = num_linear
#         args["num-pheno-thresholds"] = num_pheno_thresholds
#         args["lr"] = lr
#     end
#     new_epochs = round(Int, resources)
#     args["num-epochs"] = new_epochs
#     model = LuxPhenoComboModel_init(seq_len, args["num-linear"], args["num-ands"], args["num-pheno-thresholds"])
#     if (state === nothing)
#         ps, st = Lux.setup(rng, model) |> to_device
#         opt_state = Optimisers.setup(Optimisers.AdamW(lr), ps)
#     end
#     # opt_state, ps, best_ps, epoch_train_losses, epoch_test_losses = dual_pheno_train(model, train_data, test_data, ps, st, opt_state, args["num-epochs"])
#     opt_state, ps, best_ps, epoch_train_losses, epoch_test_losses = dual_pheno_train(model, train_dl, test_dl, ps, st, opt_state, round(Int,new_epochs-epochs_run))
#     # min_test_loss, min_epoch = findmin(filter(!isnan, epoch_test_losses))
#     final_train_loss = epoch_train_losses[end]
#     final_test_loss = epoch_test_losses[end]
#     print(epochs_run, "-", resources, "\t", args["num-ands"], "\t", args["num-linear"], "\t", args["num-pheno-thresholds"], "\t", final_train_loss, "\t", final_test_loss, "\n")
#     epochs_run += new_epochs
#     state = ps, st, opt_state, epochs_run, args |> cpu_device
#     final_test_loss, state
# end


function hyperopt_search(train_dl, test_dl, default_args, num_samples=50, save_ho_file="logic_hyperoptimiser.jld2")
    default_args["lr"] = 1e-4
    args = copy(default_args)

    # candidates = (
    #     num_ands = round.(Int,exp2.(range(0,5))),
    #     num_linear = round.(Int,exp2.(range(0,5))),
    #     num_pheno_thresholds = range(1,20),
    #     lr = exp10.(LinRange(-1,-7,1000)),
    #     state = [nothing],
    #     default_args = [args],
    # )
    # ho = hyperband(objective, candidates, R=num_samples, η=3, threads=true)

    ho = @hyperopt for resources=num_samples,
            num_ands = round.(Int,exp2.(range(0,5))),
            num_linear = round.(Int,exp2.(range(0,5))),
            num_pheno_thresholds = range(1,20),
            lr = exp10.(LinRange(-1,-7,1000)),
            sampler=Hyperband(R=num_samples, η=3, inner=RandomSampler())
            # sampler=Hyperband(R=num_samples, η=3, inner=BOHB(dims=[Hyperopt.Continuous(), Hyperopt.Continuous(), Hyperopt.Continuous(), Hyperopt.Continuous()]))
            # sampler = RandomSampler()
            # num_ands   = range(1,64),
            # num_linear = range(1,64),
            # num_pheno_thresholds = range(1,20)
        # sampler = Hyperband(R=50, η=3, inner=RandomSampler())
        # num_ands = Range)
        if !(state === nothing)
            ps, st, opt_state, epochs_run, args = state
            ps = ps |> to_device
            st = st |> to_device
        else
            epochs_run = 0
            args = copy(default_args)
            args["num-ands"] = num_ands
            args["num-linear"] = num_linear
            args["num-pheno-thresholds"] = num_pheno_thresholds
            args["lr"] = lr
        end
        new_epochs = round(Int, resources)
        args["num-epochs"] = new_epochs
        model = LuxPhenoComboModel_init(seq_len, args["num-linear"], args["num-ands"], args["num-pheno-thresholds"])
        if (state === nothing)
            ps, st = Lux.setup(rng, model) |> to_device
            opt_state = Optimisers.setup(Optimisers.AdamW(lr), ps)
        end
        # opt_state, ps, best_ps, epoch_train_losses, epoch_test_losses = dual_pheno_train(model, train_data, test_data, ps, st, opt_state, args["num-epochs"])
        opt_state, ps, best_ps, epoch_train_losses, epoch_test_losses = dual_pheno_train(model, train_dl, test_dl, ps, st, opt_state, round(Int,new_epochs-epochs_run))
        # min_test_loss, min_epoch = findmin(filter(!isnan, epoch_test_losses))
        final_train_loss = epoch_train_losses[end]
        final_test_loss = epoch_test_losses[end]
        print(epochs_run, "-", resources, "\t", args["num-ands"], "\t", args["num-linear"], "\t", args["num-pheno-thresholds"], "\t", final_train_loss, "\t", final_test_loss, "\n")
        epochs_run += new_epochs
        state = ps, st, opt_state, epochs_run, args |> cpu_device
    final_test_loss, state
    end

    # @show ho
    of = jldopen(save_ho_file, "w")
    of["ho"] = ho
    close(of)
    return ho
end


s = ArgParseSettings()
@add_arg_table! s begin
    "--h5-dataset"
        help = "an option with an argument"
        default = "/data/ukbb/net_input/gwas_ldprune_320.h5"
    "--output-file", "-o"
        help = "save model and summary statistics here as jld1"
        default = "logic_model_grid_search.jld2"
    "--test-frac"
        arg_type = Float64
        default=0.3
    "--batch-size"
        arg_type = Int
        default=128
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
        default=10
    "--device"
        default="cpu"
    "--no-use-pretrain"
        action = :store_false
        dest_name = "use-pretrain"
    "--use-pretrain"
        action = :store_true
    "--binarise-phenos"
        default = true
    "--search-type"
        default="mlj"
    "--num-samples"
        arg_type = Int
        default = 64
    "--rerun"
        action="store_true"
    "--retest"
        arg_type = Bool
        action="store_true"
end
args = parse_args(ARGS, s)
args["single-input-matrix"] = false
num_epochs = args["num-epochs"]
num_and_layers = args["num-ands"]
num_lin_out = args["num-linear"]
num_thresholds = args["num-pheno-thresholds"]
h5ds_file = args["h5-dataset"]
h5_base = string(match(r"[_\-a-z0-9]+.h5", h5ds_file).match[1:end-3])
results_file="logic_verify_results_" * h5_base *".jld2"
binary_results_file="logic_verify_results_" * h5_base *"-binary.jld2"

cpu_device = LuxCPUDevice()
if args["device"] == "cpu"
    to_device = LuxCPUDevice()
else
    device!(parse(Int,args["device"]))
    to_device = LuxCUDADevice()
end

if args["rerun"] || !isfile(args["output-file"]) || args["retest"] || !isfile(results_file) || !isfile(binary_results_file)
    train_dl, test_dl, verify_dl, tok_to_string = load_data(args)
else
    verify_dl = nothing
end

if args["rerun"] || !isfile(args["output-file"])
    # load gout data
    seq_len = size(train_dl.data.data[1])[1]
    if args["search-type"] == "hyperopt"
        ho = hyperopt_search(train_dl, test_dl, args, args["num-samples"], "logic_hyperoptimiser.jld2")
        ps, st, opt_state, epochs_run, best_params = ho.minimizer
        iter_args = ho.history
        iter_loss = ho.results
    elseif args["search-type"] == "grid"
        best_params, iter_args, iter_loss = grid_search(train_dl, test_dl, args)
    elseif args["search-type"] == "mlj"
        best_params, iter_args, iter_loss = mlj_search(train_dl, test_dl, args, args["num-samples"])
    else
        print("search-type must be one of hyperopt, grid, mlj")
    end
    of = jldopen(args["output-file"], "w")
    of["best_params"] = best_params
    of["iter_args"] = iter_args
    of["iter_loss"] = iter_loss
    close(of)
end

of = jldopen(args["output-file"])
best_params = of["best_params"]
iter_loss = of["iter_loss"]
close(of)
of = jldopen("tuned_logic_model_" * h5_base * ".jld2") # mlj model
logic_machine = of["tuned_logic_machine"]
close(of)

print("best params: ")
@show best_params
print("best loss: ", minimum(filter.(!isnan, iter_loss)))

# plot mlj parameter search
rep = report(logic_machine)
rep.plotting.parameter_names[rep.plotting.parameter_names .== "num_linear"] .= "No. Linear"
rep.plotting.parameter_names[rep.plotting.parameter_names .== "num_ands"] .= "No. Ands"
rep.plotting.parameter_names[rep.plotting.parameter_names .== "num_pheno_thresholds"] .= "No. Pheno Thresholds"
rep.plotting.parameter_names[rep.plotting.parameter_names .== "lr"] .= "Learning Rate"
mlj_search_plot = Plots.plot(logic_machine, dpi=300)
plot!(size=(800,800))
savefig(mlj_search_plot, "figures/mlj_logic_search_plot_" * h5_base * ".pdf")
savefig(mlj_search_plot, "figures/mlj_logic_search_plot_" * h5_base * ".png")

# best_params = HPLogicModel(num_linear = 1, …)
# HPLogicModel(
#   num_linear = 1, 
#   num_ands = 1, 
#   num_pheno_thresholds = 10, 
#   num_epochs = 20, 
#   use_device = LuxCPUDevice, 
#   lr = 0.10000000000000002, 
#   batch_size = 32)
# 
# best loss: [0.48513252034070037]

function eval_machine(logic_machine, eval_dat, args, retest=true, results_file="logic_verify_results.jld2")
    if retest || !isfile(results_file)
        tX, ty = dl_to_table(eval_dat)

        yhat = 
        if typeof(logic_machine) <: Machine
            predict(logic_machine, tX)
        else
            bmodel, bps, bst = logic_machine
            X = tab_to_X(tX)
            to_device = hyper_model.use_device
            seq_len = size(X[1])[1]
            snvs, gout, urate, bmi, height, age, male = X
            pred = bmodel(snvs |> to_device, bmi |> to_device, age |> to_device, Float32.(male) |> to_device, bps, bst)
            pred_2 = hcat(1. .- pred, pred)
            pred_cat = MLJModelInterface.UnivariateFinite([false, true], pred_2, pool=missing)
            pred_cat
        end
        logic_logloss = mean(LogLoss()(yhat, ty))
        logic_accuracy = Accuracy()(mode.(yhat), ty)
        save(results_file, Dict(
            "y" => ty,
            "yhat" => yhat,
            "logloss"  => logic_logloss,
            "accuracy" => logic_accuracy,
            "args" => args
            ))
    else
        results = JLD2.load(results_file)
        yhat = results["yhat"]
        ty = results["y"]
        logic_logloss = results["logloss"]
        logic_accuracy = results["accuracy"]
    end
    ty_cat = coerce(ty, OrderedFactor)
    fpr, tpr, thresholds = roc_curve(yhat, ty_cat)
    auroc = auc(yhat, ty_cat)
    plot_roc("figures/logic_roc_" * h5_base, fpr, tpr, auroc)
    Printf.@printf("auroc: %f\n", auroc)
    Printf.@printf("accuracy: %f\n", logic_accuracy)
    # plot machine hyperparameter search
    # Plots.plot

    return (logic_logloss, logic_accuracy)
end

eval_machine(logic_machine, verify_dl, args, args["retest"], results_file)

# binarise and test binary model
ps, st = logic_machine.fitresult.fitresult
bmodel, bps, bst = binarise_lux_pheno_combo_model(ps, st, logic_machine.fitresult.model)
eval_machine((bmodel, bps, bst), verify_dl, args, args["retest"], binary_results_file)