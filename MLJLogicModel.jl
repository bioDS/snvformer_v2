using ArgParse
using JLD2
using Lux
using Printf
using CUDA
using LuxCUDA
# using ThreadPools
# using Base.Threads
include("data.jl")
include("logic_models.jl")
# import MLJBase
using Tables
using MLJ
import MLJBase
# using CategoricalDistributions
import MLJModelInterface
using Plots

# logic model hyperparameters
# needed to initialise 
# model = LuxPhenoComboModel_init(seq_len, num-linear, num-ands, num-pheno-thresholds)
mutable struct HPLogicModel <: MLJBase.Probabilistic
    num_linear::Int
    num_ands::Int
    num_pheno_thresholds::Int
    num_epochs::Int
    use_device::LuxDeviceUtils.AbstractLuxDevice
    lr::Float64
    batch_size::Int
end

function HPLogicModel(;
        num_linear=2,
        num_ands=2,
        num_pheno_thresholds=10,
        num_epochs=10,
        use_device=LuxCPUDevice(),
        lr=1e-4,
        batch_size=4)
    
    return HPLogicModel(
            num_linear,
            num_ands,
            num_pheno_thresholds,
            num_epochs,
            use_device,
            lr,
            batch_size)
end

# Define a fit function for use in MLJ framework.
# hyperparameters are defined by model.
# returns:
#   fitresult: parameters for a model fit according to hyperparameters
#   cache:     model specific
# X is a table containing: [snvs, gout, urate, bmi, height, age, male]
# y is a vector: [gout]


function tab_to_X(tab)
    snvs = tab[1:(end-6),:]
    first_gout_ind = findall(names(tab) .== "gout")[1]
    snvs = transpose(Matrix(tab[1:end,1:first_gout_ind-1]))
    X = (snvs, tab.gout, tab.urate, tab.bmi, tab.height, tab.age, tab.male)
    X
end

function tX_to_X(tX_transpose)
    # tX = transpose(tX_transpose)
    male = Bool.(tX[end,:])
    age = Int.(tX[end-1,:])
    height = tX[end-2,:]
    bmi = tX[end-3,:]
    urate = tX[end-4,:]
    gout = Bool.(tX[end-5,:])
    snvs = tX[1:(end-6),:]
    X = (snvs, gout, urate, bmi, height, age, male)
    X
end

function MLJBase.fit(hyper_model::HPLogicModel, verbosity, tX, y)
    # tX is cat(X, gout, urate, bmi, height, age, male)
    # convert to X, phenos...
    # setup model
    cpu_device = LuxCPUDevice()
    X = tab_to_X(tX)
    seq_len = size(X[1])[1]
    to_device = hyper_model.use_device
    model = LuxPhenoComboModel_init(seq_len, hyper_model.num_linear, hyper_model.num_ands, hyper_model.num_pheno_thresholds)
    ps, st = Lux.setup(rng, model) |> hyper_model.use_device
    opt_state = Optimisers.setup(Optimisers.AdamW(hyper_model.lr), ps)

    X_with_y = (X..., y)

    # optimise model
    # train_losses = []
    train_dl = DataLoader(X_with_y, hyper_model.batch_size)
    for e in range(1, hyper_model.num_epochs)
        for (snvs, _, urate, bmi, height, age, male, batch_y) in train_dl
            loss, pb = pullback(ps) do ps
                pred = model(snvs |> to_device, bmi |> to_device, age |> to_device, Float32.(male) |> to_device, ps, st)
                loss = binarycrossentropy(pred, batch_y |> to_device)
                loss
            end
            (∇params,) = pb(to_device(one(loss)))
            opt_state, ps = Optimisers.update(opt_state, ps, ∇params)
        end
        # push!(epoch_parameters, cpu(deepcopy(ps)))
        # if e % (num_epochs/10) == 0
        # train_mean_loss = get_mean_loss(model, ps, st, train_data)
        # test_mean_loss = get_mean_loss(model, ps, st, test_data)
        # append!(train_losses, train_mean_loss)
        # append!(test_losses, test_mean_loss)
        # println((train_mean_loss, test_mean_loss))
        # end
    end
    # best_param_ind = argmin(test_losses)
    # best_ps = epoch_parameters[best_param_ind]
    # opt_state, ps, best_ps, train_losses, test_losses
    fitresult = (ps, st) |> cpu_device
    cache = nothing
    report = nothing
    return fitresult, cache, report
end

function MLJBase.predict(hyper_model::HPLogicModel, fitresult, tX)
    X = tab_to_X(tX)
    to_device = hyper_model.use_device
    seq_len = size(X[1])[1]
    model = LuxPhenoComboModel_init(seq_len, hyper_model.num_linear, hyper_model.num_ands, hyper_model.num_pheno_thresholds)
    ps, st = fitresult
    snvs, gout, urate, bmi, height, age, male = X
    pred = model(snvs |> to_device, bmi |> to_device, age |> to_device, Float32.(male) |> to_device, ps, st)
    pred_2 = hcat(1. .- pred, pred)
    pred_cat = MLJModelInterface.UnivariateFinite([false, true], pred_2, pool=missing)
    return pred_cat
end

function dl_to_matrix(train_dl)
    train_data = train_dl.data.data
    X = train_data
    y = train_data[2]

    snvs = X[1]
    phenos = X[2:end]
    tX = transpose(cat(snvs, [reshape(p, (1,:)) for p in phenos]..., dims=1))
    return tX, y
end

function dl_to_table(train_dl)
    train_data = train_dl.data.data
    X = train_data
    y = train_data[2]

    snvs = X[1]
    phenos = X[2:end]
    snv_tab = DataFrame(transpose(snvs), :auto)
    pheno_tab = DataFrame(
        gout = phenos[1],
        urate = phenos[2],
        bmi = phenos[3],
        height = phenos[4],
        age = phenos[5],
        male = phenos[6]
    )
    combined_tab = hcat(snv_tab, pheno_tab)
    # tX = transpose(cat(snvs, [reshape(p, (1,:)) for p in phenos]..., dims=1))
    return combined_tab, y
end

# function combine_datasets(X, Xnew)
#     combined_snvs = cat(X[1], Xnew[1], dims=2)
#     combined_phenos = [cat(a,b, dims=1) for (a,b) in zip(X[2:end], Xnew[2:end])]
#     combined_X = (combined_snvs, combined_phenos...)
#     train_frac = length(X[1])/length(combined_X[1])
#     return combined_X, train_frac
# end


function testing()
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
        "--device"
            default="cpu"
        "--no-use-pretrain"
            action = :store_false
            dest_name = "use-pretrain"
        "--use-pretrain"
            action = :store_true
        "--binarise-phenos"
            default = true
    end
    args = parse_args(ARGS, s)
    args["single-input-matrix"] = false
    args["batch-size"] = 4 # not actually used
    train_dl, test_dl, verify_dl, tok_to_string = load_data(args)

    # train_data = train_dl.data.data # [snvs, gout, urate, bmi, height, age, male]
    # X = train_data
    # y = train_data[2]
    # X, y = dl_to_matrix(train_dl)
    # tX, ty = dl_to_matrix(test_dl)
    X, y = dl_to_table(train_dl)
    tX, ty = dl_to_table(test_dl)

    # tx = cat(X..., dims=1)

    # X_df = DataFrame(
    #     "snvs" => X[1],
    #     "gout" => X[2],
    #     "urate" => X[3],
    #     "bmi" => X[4],
    #     "height" => X[5],
    #     "age" => X[6],
    #     "male" => X[7]
    # )

    # Xnew = test_dl.data.data

    hyper_model = HPLogicModel()

    logic_machine = machine(hyper_model, X, y)
    MLJ.fit!(logic_machine)
    tyh = MLJ.predict(logic_machine, tX)
    # convert to distribution
    # pool=CategoricalDistributions.CategoricalPool(["false", "true"])
    # UnivariateFinite(["false", "true"], [0.1, 0.9], pool=pool)
    # tmp = [UnivariateFinite(["false", "true"], [1-x, x], pool=pool) for x in tyh]
    # # t2 = UnivariateFiniteVector(tmp)
    # t2 = UnivariateFinite(tyh)
    # tyh
    # predicted = hcat(1. .- tyh, tyh)
    # tmp = MLJModelInterface.UnivariateFinite([false, true], predicted)
    LogLoss()(tyh, ty)
    # rms(tyh, ty)

    combined_X = vcat(X, tX)
    combined_y = cat(y, ty, dims=1)
    train_frac = length(y)/length(combined_y)

    # combined_snvs = cat(X[1], Xnew[1], dims=2)
    # combined_phenos = [cat(a,b, dims=1) for (a,b) in zip(X[2:end], Xnew[2:end])]
    # combined_X = (combined_snvs, combined_phenos...)
    # combined_y = cat(X[7], Xnew[7], dims=1)
    # # combined_mat = cat(train_mat, test_mat, dims=2)
    # # combined_gout = cat(train_gout, test_gout, dims=1)
    # train_frac = length(X[1])/length(combined_X[1])

    # tX = cat(combined_snvs, combined_phenos..., dims=1)
    # g = combined_phenos[1]
    # mg = reshape(g, (1,:))
    # cat(combined_snvs, mg, dims=1)
    # cat(combined_snvs, [reshape(p, (1,:)) for p in combined_phenos]..., dims=1)
    # # tX = cat(combined_snvs, resize(combined_phenos[1], 1, :), dims=1)

    # sanity test
    #X, y = @load_iris
    #mach = fit!(machine(MyClassifier(), X, y))
    #predict(mach, selectrows(X, 1:2))

    #rX, ry = @load_boston
    #regressor = machine(MyRegressor(lambda=1.0), rX, ry)
    #evaluate!(regressor, resampling=CV(), measure=rms, verbosity=0)


    logic_machine2 = machine(hyper_model, combined_X, combined_y)

    l = LogLoss()

    evaluate(hyper_model, X, y, resampling=CV(nfolds=3), measure=(LogLoss()))

    # evaluate!(logic_machine, resampling=CV(), measure=LogLoss())
    # MLJ.fit!(logic_machine2, )
    evaluate!(logic_machine2, resampling=Holdout(fraction_train=train_frac), measure=LogLoss())

    tunable_logic_model = TunedModel(model=hyper_model,
        resampling=Holdout(fraction_train=train_frac),
        tuning=Grid(goal=3),
        range=[
            range(hyper_model, :num_linear, lower=1, upper=10),
        ],
        measure=LogLoss()
    )
    tunable_machine = machine(tunable_logic_model, combined_X, combined_y)
    MLJ.fit!(tunable_machine)
    Plots.plot(tunable_machine)
end