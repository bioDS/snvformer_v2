include("data.jl")
include("logic_models.jl")
include("MLJLogicModel.jl")
using ArgParse
using JLD2
using Lux
using CUDA
using DataLoaders
using DataFrames
using Optimisers
using ROCAnalysis
using Flux.Functors

# hyperopt w/ MLJ
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


s = ArgParseSettings()
@add_arg_table! s begin
    "--h5-dataset"
        help = "an option with an argument"
        default = "/data/ukbb/net_input/gwas_ldprune_320.h5"
    "--output-file", "-o"
        help = "save model and summary statistics here as jld1"
        default = "logic_model_output.jld2"
    "--test-frac"
        arg_type = Float64
        default=0.3
    "--batch-size"
        arg_type = Int
        default=32
    "--num-ands"
        arg_type = Int
        default=1
    "--num-linear"
        arg_type = Int
        default=1
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
    "--binarise-phenos"
        default = true 
    "--lr"
        default=0.1
end
args = parse_args(ARGS, s)
args["single-input-matrix"] = false
num_epochs = args["num-epochs"]
num_and_layers = args["num-ands"]
num_lin_out = args["num-linear"]
num_thresholds = args["num-pheno-thresholds"]

cpu_device = LuxCPUDevice()
if args["device"] == "cpu"
    to_device = LuxCPUDevice()
else
    device!(parse(Int,args["device"]))
    to_device = LuxCUDADevice()
end

# load gout data
train_dl, test_dl, verify_dl, tok_to_string = load_data(args)
seq_len = size(train_dl.data.data[1])[1]

# MLJ Version, doesn't choose best epoch
# hyper_model = HPLogicModel(
#     num_epochs=args["num-epochs"],
#     use_device=to_device,
#     batch_size=args["batch-size"],
#     num_linear=args["num-linear"],
#     num_ands=args["num-ands"],
#     num_pheno_thresholds=args["num-pheno-thresholds"],
#     lr=args["lr"]
# )
# X, y = dl_to_table(train_dl)
# logic_machine = machine(hyper_model, X, y)
# MLJ.fit!(logic_machine)
# tX, ty = dl_to_table(test_dl)
# tyh = MLJ.predict(logic_machine, tX)
# test_loss = LogLoss()(tyh, ty)
# vX, vy = dl_to_table(verify_dl)
# vyh = MLJ.predict(logic_machine, vX)
# verify_loss = LogLoss()(vyh, vy)
# previous run was with use-pretrain=true
#of = jldopen("logic_model_pretrain_included_results.jld", "w")
#of["tyh"] = tyh
#of["ty"] = ty
#of["logloss"] = loss
#close(of)

# pure-logic model, m/f split:
## model & optimiser setup
model = LuxPhenoComboModel_init(seq_len, num_lin_out, num_and_layers, num_thresholds)
ps, st = Lux.setup(rng, model) |> to_device
opt_state = Optimisers.setup(Optimisers.AdamW(args["lr"]), ps)

# training loop
opt_state, ps, best_ps, train_losses, test_losses = dual_pheno_train(model, train_dl, test_dl, ps, st, opt_state, args["num-epochs"])

# print & save results
println("non-binary model")
mean_loss, acc, roc_curve, preds, targets = eval_dual_pheno_model(verify_dl, model, best_ps |> to_device, st)
auroc = ROCAnalysis.auc(roc_curve)
println("auroc: ",  auroc)
println("acc: ",  acc)
println("mean_loss: ", mean_loss)
output_file = args["output-file"]
of = jldopen(output_file, "w")
of["verify_mean_loss"] = mean_loss
of["verify_acc"] = acc
of["verify_roc_curve"] = roc_curve
of["verify_preds"] = preds
of["verify_targets"] = targets
of["best_ps"] = best_ps
of["st"] = st
of["train_losses"] = train_losses
of["test_losses"] = test_losses
of["args"] = args
close(of)

# binary model is pointless when we have only one linear output
# also, binarise isn't implemented for pheno model.