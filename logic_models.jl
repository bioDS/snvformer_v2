using NNlib
using Zygote
using Optimisers
using Functors
using Lux
using DataLoaders
using Flux.Losses
using Random
using Statistics
using Distributions
using ChainRulesCore
using Plots

rng = Random.default_rng()

## range limiting functions

# Simple cap, works in simple cases at least.
# notably often fails when negations is used in dnf test,
# (inputs often lead to all 0.0 or all 1.0)
rlf1(x::Real) = min.(1,max.(0,x))

# Differentiable below 0 and above 1, might help
# with more complicated cases (or might not).
# Also works in testing with simple problems.
# resolves all 0.0 and all 1.0 problem above
# in dnf w/ negation testing.
function rlf2(x::Real)
    T = eltype(x)
    if x .< 0.2
        exp.(x + T(log.(0.2)) - T(0.2))
    elseif x > 0.8
        T(1.0) - exp((T(1.0) - x) + T(log(0.2) - 0.2))
    else
        x
    end
end

@scalar_rule rlf1(x::Real) (
    if x < 0 || x > 1
        zero(x)
    else
        one(x)
    end
)

@scalar_rule rlf2(x::Real) (
    T = eltype(x);
    if x .< 0.2 
        exp.(x + T(log.(0.2)) - T(0.2))
    elseif x .> 0.8
        exp.((1-x) + T(log.(0.2)) - T(0.2))
    else
        one(x)
    end
)

# choose function here
rlf = rlf2

function plot_rlf()
    x = range(-1, 2, length=300)
    y = rlf2.(x)
    pl = plot(x, y, legend=false)
    savefig(pl, "rlf_fig.pdf")
end

## Fuzzy-logic definitions
function and_k(x, weights)
    T = eltype(x)
    w = T(2)*softmax(weights)
    #sigmoid.(((rx' * w .- 1).*10) .- 1.0)
    #sigmoid.(((x' * w .- 1).*10) .- 1.0)
    # min.(1,max.(0,(x' * w) .- 1.0))
    # rlf.((x' * w) .- 1.0)
    rlf.((x' * w) .- (sum(abs.(w)) - T(1.0)))
    # rlf.((x' * w) ./ (sum(abs.(w))))
end


function or_k(x, weights)
    w = 2*softmax(weights)
    # w = weights
    # sigmoid.(((x' * w) .* 10) .- 1.0)
    rlfin = x * w
    rlf.(rlfin)
end

function binary_and_k(x, weights)
    w = weights
    # min.(1,max.(0,(x' * w) .- 1.0))
    min.(1,max.(0,(x' * w) .- ((sum(abs.(w))) - 1.0)))
    # rlf.((x' * w) ./ (sum(abs.(w))))
end

function binary_or_k(x, weights)
    w = weights
    min.(1,max.(0,(x * w)))
end

function not(x)
    T = eltype(x)
    -x .+ T(1.0)
end

function top_two(vec::Vector{Float32})
    mv1 = -Inf32
    mi1 = 0
    mv2 = -Inf32
    mi2 = 0
    for (ind, val) in enumerate(vec)
        if val >= mv1
            mv2 = mv1
            mi2 = mi1
            mv1 = val
            mi1 = ind
        elseif val >= mv2
            mv2 = val
            mi2 = ind
        end
    end
    return (mv1, mv2), (mi1, mi2)
end

binary_and(x1, x2) = max(0,(x1 + x2 - 1.0))
binary_or(x1, x2) = min(1,(x1 + x2))

function dnf_model_init(input_len)
    and1_weights = Lux.glorot_normal(rng, 2*input_len)
    and2_weights = Lux.glorot_normal(rng, 2*input_len)
    or_weights = Lux.glorot_normal(rng, 4)
    # and1_weights = Lux.glorot_normal(rng, input_len)
    # and2_weights = Lux.glorot_normal(rng, input_len)
    # or_weights = Lux.glorot_normal(rng, 2)
    ((and1_weights=and1_weights, and2_weights=and2_weights, or_weights))
end

# ((x_? ∧ x_?) ∨ (x_? ∧ x_?)
function dnf_model(rx, state)
    maybe_not_rx = vcat(rx, not(rx))
    and1_weights, and2_weights, or_weights = state
    conj1 = and_k(maybe_not_rx, and1_weights)
    conj2 = and_k(maybe_not_rx, and2_weights)
    # disj = or_k(hcat(conj1, conj2)', or_weights)
    disj = or_k(hcat(conj1, conj2, not(conj1), not(conj2))', or_weights)
    disj
end

# N.B., negation in or is not strictly necessary
function scalable_dnf_model_init(num_and, seq_len)
    and_weights = [Lux.glorot_normal(rng, 2*seq_len) for i in range(1,num_and)]
    or_weights = Lux.glorot_normal(rng, num_and)
    ((and_weights = and_weights, or_weights = or_weights))
end

function scalable_dnf_model(rx, state)
    and_weights, or_weights = state
    num_and = length(and_weights)
    #and_outputs = [and_k(vcat(rx, not(rx)), and_weights[i])[1] for i in range(1,num_and)]
    #or_output = or_k(vcat(and_outputs), or_weights)
    maybe_not_x = vcat(rx, not(rx))
    # and_outputs = hcat(map(w -> and_k(maybe_not_x, w), and_weights)...)'
    and_outputs = and_k(maybe_not_x, and_weights)
    or_output = or_k(vcat(and_outputs), or_weights)
    or_output
end

function binary_scalable_dnf_model(rx, state)
    and_weights, or_weights = state
    num_and = length(and_weights)
    and_outputs = [binary_and_k(vcat(rx, not(rx)), and_weights[i])[1] for i in range(1,num_and)]
    or_output = binary_or_k(vcat(and_outputs), or_weights)
    or_output
end

# multiscale is wip, maybe not necessary.
function multiscale_dnf_model_init(max_num_and, seq_len)
    dnf_models = []
    for log_num_and in range(1, floor(log2(max_num_and)))
        num_and = Int(2^log_num_and)
        new_model_state = scalable_dnf_model_init(num_and, seq_len)
        push!(dnf_models, new_model_state)
    end
    dnf_models
end

function multiscale_dnf_model(rx, full_state)
    dnf_models = full_state
    results = [scalable_dnf_model(rx, substate) for substate in dnf_models]
    results
end

function multiscale_dnf_model(rx, state)
end

function assign_vector_two!(s_vec, b_vec)
    _, inds = top_two(s_vec)
    for ind in inds
        b_vec[ind] = 1
    end
end

function assign_vector_all!(s_vec, b_vec)
    inds = findall(s_vec .> 0)
    for ind in inds
        b_vec[ind] = 1
    end
end

function assign_vector_all_weighted!(s_vec, b_vec)
    inds = findall(s_vec .> 0)
    for ind in inds
        b_vec[ind] = s_vec[ind]
    end
end

function assign_vector_set_two!(state, binary_state)
    for (vec_ind, vec) in enumerate(state)
        assign_vector_two!(vec, binary_state[vec_ind])
    end
end

function assign_vector_set_all!(state, binary_state)
    for (vec_ind, vec) in enumerate(state)
        assign_vector_all!(vec, binary_state[vec_ind])
    end
end

function assign_vector_set_all_weighted!(state, binary_state)
    for (vec_ind, vec) in enumerate(state)
        assign_vector_all_weighted!(vec, binary_state[vec_ind])
    end
end

# N.B.
# quick testing suggests that ands should use vector_set_two
# whereas or should use vector_set_all (not weighted)
function binarise_scalable_dnf_model(state)
    and_weights, _ = state
    num_and = length(and_weights)
    seq_len = Int(length(and_weights[1])//2)
    binary_state = scalable_dnf_model_init(num_and, seq_len)
    binary_state[2][:] .= 0.0
    for i in range(1,length(binary_state[1]))
        binary_state[1][i][:] .= 0.0f0
    end
    assign_vector_set_all!(state[1], binary_state[1])
    assign_vector_all!(state[2], binary_state[2])
    binary_state
end

function binary_dnf_model(rx, state)
    maybe_not_rx = vcat(rx, not(rx))
    and1_weights, and2_weights, or_weights = state
    conj1 = binary_and_k(maybe_not_rx, and1_weights)
    conj2 = binary_and_k(maybe_not_rx, and2_weights)
    # disj = binary_or_k(hcat(conj1, conj2)', or_weights)
    disj = binary_or_k(hcat(conj1, conj2, not(conj1), not(conj2))', or_weights)
    disj
end

function binarise_model2(state)
    binary_state = dnf_model_init(Int(length(state[1])//2))
    # binary_state = dnf_model_init(Int(length(state[1])))
    for i in range(1,length(binary_state))
        binary_state[i][1:end] .= 0
    end
    for (vec_ind, vec) in enumerate(state)
        inds = findall(vec .> 0)
        for ind in inds
            binary_state[vec_ind][ind] = 1
        end
    end
    binary_state
end

function binarise_model(state)
    binary_state = dnf_model_init(Int(length(state[1])//2))
    # binary_state = dnf_model_init(Int(length(state[1])))
    for i in range(1,length(binary_state))
        binary_state[i][1:end] .= 0
    end
    for (vec_ind, vec) in enumerate(state)
        _, inds = top_two(vec)
        for ind in inds
            binary_state[vec_ind][ind] = 1
        end
    end
    binary_state
end

function train(model, data, model_state, opt_state, num_epochs=50)
    for e in range(1,num_epochs)
        for (rx, ryb) in data
            loss, pb = pullback(model_state) do ms
                pred = model(rx, ms)
                loss = binarycrossentropy(pred, ryb)
                loss
            end
            (∇params,) = pb(loss)
            opt_state, model_state = Optimisers.update(opt_state, model_state, ∇params)
        end
        if e % (num_epochs/10) == 0
            losses = Vector{Float64}()
            for (rx, ryb) in data
                pred = model(rx, model_state)
                loss = binarycrossentropy(pred, ryb)[1]
                append!(losses, loss)
            end
            mean_loss = mean(losses)
            @show mean_loss
        end
    end
    opt_state, model_state
end

function shannon_entropy(vec::Vector{<:AbstractFloat})
    smv = softmax(vec)
    -sum(smv .* log.(smv))
end

function dist_from_two_entropy(vec::Vector{<:AbstractFloat})
    ent = shannon_entropy(vec)
    goal_vec = vcat(1.0, 1.0, zeros(size(vec)[1]-2))
    goal_ent = shannon_entropy(goal_vec)
    dist = mse(ent, goal_ent)
    dist
end

function train_entropy_loss(model, data, model_state, opt_state, num_epochs=50)
    for e in range(1,num_epochs)
        for (rx, ryb) in data
            loss, pb = pullback(model_state) do ms
                pred = model(rx, ms)
                sum_weight_entropy = dist_from_two_entropy(ms)
                # sum_weight_entropy = mean([dist_from_two_entropy[i] for i in ms])
                #sum_weight_entropy = mean([dist_from_two_entropy(ms.and_weights),
                #    dist_from_two_entropy(ms.nv_or_weights),
                #    dist_from_two_entropy(ms.final_and_weights)])
                loss = binarycrossentropy(pred, ryb) + 0.2 * sum_weight_entropy
                loss
            end
            ∇params = pb(loss)[1]
            opt_state, model_state = Optimisers.update(opt_state, model_state, ∇params)
        end
        if e % (num_epochs/10) == 0
            losses = Vector{Float64}()
            for (rx, ryb) in data
                pred = model(rx, model_state)
                loss = binarycrossentropy(pred, ryb)[1]
                append!(losses, loss)
            end
            mean_loss = mean(losses)
            @show mean_loss
        end
    end
    opt_state, model_state
end


function eval_model(dataset, model, model_state)
    losses = Vector{Float64}()
    model_correct = 0
    for (rx, ryb) in dataset
        pred = model(rx, model_state)
        model_correct += sum((pred .> 0.5) .== ryb)
        loss = binarycrossentropy(pred, ryb)[1]
        append!(losses, loss)
    end
    @show mean(losses)
    @show model_correct/dataset.data.count
end

function combo_model(rx, state; binary=false)
    (lin_model, lin_ps, lin_st), dnf_state = state
    lin_out = lin_model(rx, lin_ps, lin_st)[1]
    if binary
        pred = binary_scalable_dnf_model(lin_out, dnf_state)
    else
        pred = scalable_dnf_model(lin_out, dnf_state)
    end
    pred
end

function combo_model_init(seq_len, num_linear_out, num_ands)
    lin_model = Dense(seq_len, num_linear_out, sigmoid)
    lin_ps, lin_st = Lux.setup(rng, lin_model)
    dnf_state = scalable_dnf_model_init(num_ands, num_linear_out)
    ((lin_model, lin_ps, lin_st), dnf_state)
end

struct LuxDnfLayer{F1} <: Lux.AbstractExplicitLayer
    num_inputs::Int
    num_ands::Int
    init_fn::F1
end

function LuxDnfLayer(num_ands, num_inputs, init_weight_fn=Lux.glorot_uniform)
    LuxDnfLayer{typeof(init_weight_fn)}(
        num_inputs,
        num_ands,
        init_weight_fn
    )
end

function Lux.initialparameters(rng::AbstractRNG, l::LuxDnfLayer)
    # and_weights = [l.init_fn(rng, 2*l.num_inputs) for i in range(1,l.num_ands)]
    and_weights = l.init_fn(rng, 2*l.num_inputs, l.num_ands)
    or_weights = l.init_fn(rng, l.num_ands)
    return (and_weights = and_weights, or_weights = or_weights)
end

function Lux.parameterlength(l::LuxDnfLayer)
    num_and_params = l.num_ands * (2*l.num_inputs)
    num_or_params = l.num_ands
    num_and_params + num_or_params
end

function (l::LuxDnfLayer)(x::AbstractMatrix, ps, st::NamedTuple)
    maybe_not_x = vcat(x, not(x))
    # and_outputs = hcat(map(w -> and_k(maybe_not_x, w), ps.and_weights)...)'
    and_outputs = and_k(maybe_not_x, ps.and_weights)
    or_output = or_k(vcat(and_outputs), ps.or_weights)
    or_output
end

struct LuxBinaryDnfLayer <: Lux.AbstractExplicitLayer
    num_ands::Int
    num_inputs::Int
end

function LuxBinaryDnfLayer(num_ands, num_inputs)
    LuxBinaryDnfLayer(
        num_ands,
        num_inputs,
    )
end

function Lux.initialparameters(rng::AbstractRNG, l::LuxBinaryDnfLayer)
    and_weights = zeros(2*l.num_inputs, l.num_ands)
    or_weights = zeros(l.num_ands)
    # and_weights = [zeros(rng, 2*l.num_inputs) for i in range(1,l.num_ands)]
    # or_weights = zeros(rng, l.num_ands)
    return (and_weights = and_weights, or_weights = or_weights)
end

function Lux.parameterlength(l::LuxBinaryDnfLayer)
    num_and_params = l.num_ands * (2*l.num_inputs)
    num_or_params = l.num_ands
    num_and_params + num_or_params
end

function (l::LuxBinaryDnfLayer)(x::AbstractMatrix, ps, st::NamedTuple)
    maybe_not_x = vcat(x, not(x))
    and_outputs = binary_and_k(maybe_not_x, ps.and_weights)
    or_output = binary_or_k(vcat(and_outputs), ps.or_weights)
    or_output
end


Lux.statelength(::LuxDnfLayer) = 0
Lux.statelength(::LuxBinaryDnfLayer) = 0

function binarise_lux_dnf_layer(ps, st, hyper_model)
    num_and = hyper_model.num_ands
    # seq_len = Int(size(ps.and_weights)[2]//2)
    bmodel = LuxBinaryDnfLayer(num_and, hyper_model.num_linear)
    bps, bst = Lux.setup(rng, bmodel)
    bps = deepcopy(ps)
    # assign_vector_set_all!(ps.and_weights, bps.and_weights)
    # assign_vector_all!(ps.or_weights, bps.or_weights)
    bps, bst
end

function gt_min_one(weights)
    w = weights .> 0
    if sum(w) == 0
        _, max_ind = findmax(w)
        w[max_ind] = 1.0
    end
    return w
end

function binarise_lux_pheno_combo_model(ps, st, hyper_model)
    num_linear = hyper_model.num_linear
    seq_len = size(ps.linear[1])[2]
    num_ands = hyper_model.num_ands
    num_pheno_thresholds = hyper_model.num_pheno_thresholds
    # num_dnf_inputs = num_lin_out + 4*num_thresholds + 1
    double_num_dnf_inputs, _ = size(ps.dnf.and_weights)
    num_dnf_inputs = Int(double_num_dnf_inputs // 2)
    # dnf_bps, dnf_bst = binarise_lux_dnf_layer(ps.dnf, st.dnf, hyper_model)
    bmodel = LuxBinaryPhenoComboModel(Dense(seq_len, num_linear),
                LuxBinaryDnfLayer(num_ands, num_dnf_inputs),
                LuxThresholdLayer(num_pheno_thresholds, 20, 45), # bmi
                LuxThresholdLayer(num_pheno_thresholds, 40, 70)) # age
    # bmodel = LuxBinaryPhenoComboModel(Dense(seq_len, num_linear), LuxBinaryDnfLayer(num_dnf_inputs, num_ands))
    bps, bst = Lux.setup(rng, bmodel)
    bps.linear.weight .= gt_min_one(ps.linear.weight)
    bps.linear.bias .= gt_min_one(ps.linear.bias)
    # bps.dnf.and_weights .= ps.dnf.and_weights
    # bps.dnf.or_weights .= ps.dnf.or_weights
    bps.dnf.and_weights .= gt_min_one(ps.dnf.and_weights)
    bps.dnf.or_weights .= gt_min_one(ps.dnf.or_weights)
    bmodel, bps, bst
end

struct LuxComboModel{F1} <: Lux.AbstractExplicitContainerLayer{(:linear, :dnf)}
    linear::Dense
    dnf::LuxDnfLayer{F1}
end

function (model::LuxComboModel)(x::AbstractMatrix, ps, st::NamedTuple)
    linear_out = model.linear(x, ps.linear, st.linear)[1]
    dnf_out = model.dnf(linear_out, ps.dnf, st.dnf)
    dnf_out
end

struct LuxBinaryComboModel <: Lux.AbstractExplicitContainerLayer{(:linear, :dnf)}
    linear::Dense
    dnf::LuxBinaryDnfLayer
end

function (model::LuxBinaryComboModel)(x::AbstractMatrix, ps, st::NamedTuple)
    linear_out = model.linear(x, ps.linear, st.linear)[1]
    dnf_out = model.dnf(linear_out, ps.dnf, st.dnf)
    dnf_out
end


function one_epoch(data, model, ps, st, opt_state; use_gpu=false)
    for (rx, ryb) in data
        if use_gpu
            rx = rx |> gpu
            ryb = ryb |> gpu
        end
        loss, pb = pullback(ps) do ps
            # ryb2 = reshape(ryb, (1,size(ryb)...))
            # pred = model(rx, ps, st)
            pred = Lux.apply(model, rx, ps, st)
            loss = binarycrossentropy(pred, ryb)
            loss
        end
        (∇params,) = pb((loss))
         opt_state, ps = Optimisers.update(opt_state, ps, ∇params)
    end
    opt_state, ps 
end

function one_epoch_classic(data, model, model_state, opt_state)
    for (rx, ryb) in data
        loss, pb = pullback(model_state) do ms
            # ryb2 = reshape(ryb, (1,size(ryb)...))
            # pred = model(rx, ps, st)
            pred = model(rx, model_state)
            loss = binarycrossentropy(pred, ryb)
            loss
        end
        (∇params,) = pb((loss))
        opt_state, model_state = Optimisers.update(opt_state, model_state, ∇params)
    end
    opt_state, model_state
end


function lux_train(train_data, test_data, num_epochs, model, ps, st, opt_state; use_gpu=false)
    epoch_test_losses = []
    epoch_train_losses = []
    epoch_end_ps = []
    for e in range(1, num_epochs)
        for (rx, ryb) in train_data
            if use_gpu
                rx = rx |> gpu
                ryb = ryb |> gpu
            end
            loss, pb = pullback(ps) do ps
                pred = Lux.apply(model, rx, ps, st)
                loss = binarycrossentropy(pred, ryb)
                loss
            end
            (∇params,) = pb((loss))
            opt_state, ps = Optimisers.update(opt_state, ps, ∇params)
        end
        if e % (num_epochs/10) == 0
            losses = Vector{Float64}()
            for (rx, ryb) in train_data
                if use_gpu
                    rx = rx |> gpu
                    ryb = ryb |> gpu
                end
                pred = Lux.apply(model, rx, ps, st)
                loss = binarycrossentropy(pred, ryb)
                append!(losses, loss)
            end
            mean_loss = mean(losses)
            @show mean_loss
        end
        test_losses = Vector{Float64}()
        for (rx, ryb) in test_data
            if use_gpu
                rx = rx |> gpu
                ryb = ryb |> gpu
            end
            pred = Lux.apply(model, rx, ps, st)
            loss = binarycrossentropy(pred, ryb)
            append!(test_losses, loss)
        end
        test_mean_loss = mean(test_losses)
        append!(epoch_test_losses, test_mean_loss)
        train_losses = Vector{Float64}()
        for (rx, ryb) in train_data
            if use_gpu
                rx = rx |> gpu
                ryb = ryb |> gpu
            end
            pred = Lux.apply(model, rx, ps, st)
            loss = binarycrossentropy(pred, ryb)
            append!(train_losses, loss)
        end
        train_mean_loss = mean(train_losses)
        append!(epoch_train_losses, train_mean_loss)
        push!(epoch_end_ps, ps)
    end
    opt_state, ps, epoch_train_losses, epoch_test_losses, epoch_end_ps
end


function lux_eval_model(dataset, model, ps, st; use_gpu=false)
    losses = Vector{Float64}()
    model_correct = 0
    for (rx, ryb) in dataset
        if use_gpu
            rx = rx |> gpu
            # ryb = ryb |> gpu
        end
        pred = model(rx, ps, st) |> cpu
        model_correct += sum((pred .> 0.5) .== ryb)
        loss = binarycrossentropy(pred, ryb)[1]
        append!(losses, loss)
    end
    (mean(losses), model_correct/size(dataset.data.data[2])[1])
end

function binarise_combo_model(state)
    (lin_model, lin_ps, lin_st), dnf_state = state
    binary_dnf_state = binarise_scalable_dnf_model(dnf_state)
    ((lin_model, lin_ps, lin_st), binary_dnf_state)
end# Pkg.add(["NNlib", "Zygote", "Optimisers", "Functors", "Lux", "DataLoaders", "Random", "Statistics", "Distributions", "ChainRulesCore", "ChainRulesTestUtils"])

function above_below_thresholds(vars, thresholds)
    nomore = hcat([vars .<= t for t in thresholds]...)
    atleast = hcat([vars .>= t for t in thresholds]...)
    combined = hcat(nomore, atleast)
    return combined'
end

function singlesex_pheno_model_init(num_and, seq_len, num_thresholds)
    # range of about 60
    # use n steps in the range 15 - 35
    bmi_range_min = 20
    bmi_range_max = 45
    bmi_thresholds = ([i for i in 1:num_thresholds] ./ num_thresholds) .* (bmi_range_max - bmi_range_min) .+ bmi_range_min
    age_range_min = 40
    age_range_max = 70
    age_thresholds = ([i for i in 1:num_thresholds] ./ num_thresholds) .* (age_range_max - age_range_min) .+ age_range_min

    num_inputs = 2*seq_len + 4*num_thresholds
    and_weights = [Lux.glorot_normal(rng, num_inputs) for i in range(1,num_and)]
    or_weights = Lux.glorot_normal(rng, num_and)
    ((and_weights = and_weights, or_weights = or_weights, bmi_thresholds=bmi_thresholds, age_thresholds=age_thresholds))
end

# based on scalable_dnf_model, but includes phenotypes directly in logical layer, skipping linear combinations.
function singlesex_pheno_model(state, snvs, bmi, age; binary=false)
    if binary
        and_func = binary_and_k
        or_func = binary_or_k
    else
        and_func = and_k
        or_func = or_k
    end
    and_weights, or_weights, bmi_thresholds, age_thresholds = state
    # linear combinations of presence/absence of snv types.
    maybe_not_x = vcat(snvs, not(snvs))

    # binary variables for bmi and age above and below certain thresholds
    bmi_vec = above_below_thresholds(bmi, bmi_thresholds)
    age_vec = above_below_thresholds(age, age_thresholds)
    
    combined_vec = vcat(age_vec, bmi_vec, maybe_not_x)

    and_outputs = hcat(map(w -> and_func(combined_vec, w), and_weights)...)'
    or_output = or_func(vcat(and_outputs), or_weights)
    or_output
end

struct LuxThresholdLayer <: Lux.AbstractExplicitLayer
    num_thresholds::Int
    range_min::Int
    range_max::Int
end

function Lux.initialparameters(rng::AbstractRNG, l::LuxThresholdLayer)
    num_thresholds = l.num_thresholds
    thresholds = ([i for i in 1:num_thresholds] ./ num_thresholds) .* (l.range_max - l.range_min) .+ l.range_min
    thresholds = reshape(thresholds, 1, :)

    return (thresholds=thresholds,)
end

Lux.initialstates(::AbstractRNG, ::LuxThresholdLayer) = NamedTuple()

function (l::LuxThresholdLayer)(vars::AbstractVector, ps, st::NamedTuple)
    tv = reshape(vars, :, 1)
    nomore = broadcast(<=, ps.thresholds, tv)
    atleast = broadcast(>=, ps.thresholds, tv)
    combined = hcat(nomore, atleast)
    return combined'
end

struct LuxPhenoComboModel <: Lux.AbstractExplicitContainerLayer{(:linear, :dnf, :bmi_thresholds, :age_thresholds)}
    linear::Dense
    dnf::LuxDnfLayer
    bmi_thresholds::LuxThresholdLayer
    age_thresholds::LuxThresholdLayer
end

function LuxPhenoComboModel_init(seq_len, num_lin_out, num_and_layers, num_thresholds)
    num_dnf_inputs = num_lin_out + 4*num_thresholds + 1
    model = LuxPhenoComboModel(Dense(seq_len, num_lin_out),
                LuxDnfLayer(num_dnf_inputs, num_and_layers, Lux.glorot_normal),
                LuxThresholdLayer(num_thresholds, 20, 45), # bmi
                LuxThresholdLayer(num_thresholds, 40, 70)) # age
    model
end

function (model::LuxPhenoComboModel)(snvs::AbstractMatrix, bmi::AbstractVector, age::AbstractVector, male::AbstractVector, ps, st::NamedTuple)
    bmi_vec = @async model.bmi_thresholds(bmi, ps.bmi_thresholds, st.bmi_thresholds)
    age_vec = @async model.age_thresholds(age, ps.age_thresholds, st.age_thresholds)
    linear_out = @async model.linear(snvs, ps.linear, st.linear)[1]
    # logical combination of bmi, age, and linear-snv-thresholds
    combined_vec = vcat(fetch(age_vec), fetch(bmi_vec), reshape(male,1,size(male)...), fetch(linear_out))
    dnf_out = model.dnf(combined_vec, ps.dnf, st.dnf)
    dnf_out
end

struct LuxBinaryPhenoComboModel <: Lux.AbstractExplicitContainerLayer{(:linear, :dnf, :bmi_thresholds, :age_thresholds)}
    linear::Dense
    dnf::LuxBinaryDnfLayer
    bmi_thresholds::LuxThresholdLayer
    age_thresholds::LuxThresholdLayer
end

# function LuxBinaryPhenoComboModel_init(seq_len, num_lin_out, num_and_layers, num_thresholds)
#     num_dnf_inputs = num_lin_out + 4*num_thresholds + 1
#     model = LuxPhenoComboModel(Dense(seq_len, num_lin_out),
#                 LuxBinaryDnfLayer(num_dnf_inputs, num_and_layers),
#                 LuxThresholdLayer(num_thresholds, 20, 45), # bmi
#                 LuxThresholdLayer(num_thresholds, 40, 70)) # age
#     model
# end

function (model::LuxBinaryPhenoComboModel)(snvs::AbstractMatrix, bmi::AbstractVector, age::AbstractVector, male::AbstractVector, ps, st::NamedTuple)
    bmi_vec = @async model.bmi_thresholds(bmi, ps.bmi_thresholds, st.bmi_thresholds)
    age_vec = @async model.age_thresholds(age, ps.age_thresholds, st.age_thresholds)
    linear_out = @async model.linear(snvs, ps.linear, st.linear)[1]
    # logical combination of bmi, age, and linear-snv-thresholds
    combined_vec = vcat(fetch(age_vec), fetch(bmi_vec), reshape(male,1,size(male)...), fetch(linear_out))
    dnf_out = model.dnf(combined_vec, ps.dnf, st.dnf)
    dnf_out
end

struct LuxDualPhenoComboModel <: Lux.AbstractExplicitContainerLayer{(:male_model, :female_model)}
    male_model
    female_model
end

function LuxDualPhenoComboModel_init(seq_len, num_lin_out, num_and_layers, num_thresholds)
    male_model = LuxPhenoComboModel_init(seq_len, num_lin_out, num_and_layers, num_thresholds)
    female_model = LuxPhenoComboModel_init(seq_len, num_lin_out, num_and_layers, num_thresholds)
    return LuxDualPhenoComboModel(male_model, female_model)
end

function (model::LuxDualPhenoComboModel)(snvs::AbstractMatrix, bmi::AbstractVector, age::AbstractVector, male::AbstractVector, ps::NamedTuple, st::NamedTuple)
    m_out = model.male_model(snvs[:,male], bmi[male], age[male], ps.male_model, st.male_model)
    f_out = model.female_model(snvs[:,male .== false], bmi[male .== false], age[male .== false], ps.female_model, st.female_model)
    vec = Vector{Float64}()
    m_ind = 1
    f_ind = 1
    for m in male
        if m
            vec = vcat(vec, m_out[m_ind])
            m_ind += 1
        else
            vec = vcat(vec, f_out[f_ind])
            f_ind += 1
        end
    end
    vec

end

function binarise_singlesex_pheno_model(state)
    and_weights, or_weights, bmi_thresholds, age_thresholds = state
    num_and = length(and_weights)
    num_thresholds = length(bmi_thresholds)
    seq_len = Int((length(and_weights[1]) - 4*num_thresholds)//2)
    binary_state = singlesex_pheno_model_init(num_and, seq_len, num_thresholds)
    binary_state[2][:] .= 0.0
    for i in range(1,length(binary_state[1]))
        binary_state[1][i][:] .= 0.0f0
    end
    assign_vector_set_all!(state[1], binary_state[1])
    assign_vector_all!(state[2], binary_state[2])
    binary_state.bmi_thresholds[:] = bmi_thresholds
    binary_state.age_thresholds[:] = age_thresholds
    binary_state
end

function dual_pheno_model_init(num_and, seq_len, num_thresholds)
    m = singlesex_pheno_model_init(num_and, seq_len, num_thresholds)
    f = singlesex_pheno_model_init(num_and, seq_len, num_thresholds)
    (male=m, female=f)
end

function binarise_dual_pheno_model(model_state)
    m, f = model_state
    binarise_singlesex_pheno_model(m), binarise_singlesex_pheno_model(f)
end

function dual_pheno_model(model_state, snvs, bmi, age, male; binary=false)
    male_model_state, female_model_state = model_state
    
    m_out = singlesex_pheno_model(male_model_state, snvs[:,male], bmi[male], age[male]; binary=binary)
    f_out = singlesex_pheno_model(female_model_state, snvs[:,male .== false], bmi[male .== false], age[male .== false]; binary=binary)

    #vcat(m_out, f_out)

    vec = Vector{Float64}()
    m_ind = 1
    f_ind = 1
    for m in male
        if m
            vec = vcat(vec, m_out[m_ind])
            m_ind += 1
        else
            vec = vcat(vec, f_out[f_ind])
            f_ind += 1
        end
    end
    vec
end

binary_dual_pheno_model(model_state, snvs, bmi, age, male) = dual_pheno_model(model_state, snvs, bmi, age, male; binary=true)

function get_mean_loss(model, ps, st, dataset)
    losses = Vector{Float64}()
    for (snvs, gout, urate, bmi, height, age, male) in dataset
        pred = model(snvs |> to_device, bmi |> to_device, age |> to_device, male |> to_device, ps, st)
        loss = binarycrossentropy(pred, gout |> to_device)
        append!(losses, cpu(loss))
    end
    mean(losses)
end

function dual_pheno_train(model, train_data, test_data, ps, st, opt_state, num_epochs=50)
    train_losses = []
    test_losses = []
    epoch_parameters = []
    # @showprogress for e in range(1,num_epochs)
    for e in range(1,num_epochs)
        for (snvs, gout, urate, bmi, height, age, male) in train_data
            loss, pb = pullback(ps) do ps
                pred = model(snvs |> to_device, bmi |> to_device, age |> to_device, Float32.(male) |> to_device, ps, st)
                loss = binarycrossentropy(pred, gout |> to_device)
                loss
            end
            (∇params,) = pb(to_device(one(loss)))
            opt_state, ps = Optimisers.update(opt_state, ps, ∇params)
        end
        push!(epoch_parameters, cpu(deepcopy(ps)))
        # if e % (num_epochs/10) == 0
        train_mean_loss = get_mean_loss(model, ps, st, train_data)
        test_mean_loss = get_mean_loss(model, ps, st, test_data)
        append!(train_losses, train_mean_loss)
        append!(test_losses, test_mean_loss)
        println((train_mean_loss, test_mean_loss))
        # end
    end
    best_param_ind = argmin(test_losses)
    best_ps=epoch_parameters[best_param_ind]
    opt_state, ps, best_ps, train_losses, test_losses
end

function eval_dual_pheno_model(dataset, model, ps, st)
    losses = Vector{Float64}()
    model_correct = 0
    all_predictions = Vector{Float64}()
    all_true_gout = Vector{Bool}()
    for (snvs, gout, urate, bmi, height, age, male) in dataset
        pred = model(snvs |> to_device, bmi |> to_device, age |> to_device, male |> to_device, ps, st)
        model_correct += sum((cpu(pred) .> 0.5) .== gout)
        loss = binarycrossentropy(pred, gout |> to_device)[1]
        append!(losses, loss)
        append!(all_predictions, pred)
        append!(all_true_gout, gout)
    end
    roc_curve = ROCAnalysis.roc(all_predictions[all_true_gout .== false], all_predictions[all_true_gout .== true])
    # plot(roc_curve)
    ROCAnalysis.auc(roc_curve)
    mean(losses), model_correct/size(dataset.data.data[1])[2], roc_curve, all_predictions, all_true_gout
end