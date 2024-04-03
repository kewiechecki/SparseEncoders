# SAE implementation using a partitoning submodel.
using Flux, Functors
using ProgressMeter

abstract type SparseClassifier <: SparseEncoder end

struct PSAE <: SparseClassifier
    sae::SparseEncoder
    partitioner::Chain
end
@functor PSAE

function PSAE(m::Integer,d::Integer,k::Integer,σ=relu)
    sae = SAE(m,d,σ)
    partitioner = Chain(Dense(m => k,σ))
    return PSAE(sae, partitioner)
end

function PSAE(m::Integer,d::Integer,k::Integer,
              σ_sae::Function,σ::Function)
    sae = SAE(m,d,σ_sae)
    partitioner = Chain(Dense(m => k,σ))
    return PSAE(sae, partitioner)
end

function encode(M::PSAE,X)
    return encode(M.sae,X)
end

function cluster(M::PSAE,X)
    return (softmax ∘ M.partitioner)(X)
end

function centroid(M::PSAE,X)
    E = encode(M,X)
    K = cluster(M,X)
    Ksum = sum(K,dims=2)

    C = K * E' ./ Ksum
    return C'
end

function partition(M::PSAE,X)
    return (pwak ∘ cluster)(M,X)
end

function encodepred(M::PSAE,X)
    E = encode(M,X)
    P = partition(M,X)
    return (P * E')'
end

function diffuse(M::PSAE,X)
    return encodepred(M,X)
end

function decode(M::PSAE,X)
    return decode(M.sae,X)
end


function (M::PSAE)(X::AbstractMatrix)
    Ehat = encodepred(M,X)
    return decode(M,Ehat)
end

    
function L1(M::PSAE,α,x)
    c = encodepred(M,x)
    return α * sum(abs.(c))
end

function L2(M::PSAE,lossfn,x,y)
    return lossfn(M(x),y)
end

function loss_PSAE(M::PSAE,α,lossfn,x,y)
    x = gpu(x)
    y = gpu(y)
    return L1(M,α,x) + L2(M,lossfn,x,y)
end


function train!(sae::Union{SAE,PSAE},
                M_outer,
                α::AbstractFloat,
                loader::Flux.DataLoader,
                opt::Flux.Optimiser,
                epochs::Integer,
                lossfn::Function,
                log,
                path="")
    if length(path) > 0
        mkpath(path)
    end
    @showprogress map(1:epochs) do _
        map(loader) do (x,y)
            f = loss_SAE(M_outer,α,lossfn,x)
            state = Flux.setup(opt,sae)
            l,∇ = Flux.withgradient(f,sae)
            Flux.update!(state,sae,∇[1])
            push!(log,l)
        end
    end
    if length(path) > 0
        savemodel(sae,path*"final")
        Tables.table(log) |> CSV.write(path*"loss.csv")
    end
end

function train!(sae::SparseEncoder,
                M_outer,
                α::AbstractFloat,
                loader::Flux.DataLoader,
                opt::Flux.Optimiser,
                epochs::Integer,
                lossfn::Function;
                path="")
    if length(path) > 0
        mkpath(path)
    end
    log=[]
    @showprogress map(1:epochs) do _
        map(loader) do (x,y)
            f = loss_SAE(M_outer,α,lossfn,x)
            state = Flux.setup(opt,sae)
            l,∇ = Flux.withgradient(f,sae)
            Flux.update!(state,sae,∇[1])
            push!(log,l)
        end
    end
    if length(path) > 0
        savemodel(sae,path*"/final")
        Tables.table(log) |> CSV.write(path*"/loss.csv")
    end
    return log
end


function train!(sae::SparseEncoder,
                M_outer,
                loader::Flux.DataLoader,
                opt::Flux.Optimiser,
                epochs::Integer,
                lossfn::Function,
                log,
                path="")
    if length(path) > 0
        mkpath(path)
    end
    @showprogress map(1:epochs) do _
        map(loader) do (x,y)
            f = loss_SAE(M_outer,lossfn,x)
            state = Flux.setup(opt,sae)
            l,∇ = Flux.withgradient(f,sae)
            Flux.update!(state,sae,∇[1])
            push!(log,l)
        end
    end
    if length(path) > 0
        savemodel(M,path*"/final")
        Tables.table(log) |> CSV.write(path*"/loss.csv")
    end
end
