struct DistEnc <: SparseEncoder
    encoder::Chain
    decoder::Chain
    metric::Function
end
@functor DistEnc

function DistEnc(m::Integer,d::Integer,σ::Function,metric::Function)
    θ = Chain(Dense(m => d,σ))
    ϕ = Chain(Dense(d => m,σ))
    return DistEnc(θ,ϕ,metric)
end

function encode(M::SparseEncoder,X)
    return M.encoder(X)
end

function decode(M::SparseEncoder,E)
    return M.decoder(E)
end

function dist(M::SparseEncoder,E)
    return M.metric(E)
end

function kern(M::DistEnc,E)
    D = dist(M,E)
    return wak(D)
end

function diffuse(M::DistEnc,X::AbstractArray{<:AbstractFloat})
    E = encode(M,X)
    D = kern(M,E)
    return (D * E')'
end

function (M::DistEnc)(X::AbstractArray{<:AbstractFloat})
    E = diffuse(M,X)
    return decode(M,E)
end

struct DistPart <: SparseClassifier
    encoder::Chain
    decoder::Chain
    classifier::Chain
    metric::Function
end
@functor DistPart

function DistPart(m::Integer,d::Integer,k::Integer,σ::Function,metric::Function)
    θ = Chain(Dense(m => d,σ))
    ϕ = Chain(Dense(d => m,σ))
    ψ = Chain(Dense(m => k,σ))
    return DistPart(θ,ϕ,ψ,metric)
end

function cluster(M::DistPart,X)
    return (softmax ∘ M.classifier)(X)
end

function centroid(M::DistPart,X)
    E = encode(M,X)
    K = cluster(M,X)
    Ksum = sum(K,dims=2)

    C = K * E' ./ Ksum
    return C'
end

function partition(M::DistPart,X)
    return (pwak ∘ cluster)(M,X)
end

function kern(M::DistPart,X::AbstractArray{<:AbstractFloat})
    E = encode(M,X)
    D = dist(M,E)
    P = partition(M,X)
    return wak(P .* D)
end

function diffuse(M::DistPart,X::AbstractArray{<:AbstractFloat})
    E = encode(M,X)
    D = dist(M,E)
    P = partition(M,X)
    G = wak(P .* D)
    return (G * E')'
end
