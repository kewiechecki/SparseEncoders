using Flux,Functors

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

function encode(M::DistEnc,X)
    return M.encoder(X)
end

function decode(M::DistEnc,E)
    Ê = diffuse(M,E)
    return M.decoder(Ê)
end

function dist(M::DistEnc,E)
    return M.metric(E)
end

function kern(M::DistEnc,E)
    D = dist(M,E)
    return wak(D)
end

function diffuse(M::DistEnc,E::AbstractArray{<:AbstractFloat})
    #E = encode(M,X)
    D = kern(M,E)
    return (D * E')'
end

function (M::DistEnc)(X::AbstractArray{<:AbstractFloat})
    E = encode(M,X)
    return decode(M,E)
end
