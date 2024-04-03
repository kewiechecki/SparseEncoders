using Flux, Functors, Zygote

struct SparseDict
    dict
end
@functor SparseDict

function SparseDict(d::Integer,k::Integer)
    dict = randn(Float32,d,k)
    return SparseDict(dict)
end

function (M::SparseDict)(K)
    return (K' * M.dict')'
end

function Zygote.pullback(M::SparseDict, K::AbstractArray)
    E = (K' * M.dict')'  # Forward pass

    function B(Δ)
        # Transpose Δ since E was transposed in the forward pass
        Δ = Δ'
        # Gradient w.r.t M.dict, which is Δ * K
        ∇_dict = Δ * K
        # Gradient w.r.t. K, which is M.dict * Δ
        ∇_K = M.dict * Δ
        # Return the gradients in the expected structure by Zygote
        return (dict=∇_dict,), ∇_K
    end
    
    return E, B
end

struct DictEnc <: SparseClassifier
    dict::SparseDict
    classifier::Chain
    decoder::Chain
end
@functor DictEnc

function DictEnc(classifier::Chain,decoder::Chain,m::Integer,k::Integer)
    dict = SparseDict(d,k)
    return DictEnc(dict,classifier,decoder)
end

function DictEnc(m::Integer,d::Integer,k::Integer,σ=relu)
    dict = SparseDict(d,k)
    classifier = Chain(Dense(m => k,σ))
    decoder = Chain(Dense(d => m,σ))
    return DictEnc(dict,classifier,decoder)
end

function cluster(M::DictEnc,X)
    return (softmax ∘ M.classifier)(X)
end

function dict(M::DictEnc)
    return M.dict.dict
end

function encode(M::DictEnc,X)
    return diffuse(M,X)
end

function diffuse(M::DictEnc,X)
    K = cluster(M,X)
    return M.dict(K)
end

function centroid(M::DictEnc,X)
    return M.dict.dict
end

function decode(M::DictEnc,E)
    return M.decoder(E)
end

function (M::DictEnc)(X)
    K = cluster(M,X)
    E = M.dict(K)
    return decode(M,E)
end
