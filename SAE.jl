using Flux, Functors
abstract type SparseEncoder
end

struct Autoencoder <: SparseEncoder
    encoder
    decoder
end

function encode(M::Autoencoder,x)
    return M.encoder(x)
end

function diffuse(M::Autoencoder,X)
    return encode(M,X)
end

function decode(M::Autoencoder,E)
    return M.decoder(E)
end

function (M::Autoencoder)(x)
    return decode(M,encode(M,x))
end


struct SAE <: SparseEncoder
    weight::AbstractArray
    bias::AbstractArray
    σ::Function
end
# the magic line that makes everything "just work"
@functor SAE 

# constructor specifying i/o dimensions
function SAE(m,d,σ=relu)
    weight = randn(d,m)
    bias = randn(d)
    return SAE(weight,bias,σ)
end

#
function encode(M::SAE,x)
    return M.σ.(M.weight * x .+ M.bias)
end

function decode(M::SAE,c)
    return M.weight' * c
end

function diffuse(M::SAE,X)
    return encode(M,X)
end

function (M::SparseEncoder)(x)
    c = encode(M,x)
    x̂ =  decode(M,c)
    return x̂
end

