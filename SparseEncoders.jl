module SparseEncoders
export SparseEncoder,Autoencoder,SAE,
    encode,decode,diffuse
include("SAE.jl")
end

module CuDistances
export euclidean,inveucl,cossim,sindiff,maskI,invsum,zerodiag,wak,pwak
include("distfns.jl")
end

module SparsityLoss
export entropy,L1,L1_scaleinv,L1_cos,L1_normcos,L2,loss,loss_SAE
using Main.CuDistances, Main.SparseEncoders
include("lossfns.jl")
end

module EncoderBlocks
export EncoderBlock,
    heads,size,conn,map,mapreduce,
    encode,diffuse,L1,L1_normcos
using Main.SparseEncoders
include("EncoderBlock.jl")
end

module SparseClassifiers
export SparseEncoder,Autoencoder,SAE,
    encode,decode,diffuse
export SparseClassifier,PSAE,SparseDict,
    cluster,centroid,partition,encodepred

using Main.SparseEncoders
include("PSAE.jl")
end

module DistEncoders
export euclidean,inveucl,cossim,sindiff,invsum,zerodiag,wak,pwak
export SparseEncoder,Autoencoder,SAE,
    encode,decode,diffuse
export DistEnc,kern,dist

using Flux,CUDA
using Main.CuDistances, Main.SparseEncoders
import Main.SparseEncoders: encode,decode

include("DistEnc.jl")
end

module DistClassifiers
export euclidean,inveucl,cossim,sindiff,invsum,wak,pwak
export SparseEncoder,Autoencoder,SAE,
    encode,decode,diffuse
export SparseClassifier,PSAE,SparseDict,
    cluster,centroid,partition,encodepred
export DistEnc,kern

using Main.SparseClassifiers,Main.DistEncoders
using Flux, Functors
include("DistPart.jl")
end

module DictEncoders
export SparseDict, DictEnc,
    cluster, dict, encode, diffuse, centroid, decode
using Main.SparseClassifiers
include("SparseDict.jl")
end

module Ablations
export zeroabl,meanabl

function maskrow(E::AbstractArray,i::Integer,invert=false)
    m,n = size(E)
    mask = sparse(rep(i,n),1:n,1,m,n)
    if(invert)
        mask = 1 .- mask
    end
    return mask |> gpu
end

function zeroabl(E::AbstractArray,i::Integer,invert=false)
    return E .* maskrow(E,i,invert)
end

function meanabl(E::AbstractArray,i::Integer,invert=false)
    μ = mean(E,dims=2)
    mask = maskrow(E,i,invert)
    x = E .* mask
    y = μ .* (1 .- mask)
    return x .+ y
end
end
