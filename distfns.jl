using Flux, CUDA, LinearAlgebra

function clusts(C::AbstractMatrix)
    return map(x->x[1],argmax(C,dims=1))
end

function neighborcutoff(G::AbstractArray; ϵ=0.0001)
    M = G .> ϵ
    return G .* M
end

# [CuArray] -> [CuArray]
# version of Euclidean distance compatible with Flux's automatic differentiation
# calculates pairwise distance matrix by column (default) or row
#function euclidean(x::CuArray{Float32};dims=1)
function euclidean(x::AbstractArray{<:AbstractFloat};dims=1)
    x2 = sum(x .^ 2, dims=dims)
    D = x2' .+ x2 .- 2 * x' * x
    # Numerical stability: possible small negative numbers due to precision errors
    D = sqrt.(max.(D, 0) .+ eps(Float32))  # Ensure no negative values due to numerical errors
    return D
end

# [CuArray] -> [CuArray]
# returns reciprocal Euclidean distance matrix
function inveucl(x::AbstractArray;dims=1)
    return 1 ./ (euclidean(x) .+ eps(Float32))
end

# [CuArray] -> [CuArray]
# function to calculate cosine similarity matrix
function cossim(x::AbstractArray{<:AbstractFloat};dims=1)
    # Normalize each column (or row, depending on 'dims') to unit length
    norms = sqrt.(sum(x .^ 2, dims=dims) .+ eps(Float32))
    x_normalized = x ./ norms

    # Compute the cosine similarity matrix
    # For cosine similarity, the matrix multiplication of normalized vectors gives the cosine of angles between vectors
    C = x_normalized' * x_normalized

    # Ensure the diagonal elements are 1 (numerical stability)
    #i = CartesianIndex.(1:size(C, 1), 1:size(C, 1))
    #C[i] .= 1.0

    return C
end

function sindiff(x::AbstractArray{<:AbstractFloat};dims=1)
    C = cossim(x;dims=dims)
    return sqrt.(max.(1 .- C .^ 2,0))
end

function maskI(n::Integer)
    return 1 .- I(n) |> gpu
end

function invsum(x::AbstractArray,dims::Integer=1)
    return 1 ./ sum(x .+ eps(),dims=dims)
end

function zerodiag(G::AbstractArray)
    m, n = size(G)
    G = G .* (1 .- I(n))
    return G
end

# [CuArray] -> [CuArray]
# workaround for moving identity matrix to GPU 
function zerodiag(G::CuArray)
    m, n = size(G)
    G = G .* (1 .- I(n) |> gpu)
    return G
end

# constructs weighted affinity kernel from adjacency matrix
# sets diagonal to 0
# normalizes rows/columns to sum to 1 (default := columns)
function wak(G::AbstractArray; dims=1)
    G = zerodiag(G)
    G = G ./ (sum(G,dims=dims) .+ eps(eltype(G)))
    return G
end

function pwak(K::AbstractMatrix; dims=1)
    P = K' * K
    return wak(P)
end
