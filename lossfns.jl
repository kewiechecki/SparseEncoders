
#[Float] -> Float
#Shannon entroy
function entropy(W::AbstractArray)
    W = abs.(W) .+ eps(eltype(W))
    W = W ./ sum(W)
    return -sum(W .* log2.(W))
end

# [[Float]] -> [Float]
#row/coumn Shannon entropy (default := column)
function entropy(W::AbstractMatrix;dims=1)
    W = abs.(W) .+ eps(eltype(W))
    W = W ./ sum(W,dims=dims)
    return -sum(W .* log2.(W),dims=dims)
end

function L1(x::AbstractArray)
    return sum(abs.(x))
end
    
function L1_scaleinv(x::AbstractArray)
    x = abs.(x)
    x = invsum(x) * x' * invsum(x,2)
    return only(x)
end

function L1_cos(x::AbstractArray)
    x = (abs.(cossim(x')) * (invsum(x) * x')')' * invsum(x,2)
    return sum(x)
end

function L1_normcos(x::AbstractArray)
    x = (abs.(cossim(x')) * x * abs.(cossim(x)) * invsum(x)')' * invsum(x,2)
    return sum(x)
end

function L1(M::SparseEncoder,α,x)
    c = diffuse(M,x)
    return α * sum(abs.(c))
end

function L1_scaleinv(M::SparseEncoder,α,x)
    c = diffuse(M,x)
    return α * L1_scaleinv(c)
end

function L1_normcos(M::SparseEncoder,α,x)
    c = diffuse(M,x)
    return α * L1_normcos(c)
end

function L2(M::SparseEncoder,lossfn,x,y)
    return lossfn(M(x),y)
end

function loss(M::SparseEncoder,α,lossfn,x,y)
    x = gpu(x)
    y = gpu(y)
    return L1(M,α,x) + L2(M,lossfn,x,y)
end

# if α isn't specified just calculate L2
function loss(M::SparseEncoder,lossfn,x,y)
    x = gpu(x)
    y = gpu(y)
    return L2(M,lossfn,x,y)
end

function loss_SAE(α,lossfn,x,y)
    return M->loss(M,α,lossfn,x,y)
end

function loss_SAE(M_outer,α,lossfn,x)
    x = gpu(x)
    yhat = M_outer(x)
    f = M->L1(M,α,x) + L2(M,lossfn,x,yhat)
    return m->lossfn((M_outer[2] ∘ m ∘ M_outer[1])(x),yhat)
end

function loss_SAE(lossfn,x,y)
    return M->loss(M,lossfn,x,y)
end

function loss_SAE(M_outer,lossfn,x)
    x = gpu(x)
    yhat = M_outer(x)
    f = M->L1(M,x) + L2(M,lossfn,x,yhat)
    return m->lossfn((M_outer[2] ∘ m ∘ M_outer[1])(x),yhat)
end
