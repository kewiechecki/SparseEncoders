Mechinterp data structures for Julia.
In addition to a basic
[sparse autoencoder](https://transformer-circuits.pub/2023/monosemantic-features/index.html#setup-autoencoder-motivation)
(SAE), I introduce the sparse partitioned sparse autoencoder (PSAE).

# Usage
This example trains an SAE on a toy model of superposition (the "outer model").
The outer model consists of an 
```{bash}
julia mnist.jl
```

# Defining the SAE

```{julia}
include("SAE.jl")

# SAE hyperparameters
# input dimension
m = 3
# number of features
d = 27

sae = SAE(m,d,relu)
```
# Training the SAE
Training hyperparameters
```{julia}
epochs = 100
batchsize=512

# learning rate
η = 0.001

# weight decay rate
λ = 0.001
opt = Flux.Optimiser(Flux.AdamW(η),Flux.WeightDecay(λ))

# sparcity coefficient
α = 0.001
```
Import outer model
```{julia}
include("trainingfns.jl"

M_outer = outermodel()
state_outer = JLD2.load("data/MNIST/state_outer.jld2","state_outer");
Flux.loadmodel!(M_outer,state_outer)
```

Flux makes it trivial to write a loss function incorporating the output of an outer model.
```{julia}
function loss_SAE(M_outer,α,lossfn,x)
    x = gpu(x)
  p  yhat = M_outer(x)
    f = M->L1(M,α,x) + L2(M,lossfn,x,yhat)
    return m->lossfn((M_outer[2] ∘ m ∘ M_outer[1])(x),yhat)
end
```

Train SAE on bottleneck activations of outer model
```{julia}

L_SAE = []
@showprogress map(1:epochs) do _
    map(loader) do (x,y)
        f = loss_SAE(M_outer,α,logitcrossentropy,x)
        state = Flux.setup(opt,sae)
        l,∇ = Flux.withgradient(f,sae)
        Flux.update!(state,sae,∇[1])
        push!(L_SAE,l)
    end
end

#
```

I include the polymorphic function `train!` which does this automatically.
```{julia}
#initialize training log
L_SAE = []
train!(sae,α,loader,opt,epochs,logitcrossentropy,L_SAE)
```

Save model
```{julia}
state_SAE = Flux.state(sae) |> cpu;
jldsave(path*"state_SAE.jld2";state_SAE)
Tables.table(L_SAE) |> CSV.write(path*"L_SAE.csv")
```

# PSAE

![flowchart](https://github.com/kewiechecki/SAE/blob/master/fig/flowchart.jpg?raw=true)

A PSAE attempts to simultaneously learn a sparse embedding and a sparse clustering of the data.
It is based on treating an optimal clustering as a
[maximum entropy partition of the data](https://www.mdpi.com/1099-4300/17/1/151).
It optimizes clusters based on the [noise2self](https://arxiv.org/abs/1901.11365) principle.
Essentially, it tries to find the subset of the minibatch that best predicts each sample in the minibatch.
In its current form it will probably not perform well when the number of latent clusters is much larger than the batch size.

# Weighted Affinity Kernels
The objective function tries to reduce loss when model activations are replaced by estimates produced by a
[weighted affinity kernel](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7817019/) (WAK).
The WAK represents a Markov process where a sample transitions to one of its neighbors with probability inversely proportional to distance.
The dot product of the WAK with the embedding matrix for a minibatch replaces the embeddings for each minibatch with an estimate based on its neighbors.
The basic principle is finding the best graph representation of the data.
For any graph, an adjacency matrix can be converted to a transition matrix by masking the diagonal and scaling columns to sum to 1.
```{julia}
# constructs weighted affinity kernel from adjacency matrix
# sets diagonal to 0
# normalizes rows/columns to sum to 1 (default := columns)
function wak(G::AbstractArray; dims=1)
    G = zerodiag(G)
    G = G ./ (sum(G,dims=dims) .+ eps(eltype(G)))
    return G
end
```

# Partitioned WAK
A partitoned WAK simply applies a mask to any edges between nodes in separate partitions.
For a partition matrix `P` and adjacency matrix `G`, this is given by elementwise multiplication.
If we treat clusters as maximally entropic partitions, the nodes within them should be maximally connected.
if nodes within clusters are completely connected and nodes between clusters are completely disconnected, 
we can treat the partition matrix as the adjacency matrix.
If we express clusters as 1-hot encodings, The partition matrix is given by multiplying the incidence matrix with its transpose.
Applying the `wak` transformation to this matrix gives the partitioned affinity kernel.
We call this the `pwak` transformation.
```{julia}
function pwak(K::AbstractMatrix; dims=1)
    P = K' * K
    return wak(P)
end
```
# Defining PSAE
```{julia}
include(PSAE.jl)
# max clusters
k = 12
partitioner = Dense(m => k,relu)
psae = PSAE(sae,partitioner)

L_PSAE = []

train!(psae,M_outer,α,loader,opt,epochs,logitcrossentropy,L_PSAE)
```