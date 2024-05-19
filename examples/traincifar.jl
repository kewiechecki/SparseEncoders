include("SAE.jl")
include("PSAE.jl")
include("trainingfns.jl")
include("models.jl")
include("auxfns.jl")

using StatsPlots
using Flux: logitcrossentropy

# where to write the toy model
path = "data/CIFAR10/"

epochs = 100
batchsize=512

η = 0.001
λ = 0.001
α = 0.001

opt = Flux.AdamW(η)
opt_wd = Flux.Optimiser(opt,Flux.WeightDecay(λ))

m = 3
d = 27
# max clusters
k = 12

cifar = loader(CIFAR10,batchsize)

θ,ϕ,π,L_π,L_ϕ = trainouter(m,cifar,opt,epochs;
                           path=path*"outer/nowd/")

outer_wd = trainouter(m,cifar,opt_wd,epochs;
                      path=path*"outer/wd/")

sae = trainsae(m,d,θ,π,ϕ,α,cifar,opt,epochs,
               path*"inner/nowd/")
sae_wd = trainsae(m,d,θ,π,ϕ,α,cifar,opt_wd,epochs,
                  path*"inner/wd/")

psae = trainpsae(m,d,k,θ,π,ϕ,α,cifar,opt,epochs,
                 path*"inner/nowd/")
psae_wd = trainpsae(m,d,k,θ,π,ϕ,α,cifar,opt_wd,epochs,
                    path*"inner/wd/")
