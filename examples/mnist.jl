# A toy model of superposition to demonstrate feature splitting.
# An MNIST classifier is trained with a 3-neuron bottleneck.
# This forces the model to represent more than 3 features in this layer.
# Our goal is to split the activations of this layer into linearly separable features. 

include("SAE.jl")
include("PSAE.jl")
include("trainingfns.jl")
include("models.jl")

using MLDatasets, StatsPlots, OneHotArrays
using Flux: logitcrossentropy

using JLD2,Tables,CSV

# where to write the toy model
path = "data/MNIST/outer/"

epochs = 100
batchsize=512

η = 0.001
λ = 0.001
opt = Flux.Optimiser(Flux.AdamW(η),Flux.WeightDecay(λ))

dat = MNIST(split=:train)[:]
target = onehotbatch(dat.targets,0:9)

m_x,m_y,n = size(dat.features)
X = reshape(dat.features[:, :, :], m_x, m_y, 1, n)

loader = Flux.DataLoader((X,target),
                         batchsize=batchsize,
                         shuffle=true)


M_outer = outermodel() |> gpu

L_outer = []

train!(M_outer,loader,opt,epochs,logitcrossentropy,L_outer);

#write state
state_outer = Flux.state(M_outer) |> cpu;
jldsave(path*"state_outer.jld2";state_outer)
Tables.table(L_outer) |> CSV.write(path*"L_outer.csv")
