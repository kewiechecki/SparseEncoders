include("SAE.jl")
include("PSAE.jl")
include("models.jl")
include("auxfns.jl")
include("Rmacros.jl")

using MLDatasets, StatsPlots, OneHotArrays
using Flux: logitcrossentropy

using JLD2,Tables,CSV
using StatsPlots

# where to write the toy model
path = "data/MNIST/"
batchsize=128

m = 3
d = 27
k = 12

loader = mnistloader(batchsize)
x,y = first(loader) |> gpu
labels = string.(unhot(cpu(y)))[1,:]

#load outer model
M_outer = outermodel() |> gpu
state_outer = JLD2.load(path*"state_outer.jld2","state_outer");
Flux.loadmodel!(M_outer,state_outer)

sae = SAE(m,d) |> gpu
state_SAE = JLD2.load(path*"state_SAE.jld2","state_SAE")
Flux.loadmodel!(sae,state_SAE)

sae_linear = SAE(m,d,identity) |> gpu
state_linear = JLD2.load(path*"state_linear.jld2","state_linear")
Flux.loadmodel!(sae_linear,state_linear)

partitioner = Chain(Dense(m => k,relu))

psae = PSAE(sae_linear,partitioner) |> gpu
state_PSAE1k = JLD2.load(path*"state_PSAE1k.jld2","state_PSAE1k")
Flux.loadmodel!(psae,state_PSAE1k)

plotlogits(M_outer,x,y,path*:"figures/logits.pdf")

E_outer = θ(x)
E_SAE = encode(sae,E_outer)
E_PSAE = encode(psae,E_outer)

K_outer = π(E_outer)
K_PSAE = cluster(psae,E_outer)
P = pwak(K_PSAE)
C = (K_PSAE * E_PSAE')'

E = vcat(E_outer,E_SAE,E_PSAE)
K = vcat(K_outer,K_PSAE)

E_split = vcat(rep("outer",m),
               rep("SAE",d),
               rep("PSAE",d))

K_split = vcat(rep("outer",10),
               rep("PSAE",k))

hm_E = CH.Heatmap(E',["white","red"],name="embedding",split=labels,column_split=E_split);
hm_K = CH.Heatmap(K',["white","blue"],"P(label)",split=labels,column_split=K_split);

grD.pdf(path*"figures/KE.pdf")
CH.draw(hm_E + hm_K)
grD.dev_off()

hmK_outer = Kheatmap(M_outer,x,y);
hmK_outer = Kheatmap(M_outer,x,y);
