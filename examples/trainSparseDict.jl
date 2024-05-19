include("SparseDict.jl")
include("trainingfns.jl")
include("models.jl")
include("auxfns.jl")

using MLDatasets, StatsPlots, OneHotArrays
using JLD2,Tables,CSV
using StatsPlots
using ImageInTerminal,Images

# where to write the toy model
path = "data/MNIST/"

epochs = 100
batchsize=128

η = 0.001
λ = 0.000
opt = Flux.Optimiser(Flux.AdamW(η),Flux.WeightDecay(λ))

m = 3
d = 27
# max clusters
k = 12
h = 5

loader = mnistloader(batchsize)
θ,π,ϕ = loadouter(m,path)
outer = Chain(θ,ϕ)

classifier = Chain(Dense(m => k,relu),softmax)
sparsedict = SparseDict(d,k)
decoder = Dense(d => m)

inner = Chain(classifier,sparsedict,decoder) |> gpu
L = []

train!(inner,loader,opt,epochs,Flux.mse,L;
       prefn=θ,postfn=ϕ, ignoreY=true,
       path=path*"inner/sparsedict/")

x,y = first(loader) |> gpu
labels = string.(unhot(cpu(y)))[1,:]

E = θ(x)
x̂ = ϕ(E)
K_outer = π(E)

Ê = inner(E)
E_inner = inner[1:2](E)
x_inner = ϕ(Ê)
K_inner = inner[1](E)

C = inner[2].dict
E_C = inner[3](C)
x_C = ϕ(E_C)

E_hm = vcat(E,E_inner)
K_hm = vcat(K_outer,K_inner)
E_split = vcat(rep("outer",m),
               rep("inner",d))
K_split = vcat(rep("outer",10),
               rep("inner",k))

include("Rmacros.jl")

hm_E = CH.Heatmap(E_inner',name="embedding",split=labels);
hm_K = CH.Heatmap(K_hm',["white","black"],"P(label)",split=labels,column_split=K_split);
hm_lab = CH.Heatmap(K_outer',["white","black"],"P(label)",split=labels);

grD.pdf(path*"inner/dict/KE.pdf")
CH.draw(hm_E + hm_K)
grD.dev_off()

colorview(Gray,x[:,:,1,1]')
colorview(Gray,x̂[:,:,1,1]')
colorview(Gray,x_inner[:,:,1,1]')
colorview(Gray,x_C[:,:,1,1]')

block = Parallel(+,map(_->Chain(Dense(m => 2,relu),
                                softmax,SparseDict(m,2)),
                       1:h)...) |> gpu

train!(block,loader,opt,epochs,Flux.mse,L;
       prefn=θ,postfn=ϕ, ignoreY=true,
       path=path*"inner/block/")
x_block = (ϕ ∘ block)(E)
colorview(Gray,x_block[:,:,1,1]')

K_block = map(l->l[1:2](E),block.layers)
E_block = map(l->l(E),block.layers)

C_block = map(l->l[3].dict,block.layers)
x_C_block = map(ϕ,C_block)
colorview(Gray,x_C_block[5][:,:,1,1]')

E_blocksp = mapreduce(i->rep("Head"*string(i),m),vcat,1:h)
K_blocksp = mapreduce(i->rep("Head"*string(i),2),vcat,1:h)


hm_E = CH.Heatmap(vcat(E_block...)',name="embedding",split=labels,column_split=E_blocksp);
hm_K = CH.Heatmap(vcat(K_block...)',["white","blue"],"P(clust)",split=labels,column_split=K_blocksp);

grD.pdf(path*"inner/block/KE.pdf")
CH.draw(hm_E + hm_K + hm_lab)
grD.dev_off()
