include("trainingfns.jl")
include("models.jl")
include("auxfns.jl")

using StatsPlots
using Flux: logitcrossentropy

using JLD2,Tables,CSV

using ImageInTerminal,Images

# where to write the toy model
path = "data/MNIST/outer/"

epochs = 100
batchsize = 512
m = 3

η = 0.001
λ = 0.000
opt = Flux.Optimiser(Flux.AdamW(η),Flux.WeightDecay(λ))

loader = mnistloader(batchsize)

θ = outerenc()
ϕ = outerdec()

M_outerenc = Chain(θ,ϕ) |> gpu

L_outerenc = []

train!(M_outerenc,loader,opt,epochs,Flux.mse,L_outerenc,
       ignoreY=true,savecheckpts=true,path=path*"encoder/");

x,y = first(loader) |> gpu
colorview(Gray,x[:,:,1,1:5])
colorview(Gray,M_outerenc(x[:,:,1,1:5]))

train!(M_outerenc,loader,opt,epochs,logitcrossentropy,L_outerenc,
       ignoreY=true,savecheckpts=true,path=path*"encoder/");

colorview(Gray,M_outerenc(x[:,:,1,1:5]))

train!(M_outerenc,loader,opt,epochs,Flux.mse,L_outerenc,
       ignoreY=true,savecheckpts=true,path=path*"encoder/");

colorview(Gray,M_outerenc(x[:,:,1,1:5]))

p = scatter(1:length(L_outerclas), L_outerclas,
            xlabel="batch",ylabel="loss",
            label="outer");
savefig(p,path*"classifier/loss.pdf")
