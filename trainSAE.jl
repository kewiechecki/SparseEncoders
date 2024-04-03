include("SAE.jl")
include("PSAE.jl")
include("trainingfns.jl")
include("models.jl")
include("auxfns.jl")

using StatsPlots
using Flux: logitcrossentropy

# where to write the toy model
path = "data/MNIST/"

epochs = 1000
batchsize=512

η = 0.001
λ = 0.001
opt = Flux.Optimiser(Flux.AdamW(η),Flux.WeightDecay(λ))
α = 0.001

m = 3
d = 27
# max clusters
k = 12

loader = mnistloader(batchsize)
θ,π,ϕ = loadouter(m,path)

sae = trainsae(m,d,θ,π,ϕ,α,mnist,opt,epochs,
               path*"inner/nowd/")
sae_wd = trainsae(m,d,θ,π,ϕ,α,mnist,opt_wd,epochs,
                  path*"inner/wd/")

psae = trainpsae(m,d,k,θ,π,ϕ,α,mnist,opt,epochs,
                 path*"inner/nowd/")
psae_wd = trainpsae(m,d,k,θ,π,ϕ,α,mnist,opt_wd,epochs,
                    path*"inner/wd/")

sae = SAE(m,d) |> gpu
L_classifier = train!(sae,Chain(θ,π),α,loader,opt,epochs,
                      logitcrossentropy;
                      path=path*"inner/classifier/SAE/L1_L2")

sae_L2 = SAE(m,d) |> gpu
L2_classifier = train!(sae,loader,opt,epochs,
                       logitcrossentropy;
                       prefn=θ,postfn=π,
                       path=path*"inner/classifier/SAE/L2")

sae_enc = SAE(m,d) |> gpu
L_encoder = train!(sae_enc,Chain(θ,ϕ),
                   α,loader,opt,epochs,
                   Flux.mse;
                   path=path*"inner/encoder/SAE/L1_L2")

sae_enc_L2 = SAE(m,d) |> gpu
L2_encoder = train!(sae_enc_L2,
                   loader,opt,epochs,
                   Flux.mse;
                   prefn=θ,postfn=ϕ,
                   ignoreY=true,
                   path=path*"inner/encoder/SAE/L2")

partitioner = Chain(Dense(m => k,relu))
psae = PSAE(sae,partitioner) |> gpu
L_PSAE = []
train!(psae,M_outer,α,loader,opt,epochs,Flux.mse,L_PSAE,"inner/PSAE/")

state_SAE = Flux.state(sae) |> cpu;
jldsave(path*"state_SAE.jld2";state_SAE)
Tables.table(L_SAE) |> CSV.write(path*"L_SAE.csv")

sae_linear = SAE(m,d,identity) |> gpu
L_linear = []
train!(sae_linear,M_outer,α,loader,opt,epochs,logitcrossentropy,L_linear)

state_linear = Flux.state(sae_linear) |> cpu;
jldsave(path*"state_linear.jld2";state_linear)
Tables.table(L_linear) |> CSV.write(path*"L_linear.csv")

p = scatter(1:length(L_SAE), L_SAE,
            xlabel="batch",ylabel="loss",
            legend=:none)
savefig(p,"data/MNIST/loss_SAE.pdf")


psae_linear = PSAE(sae_linear,partitioner) |> gpu
L_PSAElinear = []
train!(psae_linear,M_outer,α,loader,opt,epochs,logitcrossentropy,L_PSAElinear)
p = scatter(1:length(L_SAE), L_SAE,
            xlabel="batch",ylabel="loss",
            label="SAE")
scatter!(1:length(L_PSAE), L_PSAE,label="PSAE")
savefig(p,"data/MNIST/loss_relu.svg")

p = scatter(1:length(L_linear), L_linear,
            xlabel="batch",ylabel="loss",
            label="SAE")
scatter!(1:length(L_PSAElinear), L_PSAElinear,label="PSAE")
savefig(p,"data/MNIST/loss_linear.svg")

train!(psae_linear,M_outer,α,loader,opt,1000,logitcrossentropy,L_PSAElinear)

state_PSAE1k = Flux.state(psae_linear) |> cpu;
jldsave(path*"state_PSAE1k.jld2";state_PSAE1k)
Tables.table(L_PSAElinear) |> CSV.write(path*"L_PSAElinear1k.csv")
p = scatter(1:length(L_linear), L_linear,
            xlabel="batch",ylabel="loss",
            label="SAE")
scatter!(1:length(L_PSAElinear), L_PSAElinear,label="PSAE")
savefig(p,"data/MNIST/loss_linear1k.svg")

include("Rmacros.jl")
