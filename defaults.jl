# where to write the toy model
path = "data/MNIST/"

epochs = 1000
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
