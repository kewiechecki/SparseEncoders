include("prelude.jl")

epochs = 1000

opt = Flux.Optimiser(Flux.AdamW(η),Flux.WeightDecay(λ))
loader = mnistloader(batchsize)

outer = loadouter(m, path*"outer/nowd")
sae_wd = loadsaes(m,d,path*"inner/wd")

π = Dense(3 => 12,relu)

psae = PSAE(m,d,k) |> gpu
L_classifier = train!(psae,
                      Chain(θ,π),
                      α,loader,opt,epochs,
                      logitcrossentropy;
                      path=path*"inner/classifier/PSAE/L1_L2")

psae_L2 = PSAE(m,d,k) |> gpu
L2_classifier = train!(psae,
                       loader,opt,epochs,
                       logitcrossentropy;
                       prefn=θ,postfn=π,
                       path=path*"inner/classifier/PSAE/L2")

psae_enc = PSAE(m,d,k) |> gpu
L_encoder = train!(psae_enc,
                   Chain(θ,ϕ),
                   α,loader,opt,epochs,
                   Flux.mse;
                   path=path*"inner/encoder/PSAE/L1_L2")

psae_enc_L2 = PSAE(m,d,k) |> gpu
L2_encoder = train!(psae_enc_L2,
                    loader,opt,epochs,
                    Flux.mse;
                    prefn=θ,postfn=ϕ,
                    ignoreY=true,
                    path=path*"inner/encoder/PSAE/L2")

sae_linear = SAE(m,d,identity) |> gpu
state_linear = JLD2.load(path*"state_linear.jld2","state_linear")
Flux.loadmodel!(sae_linear,state_linear)

partitioner = Chain(Dense(m => k,relu))
psae = PSAE(sae_linear,partitioner) |> gpu

L = []
train!(psae,outer,α,loader,opt,epochs,logitcrossentropy,L,
       path=path*"inner")

state_PSAE = Flux.state(psae) |> cpu;
jldsave(path*"state_PSAE.jld2";state_PSAE)
Tables.table(L) |> CSV.write(path*"L_PSAE.csv")
