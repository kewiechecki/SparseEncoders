include("prelude.jl")

mnist = mnistloader(batchsize)

epochs = 100
b = 5
m = 12
β = 0.001

L = (M,x,y)->begin
    E = M[1](x)
    
    K = map(cluster,M[2],E)
    F = map((m,x)->m.dict(x),heads(M[2]),K)
    Ê = map(decode,heads(M[2]),F)

    x̂ = M[3](sum(Ê))

    #H = sum(map(sum ∘ entropy,K)) * β
    
    l_0 = sum(map(L1_normcos,F)) * α
    l_1 = sum(map(L1_normcos,Ê)) * α
    l_2 = Flux.mse(x̂,x)
    return l_0 + l_1 + l_2,l_0,l_1,l_2
end

θ = Chain(mnistconv(),Dense(12 => 10,relu),softmax) |> gpu
ϕ = Chain(Dense(3 => 12),mnistdeconv()) |> gpu
ψ = Dense(10 => 3,relu) |> gpu

κ = EncoderBlock(DictEnc,b,+,m,d,k) |> gpu

ω = Chain(θ,κ,ϕ)

L = (M,x,y)->begin
    ŷ_1 = M[1](x)
    x̂ = M[2](ŷ_1)
    ŷ_2 = M[1](x̂)
    
    CE1 = Flux.crossentropy(ŷ_1,y)
    CE2 = Flux.crossentropy(ŷ_2,y)
    MSE = Flux.mse(x̂,x)
    return CE1 + MSE,CE1,CE2,MSE
end
    
encoder = Chain(mnistconv(),Dense(12 => 10,relu),softmax) |> gpu
decoder = Chain(Dense(10 => 3,relu),Dense(3 => 12,relu),mnistdeconv()) |> gpu
outer = Chain(encoder,decoder)

save(date()*"/mnist/outer/input.pdf",imgbatch(x))
L_outer = train!(outer,date()*"/mnist/outer",L,mnist,opt,epochs)
save(date()*"/mnist/outer/decoded.pdf",imgbatch(outer(x)))

block = EncoderBlock(DictEnc,b,+,10,d,k) |> gpu
ω = Chain(encoder,Chain(block,decoder))
train!(ω,date()*"/mnist/dict",L,mnist,opt,epochs)
save(date()*"/mnist/dict/decoded.pdf",imgbatch(ω(x)))
