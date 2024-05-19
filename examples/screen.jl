include("prelude.jl")

#hyperparams
nhead = 5
α = 0.00001
d = 56
k = 27

# data
groups = readcsv("data/screen/groups.csv")

dat = readcsv("data/screen/z_dat.csv")
dat = Matrix(dat[:,2:end]);
dat = hcat(filter(x->sum(x) != 0,eachslice(dat,dims=2))...);

X = scaledat(dat')
m,n = size(X)
outerlayers = accumulate(÷,rep(2,4),init=2*m)
d = last(outerlayers)
k = d

frac = 10
n_test = Integer(2^round(log2(n) - log2(frac)))
n_train = n - n_test
test,train = sampledat(X,n_test) |> gpu
X = gpu(X)

dat = Flux.DataLoader((train,train),batchsize=n_test,shuffle=true) 
x,y = first(dat) |> gpu

#loss fns
L_outer = (M,x,y)->begin
    l2_train = Flux.mse(M(x),x)
    l2_test = Flux.mse(M(test),test)
    return l2_train, l2_test
end

L = (M,x,y)->begin
    E_train = encoder(x)
    E_test = encoder(test)

    #l1_train = L1_normcos(E_train)
    #l1_test = L1_normcos(E_test)
    l1_train = L1(M,α,E_train)
    l1_test = L1(M,α,E_test)
    
    l2_train = Flux.mse((decoder ∘ M)(E_train),x)
    l2_test = Flux.mse((decoder ∘ M)(E_test),test)
    #return l1_train + l2_train, l1_train, l1_test, l2_train, l2_test
    return l2_train, l1_train, l1_test, l2_train, l2_test
end

L_block = (M,x,y)->begin
    E = encoder(x)
    E_test = encoder(test)
    l0 = mapreduce(L1,M,α,E)
    l1 = L1(M,α,E)
    l2 = Flux.mse((decoder ∘ M)(E),x)

    l0_test = mapreduce(L1,M,α,E_test)
    l1_test = L1(M,α,E_test)
    l2_test = Flux.mse((decoder ∘ M)(E_test),test)
    #return l0 + l1 + l2,l0,l0_test,l1,l1_test,l2,l2_test
    return l2,l0,l0_test,l1,l1_test,l2,l2_test
end

L_endtoend = (M,x,y)->begin
    E_train = M[1](x)
    E_test = M[1](test)

    #l1_train = L1_normcos(E_train)
    #l1_test = L1_normcos(E_test)
    l1_train = L1(M[2],α,E_train)
    l1_test = L1(M[2],α,E_test)
    
    l2_train = Flux.mse(M[2:3](E_train),x)
    l2_test = Flux.mse(M[2:3](E_test),test)
    return l1_train + l2_train, l1_train, l1_test, l2_train, l2_test
end

L_block_endtoend = (M,x,y)->begin
    E = M[1](x)
    E_test = M[1](test)
    l0 = mapreduce(L1,M[2],α,E)
    l1 = L1(M[2],α,E)
    l2 = Flux.mse(M[2:3](E),x)

    l0_test = mapreduce(L1,M[2],α,E_test)
    l1_test = L1(M[2],α,E_test)
    l2_test = Flux.mse(M[2:3](E_test),test)
    return l0 + l1 + l2,l0,l0_test,l1,l1_test,l2,l2_test
end

# model constructors
classifier = ()->mlp4x(d,k,3)
enc_inner = ()->Chain(Dense(d => ,relu))
dec_inner = ()->Chain(Dense(4*d => d,relu))

g = (F,args...;kwargs...)->EncoderBlock(F,nhead,+,args...;kwargs...)

enc = ()->Dense(m => d,tanh) |> gpu
dec = ()->Chain(x->tanh.(x),Dense(d => m,tanh)) |> gpu

f_endtoend = (args...)->begin
    κ = fmap(args...) |> gpu
    θ = enc()
    ϕ = dec()
    return Chain(θ,κ,ϕ)
end

g_endtoend = (F,args...;kwargs...)->begin
    κ = EncoderBlock(F,nhead,+,args...;kwargs...) |> gpu
    θ = enc()
    ϕ = dec()
    return Chain(θ,κ,ϕ)
end

f_train = (M,L)->train!(M[1],M[2],L,dat,opt,epochs)
# outer model
encoder = Dense(m => d,tanh) |> gpu
decoder = Dense(d => m,tanh) |> gpu
outer = Chain(encoder,decoder)
train!(outer,date()*"/screen/outer",L_outer,dat,opt,epochs)

outer = loadmodel(outer,date()*"/screen/outer")
encoder = outer[1][1]
decoder = outer[1][2]
decoder = Chain(x->tanh.(x),decoder)

#inner models
d = 14
k = 14

inner = innermodels(fmap,d,d*4,k,date()*"/screen/1layer",relu)   
map(M->train!(M[1],M[2],L,dat,opt,epochs),inner)
inner = loadmodels(inner)

block = innermodels(g,d,d*4,k,date()*"/screen/1layer/block",relu)   
map(M->train!(M[1],M[2],L_block,dat,opt,epochs),block)
block = loadmodels(block)

endtoend = innermodels(f,d,d*4,k,date()*"/screen/1layer/endtoend",relu)   
map(M->train!(M[1],M[2],L_endtoend,dat,opt,epochs),endtoend)
endtoend = loadmodels(endtoend)

e2eblock = innermodels(g_endtoend,d,d*4,k,date()*"/screen/1layer/endtoend/block",relu)   
map(M->train!(M[1],M[2],L_block,dat,opt,epochs),e2eblock)
e2eblock = loadmodels(e2eblock)

inner = innermodels((F,args...)->F(args...),
                    enc_inner,dec_inner,classifier,
                    d,4*d,k,date()*"/screen/L2")
map(M->f_train(M,L),inner)
inner = loadmodels(inner)
# loss plots
key = [:sae,:psae,:sparsedict,
       :distenc_eucl,:distenc_cossim,
       :distpart_eucl,:distpart_cossim]
l = map(i->models[i][2][:,1],key)
l1 = map(i->models[i][2][:,2],key)
l2 = map(i->models[i][2][:,3],key)
label = ["SAE","PSAE","DictEnc","DistEucl","DistCos","PartEucl","PartCos"]

label = Dict([(:sae,"SAE"),(:psae,"PSAE"),(:sparsedict,"DictEnc"),
              (:distenc_eucl,"DistEucl"),(:distenc_cossim, "DistCos"),
              (:distpart_eucl,"PartEucl"),(:distpart_cossim,"PartCos")])

plotloss(l,label,"loss",date(),"screen/loss.pdf",ylims=(0,5))
plotloss(l1,label,"L1",date(),"screen/L1.pdf",ylims=(0,5))
plotloss(l2,label,"MSE",date(),"screen/L2.pdf",ylims=(0,0.03))

maplab(inner) do key,κ
    θ = encoder()
    ϕ = decoder()
    M = Chain(θ,κ,ϕ)
    train!(M,date()*"/"*string(key),L,dat,opt,epochs)
end

outer = Chain(Dense(m => d, tanh),SAE(d, d*4,relu),x->tanh.(x),Dense(d => m,tanh)) |> gpu
l = train!(outer,date()*"/screen/sae/",L_outer,dat,opt,epochs)

l = map(1:m) do i
    M = Chain(Dense(m => i,tanh),Dense(i =>m,tanh))|>gpu
    l = train!(M,"",(M,x,y)->Flux.mse(M(x),x),dat,opt,epochs;savecheckpts=false)
end
l = hcat(l...)
writecsv(hcat(l...),date(),"loss_byparams.csv")
bic = map((l,i)->(2*i*m+2*i)*log(n) - 2 * log(l),l[7000,:],1:m)

outer_wd = Chain(θ,ϕ) |> gpu
train!(outer_wd,date()*"/screen/outer_wd/",L_outer,dat,opt_wd,epochs)

E_test = θ(test)
E_train = θ(train)
dat = Flux.DataLoader((E_train,train),batchsize=n_test,shuffle=true) 

L_inner = (M,x,y)->begin
    #l_0 = mapreduce(L1_normcos,M,α,E)
    l_1 = L1_normcos(M,α,x)
    l_2 = Flux.mse((ϕ ∘ M)(x),y)
    
    return l_1 + l_2,l_1,l_2
end

L = (M,x,y)->begin
    l,l_1,l_2 = L_inner(M,x,y)
    l_test,l1_test,l2_test = L_inner(M,E_test,test)
    return l,l_1,l_2,l1_test,l2_test
end

encoder = ()->mlp4x(d,d,1,tanh)
classifier = ()->mlp4x(d,k,1)

g = M->train!(M[1],M[2],L,dat,opt_wd,epochs)

models = Dict([(:distenc_eucl,
                (DistEnc(encoder(),encoder(),inveucl) |> gpu,
                 date()*"/screen/inner/distenc/eucl")),
               (:distenc_cossim,
                (DistEnc(encoder(),encoder(),cossim) |> gpu,
                 date()*"/screen/inner/distenc/cossim")),
               (:distpart_eucl,
                (DistPart(encoder(),encoder(),classifier(),inveucl) |> gpu,
                 date()*"/screen/inner/distpart/eucl")),
               (:distpart_cossim,
                (DistPart(encoder(),encoder(),classifier(),cossim) |> gpu,
                 date()*"/screen/inner/distpart/cossim")),
               (:sparsedict,
                (DictEnc(classifier(),encoder(),m,k) |> gpu,
                 date()*"/screen/inner/sparsedict"))])

map(g,models)

L_block = (M,x,y)->begin
    l_0 = mapreduce(L1_normcos,M,α,x)
    l_1 = L1_normcos(M,α,x)
    l_2 = Flux.mse((ϕ ∘ M)(x),y)
    return l_0 + l_1 + l_2,l_0,l_1,l_2
end

L = (M,x,y)->begin
    l,l_0,l_1,l_2 = L_block(M,x,y)
    l_test,l0_test,l1_test,l2_test = L_block(M,E_test,test)
    return l,l_0,l_1,l_2,l0_test,l1_test,l2_test
end

f = (F,args...;kwargs...)->EncoderBlock(F,h,+,args...;kwargs...)
block = innermodels(f,d,m,k,date()*"/screen/block/")
map(g,block)
