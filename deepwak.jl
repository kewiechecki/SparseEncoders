include("prelude.jl")

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

frac = 10
n_test = Integer(2^round(log2(n) - log2(frac)))
n_train = n - n_test
test,train = sampledat(X,n_test) |> gpu
X = gpu(X)

dat = Flux.DataLoader((train,train),batchsize=n_test,shuffle=true) 
x,y = first(dat) |> gpu

L = (M,x,y)->begin
    E_train = x
    E_test = test

    #l1_train = L1_normcos(E_train)
    #l1_test = L1_normcos(E_test)
    #l1_train = L1(M,α,E_train)
    l1_test = L1(M,α,E_test)
    
    l2_train = Flux.mse(M(E_train),x)
    l2_test = Flux.mse(M(E_test),test)

    #H_train = (mean ∘ hcat)(map(entropy ∘ cluster,M,x)...)
    H_test = (mean ∘ hcat)(map(entropy ∘ cluster,M,test)...)
    #return l1_train + l2_train, l1_train, l1_test, l2_train, l2_test
    return l2_train, l1_train, l1_test, l2_test,H_test
end

enc = ()->mlp(outerlayers,tanh) |> gpu
dec = ()->mlp(reverse(outerlayers),tanh) |> gpu
classifier = ()->mlp4x(m,k,3) |> gpu

g = (F,args...;kwargs...)->EncoderBlock(F,nhead,+,args...;kwargs...)
M = EncoderBlock(Parallel(+,map(_->DistPart(enc(),dec(),classifier(),
                                           cossim),
                     1:nhead)...))
M = EncoderBlock(DistPart,5,+,enc(),dec(),classifier(),cossim)
M = EncoderBlock(PSAE,5,+,enc(),dec(),classifier(),cossim)
l = train!(M,date()*"/DeePWAK/cossim",L,dat,opt_wd,epochs)

distpart_cossim = EncoderBlock(Parallel(+,map(_->DistPart(enc(),dec(),
                                                          classifier(),
                                                          cossim),
                                              1:nhead)...))
distpart_eucl = EncoderBlock(Parallel(+,map(_->DistPart(enc(),dec(),
                                                          classifier(),
                                                          inveucl),
                                              1:nhead)...))
sparsedict = EncoderBlock(Parallel(+,map(_->DictEnc(SparseDict(d,k),
                                                          classifier(),
                                                          dec()),
                                              1:nhead)...))

deepwak = EncoderBlock(Parallel(+,map(_->PSAE(Autoencoder(enc(),dec()),
                                              classifier()),1:nhead)...))
l = train!(deepwak,date()*"/tmp",L,dat,opt_wd,100)
