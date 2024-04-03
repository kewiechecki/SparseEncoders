include("prelude.jl")

epochs = 100

mnist = mnistloader(batchsize)
path_nowd = path*"outer/nowd/"
path_wd = path*"outer/wd/"
path = date()

outer = loadouter(m, path_nowd)

L = (m,x,y)->begin
    E = outer[:encoder](x)
    l_1 = L1_normcos(m,α,E)
    l_2 = Flux.mse((outer[:decoder] ∘ m)(E),outer[:decoder](E))
    return l_1 + l_2,l_1,l_2
end

macro innermodels(f,m,d,k)
    return esc(quote
        sae = $f(SAE,$m,$d) |> gpu
        psae = $f(PSAE,$m,$d,$k) |> gpu
        sparsedict = $f(DictEnc,$m,$d,$k) |> gpu
        distenc_eucl = $f(DistEnc,$m,$d,relu,inveucl) |> gpu
        distenc_cossim = $f(DistEnc,$m,$d,relu,cossim) |> gpu
        distpart_eucl = $f(DistPart,$m,$d,$k,relu,inveucl) |> gpu
        distpart_cossim = $f(DistPart,$m,$d,$k,relu,cossim) |> gpu
    end)
end

macro traininner(f,path,m,d,k,L,dat,opt,epochs)
    return esc(quote
        sae = f(SAE,$m,$d) |> gpu
        l_sae = train!(sae,$path*"/SAE",$L,$dat,$opt,$epochs)

        psae = f(PSAE,$m,$d,$k) |> gpu
        l_psae = train!(psae,$path*"/PSAE",$L,$dat,$opt,$epochs)

        sparsedict = f(DictEnc,$m,$d,$k) |> gpu
        l_sparsedict = train!(sparsedict,$path*"/sparsedict",$L,$dat,$opt,$epochs)

        distenc_eucl = f(DistEnc,$m,$d,relu,inveucl) |> gpu
        l_distenc_eucl = train!(distenc_eucl,$path*"/distenc/eucl",$L,$dat,$opt,$epochs)

        distenc_cossim = f(DistEnc,$m,$d,relu,cossim) |> gpu
        l_distenc_cossim = train!(distenc_cossim,$path*"/distenc/cossim",$L,$dat,$opt,$epochs)

        distpart_eucl = f(DistPart,$m,$d,$k,relu,inveucl) |> gpu
        l_distpart_eucl = train!(distpart_eucl,$path*"/distpart/eucl",$L,$dat,$opt,$epochs)

        distpart_cossim = f(DistPart,$m,$d,$k,relu,cossim) |> gpu
        l_distpart_cossim = train!(distpart_cossim,$path*"/distpart/cossim",$L,$dat,$opt,$epochs)
    end)
end
               
    

sae = SAE(m,d) |> gpu
train!(sae,date()*"/SAE",L,mnist,opt,epochs)

psae = PSAE(m,d,k) |> gpu
train!(psae,date()*"/PSAE",L,mnist,opt,epochs)

distenc_eucl = DistEnc(m,d,relu,inveucl) |> gpu
train!(distenc_eucl,date()*"/distenc/eucl",L,mnist,opt,epochs)

distpart_eucl = DistPart(m,d,k,relu,inveucl) |> gpu
train!(distpart_eucl,date()*"/distpart/eucl",L,mnist,opt,epochs)

distenc_cos = DistEnc(m,d,relu,cossim) |> gpu
train!(distenc_cos,date()*"/distenc/cossim",L,mnist,opt,epochs)

distpart_cos = DistPart(m,d,k,relu,cossim) |> gpu
train!(distpart_cos,date()*"/distpart/cossim",L,mnist,opt,epochs)

sparsedict = DictEnc(m,d,k) |> gpu
train!(sparsedict,date()*"/sparsedict",L,mnist,opt,epochs)

path = "2024-03-16/"

batchsize=128

mnist = mnistloader(batchsize)
x,y = first(mnist) |> gpu
labels = string.(unhot(cpu(y)))[1,:];
x_sub = gpu(cpu(x)[:,:,:,1:27])

@loadinner date() m d k
models = [sae,psae,sparsedict,
          distenc_eucl,distenc_cossim,
          distpart_eucl,distpart_cossim]

l = map(m->m[2][:,1],models)
l1 = map(m->m[2][:,2],models)
l2 = map(m->m[2][:,3],models)
label = ["SAE","PSAE","DictEnc","DistEucl","DistCos","PartEucl","PartCos"]

plotloss(l,label,"loss",date(),"loss.pdf",ylims=(0,1))
plotloss(l1,label,"L1",date(),"L1.pdf",ylims=(0,1))
plotloss(l2,label,"MSE",date(),"L2.pdf",ylims=(0,0.1))

save(path*"/input.pdf",imgbatch(x_sub))
save(path*"/decoded.pdf",imgbatch((outer[:decoder] ∘ outer[:encoder])(x_sub)))

map(m->save(m[3]*"/decoded.pdf",
            imgbatch((outer[:decoder] ∘ m[1] ∘ outer[:encoder])(x_sub))),
    models)

E = outer[:encoder](x)
K = outer[:classifier](E)

hm_E = CH.Heatmap(E',"embedding",col=["white","red"],border=true,
                  show_row_dend=false,show_column_dend=false,
                  width=2*m,height=batchsize);
hm_K = CH.Heatmap(K',"P(label)",col=["white","blue"],border=true,
                  show_row_dend=false,show_column_dend=false,
                  width=20,height=batchsize);

grD.pdf(path*"/classifier.pdf",width = ((5 / batchsize) * (2*m+10))+2)
CH.draw(hm_K + hm_E)
grD.dev_off()

map(m->KEheatmap(m[1],E,m[3],"KE.pdf"),models)
