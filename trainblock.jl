include("prelude.jl")

epochs = 100
h = 5

mnist = mnistloader(batchsize)
path_nowd = path*"outer/nowd/"
path_wd = path*"outer/wd/"
path = date()

outer = loadouter(m, path_nowd)

L = (M,x,y)->begin
    E = outer[:encoder](x)
    l_0 = mapreduce(L1_normcos,M,α,E)
    l_1 = L1_normcos(M,α,E)
    l_2 = Flux.mse((outer[:decoder] ∘ M)(E),outer[:decoder](E))
    return l_0 + l_1 + l_2,l_0,l_1,l_2
end

f = (F,args...;kwargs...)->EncoderBlock(F,h,+,args...;kwargs...)
g = M->train!(M[1],M[2],L,mnist,opt,epochs)
models = innermodels(f,m,d,k,date()*"/block/L0")
map(g,models)

sae = EncoderBlock(SAE,h,+,m,d) |> gpu
l_sae = train!(sae,date()*"/block/SAE",L,mnist,opt,epochs)

psae = EncoderBlock(PSAE,h,+,m,d,k) |> gpu
l_psae = train!(psae,date()*"/block/PSAE",L,mnist,opt,epochs)

sparsedict = EncoderBlock(DictEnc,h,+,m,d,k) |> gpu
l_sparsedict = train!(sparsedict,date()*"/sparsedict",L,mnist,opt,epochs)

distenc_eucl = EncoderBlock(DistEnc,h,+,m,d,relu,inveucl) |> gpu
l_distenc_eucl = train!(distenc_eucl,date()*"/distenc/eucl",L,mnist,opt,epochs)

distenc_cossim = EncoderBlock(DistEnc,h,+,m,d,relu,cossim) |> gpu
l_distenc_cossim = train!(distenc_cossim,date()*"/distenc/cossim",L,mnist,opt,epochs)

distpart_eucl = EncoderBlock(DistPart,h,+,m,d,k,relu,inveucl) |> gpu
l_distpart_eucl = train!(distpart_eucl,date()*"/distpart/eucl",L,mnist,opt,epochs)

distpart_cossim = EncoderBlock(DistPart,h,+,m,d,k,relu,cossim) |> gpu
l_distpart_cossim = train!(distpart_cossim,date()*"/distpart/cossim",L,mnist,opt,epochs)

models = loadmodels(models)
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

plotloss(l,label,"loss",date(),"block/loss.pdf",ylims=(0,5))
plotloss(l1,label,"L1",date(),"block/L1.pdf",ylims=(0,5))
plotloss(l2,label,"MSE",date(),"block/L2.pdf",ylims=(0,0.03))

save(date()*"/input.pdf",imgbatch(x_sub))
save(date()*"/decoded.pdf",imgbatch((outer[:decoder] ∘ outer[:encoder])(x_sub)))

decodedimg = m->imgbatch((outer[:decoder] ∘ m ∘ outer[:encoder])(x_sub))

map(m->save(m[3]*"/decoded.pdf",decodedimg(m[1])), models)

map(M->save(M[3]*"/decoded_head.pdf",
            hcat(map(decodedimg,M[1])...)),
    models)

E = outer[:encoder](x)
K = outer[:classifier](E)

function heatmap(M::AbstractMatrix,col::Tuple{String,String},lab::String;
                 kwargs...)
    if stats.quantile(M,0.01)[1] >= 0
        col = col[2]
    end
    col = colorfn(M,col)
    width,height = size(M)
    return CH.Heatmap(M',col,lab;
                      border=true,
                      show_row_dend=false,show_column_dend=false,
                      width=width,height=height,
                      kwargs...);
end

function heatmaps(M::EncoderBlock,f::Function,cols::Tuple{AbstractString,AbstractString},
                  lab::AbstractString,colsp::AbstractString;
                  kwargs...)
    h = size(M)
    F = mapreduce(f,vcat,M,E)
    d = size(F)[1] // h
    colsp = mapreduce(i->rep(colsp*"_"*string(i),d),vcat,1:h)
    return heatmap(F,cols,lab;
                   column_split=colsp, kwargs...);
end

hm_lab = heatmap(K,("white","black"),"P(label)",split=labels);
hm_E = heatmap(E,("white","purple"),"embedding",split=labels);

heatmaps_F = (M;kwargs...)->begin
    F = mapreduce(encode,vcat,M,E)
    colsp = mapreduce(i->rep("F_"*string(i),d),vcat,1:size(M))
    return heatmap(F,("blue","red"),"feature";
                   column_split=colsp, kwargs...);
end

hm_F = map(heatmaps_F,models,split=labels);

mapkey(models) do M
    Ehat = M[1](E)
    colsp = vcat(rep("E",m)...,rep("Ê",m)...)
    hm_E = heatmap(vcat(E,Ehat),("darkorange","purple"),"embedding",
                   split=labels,column_split=colsp);
    cellwidth = 5 / batchsize
    width =  m + h * d + 10
    F = mapreduce(encode,vcat,M[1],E)
    hm_F = heatmaps(M[1],encode,("blue","red"),"feature","F";split=labels);

    if hasmethod(cluster,Tuple{typeof(heads(M[1])[1]),Any})
        K = mapreduce(cluster,vcat,M[1],E)
        hm_K = heatmaps(M[1],cluster,("white","blue"),"P(cluster)","K";split=labels);
        hm_F = hm_F + hm_K
        width =  width + k * h
    end

    grD.pdf(M[3]*"/embedding.pdf",
            width = cellwidth * width + 2)
    CH.draw(hm_lab + hm_E + hm_F)
    grD.dev_off()
end

map(M->map((m,i)->KEheatmap(m,E,M[3],"KE_"*string(i)*".pdf"),
           heads(M[1]),1:size(M[1])),models)
