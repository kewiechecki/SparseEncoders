include("prelude.jl")

batchsize=128

mnist = mnistloader(batchsize)
x,y = first(mnist) |> gpu
labels = string.(unhot(cpu(y)))[1,:];
x_sub = gpu(cpu(x)[:,:,:,1:27])

path_nowd = path*"outer/nowd/"
path_wd = path*"outer/wd/"

include("loadmodels.jl")

@loadmodels path_nowd m d k

ylim_enc = (0,3000)
ylim_dec_nowd = (0,0.05)
ylim_dec_wd = (0,0.03)

@plotloss path_nowd ylim_enc ylim_dec_nowd ylim_dec_wd
plotdenoised(outer,inner_nowd,x,path_nowd*"/denoised/nowd")
plotdenoised(outer,inner_wd,x,path_nowd*"/denoised/wd")

E = outer[:encoder](x)
K = outer[:classifier](E)
E_SAE = encode(sae[:L_decoder][1],E)
Ehat = decode(sae[:L_decoder][1],E_SAE)
Khat = outer[:classifier](Ehat)

save(path_nowd*"/encoder/input.pdf",imgbatch(x_sub))
save(path_nowd*"/encoder/decoded.pdf",imgbatch((outer[:decoder] ∘ outer[:encoder])(x_sub)))

map(M->map(m->save(m[3]*"/decoded.pdf",
                   imgbatch((outer[:decoder] ∘ m[1] ∘ outer[:encoder])(x_sub))),
           M),
    inner_nowd)

M_E = vcat(E,E_SAE,Ehat)'
colsp = vcat(rep("E_outer",m), rep("E_SAE",d), rep("Ehat",m))
h,w = size(M_E)
hm_E = CH.Heatmap(M_E,"embedding",
                  col=colorfn(E,"red"),
                  border=true,
                  show_row_dend=false,show_column_dend=false,
                  width=w,height=h,
                  split=labels,column_split=colsp);

M_K = vcat(K,Khat)'
h,w = size(M_K)
colsp = vcat(rep("K",10),rep("Khat",10))
hm_K = CH.Heatmap(M_K,"P(label)",col=["white","blue"],border=true,
                  width=w,height=h,
                  show_row_dend=false,show_column_dend=false,
                  split=labels,column_split=colsp);

grD.pdf(path_nowd*"/classifier/KE_lab.pdf",width = ((7 / batchsize) * (2*m+d+2*10))+3)
CH.draw(hm_K + hm_E)
grD.dev_off()

plotloss([sae[:L_decoder][2]],["SAE"],"MSE",path_nowd,"loss_SAE.pdf",ylims=ylim_dec_nowd);
plotloss([sae[:L_decoder][2],distenc_cossim[:L_decoder][2]],
          ["SAE","DistEnc"],"MSE",path_nowd,"loss_DistEnc.pdf",ylims=ylim_dec_nowd);
plotloss([sae[:L_decoder][2],
          distenc_cossim[:L_decoder][2],
          distpart_cossim[:L_decoder][2]],
          ["SAE","DistEnc","DistPart"],"MSE",path_nowd,"loss_DistPart.pdf",ylims=ylim_dec_nowd);
plotloss([sae[:L_decoder][2],
          distenc_cossim[:L_decoder][2],
          distpart_cossim[:L_decoder][2],
          psae[:L_decoder][2]],
         ["SAE","DistEnc","DistPart","PSAE"],"MSE",
         path_nowd,"loss_PSAE.pdf",ylims=ylim_dec_nowd);
plotloss([sae[:L_decoder][2],
          distenc_cossim[:L_decoder][2],
          distpart_cossim[:L_decoder][2],
          psae[:L_decoder][2],
          sparsedict[:L_decoder][2]],
         ["SAE","DistEnc","DistPart","PSAE","DictEnc"],"MSE",
         path_nowd,"loss_DictEnc.pdf",ylims=ylim_dec_nowd);

hm_E = CH.Heatmap(E',"embedding",col=["white","red"],border=true,
                  show_row_dend=false,show_column_dend=false,
                  width=2*m+d,height=batchsize);
hm_K = CH.Heatmap(K',"P(label)",col=["white","blue"],border=true,
                  show_row_dend=false,show_column_dend=false,
                  width=20,height=batchsize);

grD.pdf(path_nowd*"/classifier/KE.pdf",width = ((10 / batchsize) * (2*m+d+10))+2)
CH.draw(hm_K + hm_E)
grD.dev_off()

map(M->KEheatmap(M,E),vcat(inner_nowd,inner_wd))

map(M->featimg(M,E,outer[:decoder]),inner_wd)

l = [sae[:L_decoder],distenc_eucl[:L_decoder],distpart_eucl[:L_decoder],psae[:L_decoder],sparsedict[:L_decoder]]
map(m->KEheatmap(m[1],E,m[3],"KE.pdf"),l)

KEheatmap(psae[:L_encoder][1],E,path_nowd*"/PSAE/nowd/encoder/L1_L2","KE.pdf")
KEheatmap(distenc_cossim[:L_encoder][1],E,path_nowd*"/distenc/cossim/nowd/encoder/L1_L2","KE.pdf")
KEheatmap(distpart_cossim[:L_encoder][1],E,path_nowd*"/distpart/cossim/nowd/encoder/L1_L2","KE.pdf")

lab_inner = ["SAE","PSAE","DictEncoder","EuclDist","CossimDist",
            "PartitionedEucl","PartitionedCossim"]

inner_nowd = [sae,psae,sparsedict,
              distenc_eucl,distenc_cossim,
              distpart_eucl,distpart_cossim]

inner_wd = [sae_wd,psae_wd,sparsedict_wd,
            distenc_eucl_wd,distenc_cossim_wd,
            distpart_eucl_wd,distpart_cossim_wd]

plotloss(map(M->M[:L_encoder][2],inner_nowd),
         lab_inner,"MSE",path_wd,"L_inner_encoder.pdf",ylims=(0,100));
plotloss(map(M->M[:L_decoder][2],inner_nowd),
         lab_inner,"MSE",path_wd,"L_inner_decoder.pdf",ylims=(0,0.05));
plotloss(map(M->M[:L_classifier][2],inner_nowd),
         lab_inner,"logitCE",path_wd,"L_inner_classifier.pdf");

plotloss(map(M->M[:L2_encoder][2],inner_nowd),
         lab_inner,"MSE",path_wd,"L2_inner_encoder.pdf",ylims=(0,100));
plotloss(map(M->M[:L2_decoder][2],inner_nowd),
         lab_inner,"MSE",path_wd,"L2_inner_decoder.pdf",ylims=(0,0.05));
plotloss(map(M->M[:L2_classifier][2],inner_nowd),
         lab_inner,"logitCE",path_wd,"L2_inner_classifier.pdf");

plotloss(map(M->M[:L_encoder][2],inner_wd),
         lab_inner,"MSE",path_wd,"L_inner_encoder_wd.pdf",ylims=(0,100));
plotloss(map(M->M[:L_decoder][2],inner_wd),
         lab_inner,"MSE",path_wd,"L_inner_decoder_wd.pdf",ylims=(0,0.01));
plotloss(map(M->M[:L_classifier][2],inner_wd),
         lab_inner,"logitCE",path_wd,"L_inner_classifier_wd.pdf");

plotloss(map(M->M[:L2_encoder][2],inner_wd),
         lab_inner,"MSE",path_wd,"L2_inner_encoder_wd.pdf",ylims=(0,100));
plotloss(map(M->M[:L2_decoder][2],inner_wd),
         lab_inner,"MSE",path_wd,"L2_inner_decoder_wd.pdf",ylims=(0,0.01));
plotloss(map(M->M[:L2_classifier][2],inner_wd),
         lab_inner,"logitCE",path_wd,"L2_inner_classifier_wd.pdf");

f = (x,i,out)->writeimg(x,path_wd*"/denoised/nowd",out,
                        x->imgbatch(outer_wd,map(M->M[:L_encoder][1],inner_nowd),x))
f(x,:L_classifier,"L_classifier.pdf")
f(x,:L2_classifier,"L2_classifier.pdf")
f(x,:L_encoder,"L_encoder.pdf")
f(x,:L2_encoder,"L2_encoder.pdf")

f = (x,i,out)->writeimg(x,path_wd*"/denoised/wd",out,
                        x->imgbatch(outer_wd,map(M->M[:L_encoder][1],inner_wd),x))
f(x,:L_classifier,"L_classifier.pdf")
f(x,:L2_classifier,"L2_classifier.pdf")
f(x,:L_encoder,"L_encoder.pdf")
f(x,:L2_encoder,"L2_encoder.pdf")

f = (x,i,out)->writeimg(x,date()*"/denoised/wd",out,x->imgbatch(outer,sae_wd[i][1],psae_wd[i][1],x))
f(x,:L_classifier,"L_classifier.pdf")
f(x,:L2_classifier,"L2_classifier.pdf")
f(x,:L_encoder,"L_encoder.pdf")
f(x,:L2_encoder,"L2_encoder.pdf")

tmp = imgbatch(outer,sae[:L_classifier][1],psae[:L_classifier][1],x)
img = colorview(Gray,x[:,:,1,1]')
writeimg(x,date(),"tmp.pdf",x->imgbatch(outer,sae[:L_classifier][1],psae[:L_classifier][1],x))

newplot = ylab->scatter(xlabel="batch",ylabel=ylab)
f! = (L,lab)->scatter!(1:length(L),L,label=lab)

p = newplot("logitCE");
map(x->f!(x...),[(outer[:L_classifier],"no WD"),
                 (outer_wd[:L_classifier],"WD")]);
Plots.savefig(p,"data/MNIST/L_outer_classifier.pdf")

p = newplot("MSE");
map(x->f!(x...),[(outer[:L_encoder],"no WD"),
                 (outer_wd[:L_encoder],"WD")]);
savefig(p,"data/MNIST/L_outer_encoder.pdf")

p = newplot("logitCE");
map(x->points!(x...),[(sae[:L_classifier][2],"L1+L2, no WD"),
                 (sae[:L2_classifier][2],"L2, no WD"),
                 (sae_wd[:L_classifier][2],"L1+L2, WD"),
                 (sae_wd[:L2_classifier][2],"L2, WD")]);
savefig(p,"data/MNIST/L_SAE_classifier.pdf")

p = newplot("MSE");
map(x->f!(x...),[(sae[:L_encoder][2],"L1+L2, no WD"),
                 (sae[:L2_encoder][2],"L2, no WD"),
                 (sae_wd[:L_encoder][2],"L1+L2, WD"),
                 (sae_wd[:L2_encoder][2],"L2, WD")]);
savefig(p,"data/MNIST/L_SAE_encoder.pdf")

p = newplot("logitCE");
map(x->f!(x...),[(psae[:L_classifier][2],"L1+L2, no WD"),
                 (psae[:L2_classifier][2],"L2, no WD"),
                 (psae_wd[:L_classifier][2],"L1+L2, WD"),
                 (psae_wd[:L2_classifier][2],"L2, WD")]);
savefig(p,"data/MNIST/L_PSAE_classifier.pdf")

p = newplot("MSE");
map(x->f!(x...),[(psae[:L_encoder][2],"L1+L2, no WD"),
                 (psae[:L2_encoder][2],"L2, no WD"),
                 (psae_wd[:L_encoder][2],"L1+L2, WD"),
                 (psae_wd[:L2_encoder][2],"L2, WD")]);
savefig(p,"data/MNIST/L_PSAE_encoder.pdf")

E_outer = outer[:encoder](x)
K_outer = outer[:classifier](E_outer)

E_SAE = encode(sae_wd[:L_classifier][1],E_outer)
E_PSAE = encode(psae_wd[:L_classifier][1],E_outer)
Ehat_PSAE = encodepred(psae_wd[:L_classifier][1],E_outer)

covheatmap(E_PSAE,"embedding",path*"inner/wd/classifier/PSAE/cov_E.pdf")

C_SAE = cossim(E_SAE)
C_feat_SAE = cossim(E_SAE')
C_PSAE = cossim(E_PSAE)
Chat_PSAE = cossim(Ehat_PSAE)

sae_clas,L_clas = loadsae(m,d,path*"inner/nowd/classifier/SAE/L1_L2/")
sae_clas_L2,L2_clas = loadsae(m,d,path*"inner/nowd/classifier/SAE/L2/")
sae_clas_wd,L_clas_wd = loadsae(m,d,path*"inner/wd/classifier/SAE/L1_L2/")
sae_clas_L2_wd,L2_clas_wd = loadsae(m,d,path*"inner/wd/classifier/SAE/L2/")

sae_enc,L_enc = loadsae(m,d,path*"inner/nowd/encoder/SAE/L1_L2/")
sae_enc_L2,L2_enc = loadsae(m,d,path*"inner/nowd/encoder/SAE/L2/")
sae_enc_wd,L_enc_wd = loadsae(m,d,path*"inner/wd/encoder/SAE/L1_L2/")
sae_enc_L2_wd,L2_enc_wd = loadsae(m,d,path*"inner/wd/encoder/SAE/L2/")

psae_clas,L_PSAE_clas = loadpsae(m,d,k,path*"inner/nowd/classifier/PSAE/L1_L2/")
psae_clas_L2,L2_PSAE_clas = loadpsae(m,d,k,path*"inner/nowd/classifier/PSAE/L2/")
psae_clas_wd,L_PSAE_clas_wd = loadpsae(m,d,k,path*"inner/wd/classifier/PSAE/L1_L2/")
psae_clas_L2_wd,L2_PSAE_clas_wd = loadpsae(m,d,k,path*"inner/wd/classifier/PSAE/L2/")

psae_enc,L_PSAE_enc = loadpsae(m,d,k,path*"inner/nowd/encoder/PSAE/L1_L2/")
psae_enc_L2,L2_PSAE_enc = loadpsae(m,d,k,path*"inner/nowd/encoder/PSAE/L2/")
psae_enc_wd,L_PSAE_enc_wd = loadpsae(m,d,k,path*"inner/wd/encoder/PSAE/L1_L2/")
psae_enc_L2_wd,L2_PSAE_enc_wd = loadpsae(m,d,k,path*"inner/wd/encoder/PSAE/L2/")

f = (L,lab,ylab)->scatter(1:length(L),L,
                          xlabel="batch", ylabel=ylab,
                          label=lab)

f! = (L,lab)->scatter!(1:length(L),L,label=lab)

p = f(L_clas,"L1+L2, no WD","logitCE");
map(x->f!(x...),[(L2_clas,"L2, no WD"),
        (L_clas_wd,"L1+L2, WD"),
        (L2_clas_wd,"L2, WD")]);
savefig(p,"data/MNIST/L_SAE_clas.pdf")

p = f(L_PSAE_clas,"L1+L2, no WD","logitCE");
map(x->f!(x...),[(L2_clas,"L2, no WD"),
        (L_PSAE_clas_wd,"L1+L2, WD"),
        (L2_PSAE_clas_wd,"L2, WD")]);
savefig(p,"data/MNIST/L_PSAE_clas.pdf");

sp = scatter(1:length(L_clas),L_clas
p = scatter(1:length(L_SAE), L_SAE,
            xlabel="batch",ylabel="loss",
            label="SAE")
scatter!(1:length(L_PSAE), L_PSAE,label="PSAE")
savefig(p,"data/MNIST/loss_relu.svg")
