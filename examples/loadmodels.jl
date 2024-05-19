function loadmnist(m,path)
    θ = outerenc() |> gpu
    ϕ = outerdec() |> gpu
    ψ = outerclassifier() |> gpu
    M = Chain(θ,Parallel((args...)->args,ϕ,ψ))

    state_M = JLD2.load(path*"/final.jld2","state")
    Flux.loadmodel!(M,state_M)

    L = CSV.File(path*"/loss.csv") |> DataFrame
    rename!(L,["L","MSE","CE"])

    return Dict([(:encoder,θ),
                 (:classifier,ψ),
                 (:decoder,ϕ),
                 (:L,L),
                 (:path,path)])
end

macro loadinner(path,m,d,k)
    return esc(quote
        sae = loadinner(()->SAE($m,$d),$path*"SAE/")
        psae = loadinner(()->PSAE($m,$d,$k),$path*"PSAE/")
        sparsedict = loadinner(()->DictEnc($m,$d,$k),$path*"sparsedict/")
        distenc_eucl = loadinner(()->DistEnc($m,$d,relu,inveucl),$path*"distenc/eucl/")
        distenc_cossim = loadinner(()->DistEnc($m,$d,relu,cossim),$path*"distenc/cossim/")
        distpart_eucl = loadinner(()->DistPart($m,$d,$k,relu,inveucl),$path*"distpart/eucl/")
        distpart_cossim = loadinner(()->DistPart($m,$d,$k,relu,cossim),$path*"distpart/cossim/")
                   models = [sae,psae,
                             distenc_eucl,distenc_cossim,
                             distpart_eucl,distpart_cossim,
                             sparsedict]
                   models = map(m->begin
                           l = mapreduce(row->parse.(Float64,
                                                        (split∘foldl)((l,x)->replace(l,x => ""),
                                                                      ["[","]"],init=row)),
                                         hcat,m[2])'
                           return m[1],l,m[3]
                           end,models)

    end)
end

macro loadmodels(path,m,d,k)
    return esc(quote
        outer = loadouter($m, $path)

        sae = loadmodels(()->SAE($m,$d),$path*"SAE/nowd")
        sae_wd = loadmodels(()->SAE($m,$d),$path*"SAE/wd")

        psae = loadmodels(()->PSAE($m,$d,$k),$path*"PSAE/nowd")
        psae_wd = loadmodels(()->PSAE($m,$d,$k),$path*"PSAE/wd")

        sparsedict = loadmodels(()->DictEnc($m,$d,$k),$path*"sparsedict/nowd")
        sparsedict_wd = loadmodels(()->DictEnc($m,$d,$k),$path*"sparsedict/wd")

        distenc_eucl = loadmodels(()->DistEnc($m,$d,relu,inveucl),$path*"distenc/eucl/nowd")
        distenc_eucl_wd = loadmodels(()->DistEnc($m,$d,relu,inveucl),$path*"distenc/eucl/wd")
        distenc_cossim = loadmodels(()->DistEnc($m,$d,relu,cossim),$path*"distenc/cossim/nowd")
        distenc_cossim_wd = loadmodels(()->DistEnc($m,$d,relu,cossim),$path*"distenc/cossim/wd")

        distpart_eucl = loadmodels(()->DistPart($m,$d,$k,relu,inveucl),$path*"distpart/eucl/nowd")
        distpart_eucl_wd = loadmodels(()->DistPart($m,$d,$k,relu,inveucl),$path*"distpart/eucl/wd")
        distpart_cossim = loadmodels(()->DistPart($m,$d,$k,relu,cossim),$path*"distpart/cossim/nowd")
        distpart_cossim_wd = loadmodels(()->DistPart($m,$d,$k,relu,cossim),$path*"distpart/cossim/wd")

        lab_inner = ["SAE","PSAE","DictEncoder","EuclDist","CossimDist",
                    "PartitionedEucl","PartitionedCossim"]

        inner_nowd = [sae,psae,sparsedict,
                    distenc_eucl,distenc_cossim,
                    distpart_eucl,distpart_cossim]

        inner_wd = [sae_wd,psae_wd,sparsedict_wd,
                    distenc_eucl_wd,distenc_cossim_wd,
                    distpart_eucl_wd,distpart_cossim_wd]
    end)
end

macro plotloss(path,
               ylim_enc,
               ylim_dec_nowd,
               ylim_dec_wd)
    return esc(quote
                plotloss(map(M->M[:L_encoder][2],inner_nowd),
                        lab_inner,"MSE",$path,"L_inner_encoder.pdf",ylims=$ylim_enc);
                plotloss(map(M->M[:L_decoder][2],inner_nowd),
                        lab_inner,"MSE",$path,"L_inner_decoder.pdf",ylims=$ylim_dec_nowd);
                plotloss(map(M->M[:L_classifier][2],inner_nowd),
                        lab_inner,"logitCE",$path,"L_inner_classifier.pdf");

                plotloss(map(M->M[:L2_encoder][2],inner_nowd),
                        lab_inner,"MSE",$path,"L2_inner_encoder.pdf",ylims=$ylim_enc);
                plotloss(map(M->M[:L2_decoder][2],inner_nowd),
                        lab_inner,"MSE",$path,"L2_inner_decoder.pdf",ylims=$ylim_dec_nowd);
                plotloss(map(M->M[:L2_classifier][2],inner_nowd),
                        lab_inner,"logitCE",$path,"L2_inner_classifier.pdf");

                plotloss(map(M->M[:L_encoder][2],inner_wd),
                        lab_inner,"MSE",$path,"L_inner_encoder_wd.pdf",ylims=$ylim_enc);
                plotloss(map(M->M[:L_decoder][2],inner_wd),
                        lab_inner,"MSE",$path,"L_inner_decoder_wd.pdf",ylims=$ylim_dec_wd);
                plotloss(map(M->M[:L_classifier][2],inner_wd),
                        lab_inner,"logitCE",$path,"L_inner_classifier_wd.pdf");

                plotloss(map(M->M[:L2_encoder][2],inner_wd),
                        lab_inner,"MSE",$path,"L2_inner_encoder_wd.pdf",ylims=$ylim_enc);
                plotloss(map(M->M[:L2_decoder][2],inner_wd),
                        lab_inner,"MSE",$path,"L2_inner_decoder_wd.pdf",ylims=$ylim_dec_wd);
                plotloss(map(M->M[:L2_classifier][2],inner_wd),
                        lab_inner,"logitCE",$path,"L2_inner_classifier_wd.pdf");
               end)
end

function plotdenoised(outer,inner::Dict,x::AbstractArray,path::String)
    f = (x,i,out)->writeimg(x,path,out,
                            x->imgbatch(outer,map(M->M[i][1],inner),x))
    f(x,:L_classifier,"L_classifier.pdf")
    f(x,:L2_classifier,"L2_classifier.pdf")
    f(x,:L_encoder,"L_encoder.pdf")
    f(x,:L2_encoder,"L2_encoder.pdf")
    f(x,:L_decoder,"L_decoder.pdf")
    f(x,:L2_decoder,"L2_decoder.pdf")
end

function plotdenoised(outer,inner::Tuple{SparseEncoder,AbstractArray,String},x::AbstractArray)
    writeimg(x,inner[3],"denoised.pdf",
             x->imgbatch(outer,inner[1],x))
end
