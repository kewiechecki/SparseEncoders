function writefigs(path,m,d,k)
    outer = loadouter(m, path)
    sae = loadmodels(()->SAE(m,d),path*"SAE/nowd")
    sae_wd = loadmodels(()->SAE(m,d),path*"SAE/wd")

    psae = loadmodels(()->PSAE(m,d,k),path*"PSAE/nowd")
    psae_wd = loadmodels(()->PSAE(m,d,k),path*"PSAE/wd")

    sparsedict = loadmodels(()->DictEnc(m,d,k),path*"sparsedict/nowd")
    sparsedict_wd = loadmodels(()->DictEnc(m,d,k),path*"sparsedict/wd")

    distenc_eucl = loadmodels(()->DistEnc(m,d,relu,inveucl),path*"distenc/eucl/nowd")
    distenc_eucl_wd = loadmodels(()->DistEnc(m,d,relu,inveucl),path*"distenc/eucl/wd")
    distenc_cossim = loadmodels(()->DistEnc(m,d,relu,cossim),path*"distenc/cossim/nowd")
    distenc_cossim_wd = loadmodels(()->DistEnc(m,d,relu,cossim),path*"distenc/cossim/wd")

    distpart_eucl = loadmodels(()->DistPart(m,d,k,relu,inveucl),path*"distpart/eucl/nowd")
    distpart_eucl_wd = loadmodels(()->DistPart(m,d,k,relu,inveucl),path*"distpart/eucl/wd")
    distpart_cossim = loadmodels(()->DistPart(m,d,k,relu,cossim),path*"distpart/cossim/nowd")
    distpart_cossim_wd = loadmodels(()->DistPart(m,d,k,relu,cossim),path*"distpart/cossim/wd")

    lab_inner = ["SAE","PSAE","DictEncoder","EuclDist","CossimDist",
                "PartitionedEucl","PartitionedCossim"]

    inner_nowd = [sae,psae,sparsedict,
                distenc_eucl,distenc_cossim,
                distpart_eucl,distpart_cossim]

    inner_wd = [sae_wd,psae_wd,sparsedict_wd,
                distenc_eucl_wd,distenc_cossim_wd,
                distpart_eucl_wd,distpart_cossim_wd]

    plotloss(map(M->M[:L_encoder][2],inner_nowd),
            lab_inner,"MSE",path,"L_inner_encoder.pdf",ylims=(0,100));
    plotloss(map(M->M[:L_decoder][2],inner_nowd),
            lab_inner,"MSE",path,"L_inner_decoder.pdf",ylims=(0,0.05));
    plotloss(map(M->M[:L_classifier][2],inner_nowd),
            lab_inner,"logitCE",path,"L_inner_classifier.pdf");

    plotloss(map(M->M[:L2_encoder][2],inner_nowd),
            lab_inner,"MSE",path,"L2_inner_encoder.pdf",ylims=(0,100));
    plotloss(map(M->M[:L2_decoder][2],inner_nowd),
            lab_inner,"MSE",path,"L2_inner_decoder.pdf",ylims=(0,0.05));
    plotloss(map(M->M[:L2_classifier][2],inner_nowd),
            lab_inner,"logitCE",path,"L2_inner_classifier.pdf");

    plotloss(map(M->M[:L_encoder][2],inner_wd),
            lab_inner,"MSE",path,"L_inner_encoder_wd.pdf",ylims=(0,100));
    plotloss(map(M->M[:L_decoder][2],inner_wd),
            lab_inner,"MSE",path,"L_inner_decoder_wd.pdf",ylims=(0,0.01));
    plotloss(map(M->M[:L_classifier][2],inner_wd),
            lab_inner,"logitCE",path,"L_inner_classifier_wd.pdf");

    plotloss(map(M->M[:L2_encoder][2],inner_wd),
            lab_inner,"MSE",path,"L2_inner_encoder_wd.pdf",ylims=(0,100));
    plotloss(map(M->M[:L2_decoder][2],inner_wd),
            lab_inner,"MSE",path,"L2_inner_decoder_wd.pdf",ylims=(0,0.01));
    plotloss(map(M->M[:L2_classifier][2],inner_wd),
            lab_inner,"logitCE",path,"L2_inner_classifier_wd.pdf");

    f = (x,i,out)->writeimg(x,path*"/denoised/nowd",out,
                            x->imgbatch(outer_wd,map(M->M[:L_encoder][1],inner_nowd),x))
    f(x,:L_classifier,"L_classifier.pdf")
    f(x,:L2_classifier,"L2_classifier.pdf")
    f(x,:L_encoder,"L_encoder.pdf")
    f(x,:L2_encoder,"L2_encoder.pdf")

    f = (x,i,out)->writeimg(x,path*"/denoised/wd",out,
                            x->imgbatch(outer_wd,map(M->M[:L_encoder][1],inner_wd),x))
    f(x,:L_classifier,"L_classifier.pdf")
    f(x,:L2_classifier,"L2_classifier.pdf")
    f(x,:L_encoder,"L_encoder.pdf")
    f(x,:L2_encoder,"L2_encoder.pdf")
end


