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

