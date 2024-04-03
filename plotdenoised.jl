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
