include("prelude.jl")

mnist = mnistloader(batchsize)

outer = trainouter(m,mnist,opt,epochs,
                           path*"outer/nowd/")

outer_wd = trainouter(m,mnist,opt_wd,epochs,
                      path*"outer/wd/")

outer_combined = combinedouter(m,mnist,opt_wd,opt,epochs,path*"outer/combined/")

sae = trainsae(m,d,outer,α,mnist,opt,epochs,
               path*"inner/nowd/")
sae_wd = trainsae(m,d,outer,α,mnist,opt_wd,epochs,
                  path*"inner/wd/")

psae = trainpsae(m,d,k,outer,α,mnist,opt,epochs,
                 path*"inner/nowd/")
psae_wd = trainpsae(m,d,k,outer,α,mnist,opt_wd,epochs,
                    path*"inner/wd/")

