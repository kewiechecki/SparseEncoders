include("prelude.jl")

epochs = 100

mnist = mnistloader(batchsize)
path_nowd = path*"outer/nowd/"
path_wd = path*"outer/wd/"

outer = loadouter(m, path_nowd)
outer_wd = loadouter(m, path_wd)

function traininner(f::Function,dir::String)
    traininner(f, outer,α,mnist,opt,epochs,path_nowd*"/"*dir*"/nowd")
    traininner(f, outer,α,mnist,opt_wd,epochs,path_nowd*"/"*dir*"/wd")
    traininner(f, outer_wd,α,mnist,opt,epochs,path_wd*"/"*dir*"/nowd")
    traininner(f, outer_wd,α,mnist,opt_wd,epochs,path_wd*"/"*dir*"/wd")
end

traininner(()->SAE(m,d), "SAE")
traininner(()->PSAE(m,d,k), "PSAE")
traininner(()->DictEnc(m,d,k),"sparsedict")
traininner(()->DistEnc(m,d,relu,inveucl), "distenc/eucl")
traininner(()->DistEnc(m,d,relu,cossim), "distenc/cossim")
traininner(()->DistPart(m,d,k,relu,inveucl), "distpart/eucl")
traininner(()->DistPart(m,d,k,relu,cossim), "distpart/cossim")
