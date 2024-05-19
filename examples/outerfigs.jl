include("prelude.jl")
path_nowd = path*"outer/combined/nowd"
path_wd = path*"outer/combined/wd"

outer = loadmnist(m,path_nowd)
outer_wd = loadmnist(m,path_wd)

writeimg(x,path_nowd,"encoded.pdf",x->imgbatch(Chain(outer[:encoder],outer[:decoder]),x))

plotloss(mapreduce(x->x[:L].L,hcat,[outer,outer_wd]),"loss",path*"/outer/combined/loss.pdf")
