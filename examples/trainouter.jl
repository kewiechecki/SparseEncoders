include("prelude.jl")

mnist = mnistloader(batchsize)
x,y = first(mnist) |> gpu

epochs = 100

function loss(M,x,y)
    x̂,ŷ = M(x)
    MSE = Flux.mse(x̂,x)
    CE = Flux.crossentropy(ŷ,y)
    return MSE + CE,MSE,CE
end

function trainmnist(path,opt,epochs)
    θ = outerenc() |> gpu
    ϕ = outerdec() |> gpu
    ψ = outerclassifier() |> gpu
    outer = Chain(θ,Parallel((args...)->args,ϕ,ψ))

    L = train!(outer,path,loss,mnist,opt,epochs)
    return outer,L
end

trainmnist(path*"outer/combined/nowd",opt,epochs)
trainmnist(path*"outer/combined/wd",opt_wd,epochs)

outer = trainouter(m,mnist,opt,epochs,
                           path*"outer/nowd/")

outer_wd = trainouter(m,mnist,opt_wd,epochs,
                      path*"outer/wd/")

writeimg(x,path_nowd,"encoded.pdf",x->imgbatch(Chain(outer[:encoder],outer[:decoder]),x))
writeimg(x,path_wd,"encoded.pdf",x->imgbatch(Chain(outer_wd[:encoder],outer_wd[:decoder]),x))

newplot = ylab->scatter(xlabel="batch",ylabel=ylab)
f! = (L,lab)->scatter!(1:length(L),L,label=lab)

M = Chain(outer[:encoder],outer[:classifier])
checkpt = map(1:1000) do i
    state_M = JLD2.load(path*"outer/nowd/classifier/"*string(i)*".jld2","state")
    Flux.loadmodel!(M,state_M)
    return sum(M[1](x),dims=2)
end
E_nonzero = sum(hcat(checkpt...) .> 0,dims=1)
col = (i->ifelse(i,:red,:blue)).(cpu(E_nonzero) .< 3)
L = outer[:L_classifier]
p = scatter(1:length(L),L,xlabel="batch",ylabel="logitCE",c=col);
Plots.savefig(p,path*"L_outer_classifier_col.pdf");

p = newplot("logitCE");
map(x->f!(x...),[(outer[:L_classifier],"no WD"),
                 (outer_wd[:L_classifier],"WD")]);
Plots.savefig(p,path*"L_outer_classifier.pdf");

p = newplot("MSE");
p = scatter(xlabel="batch",ylabel="MSE",ylims=(0,0.1))
map(x->f!(x...),[(outer[:L_encoder],"no WD"),
                 (outer_wd[:L_encoder],"WD")]);
savefig(p,path*"L_outer_encoder.pdf");

L_clas = train!(outerclas,
                loader,opt,epochs,logitcrossentropy;
                savecheckpts=true,
                path=path*"nowd/classifier/");
M = Chain(θ,ϕ)
L_enc = train!(M,
               mnist,opt,epochs,Flux.mse;
               ignoreY=true,savecheckpts=true,
               path=path*"encoder/nowd/CE");
p = newplot("logitCE");
f!(L_enc,"no WD");
savefig(p,path*"encoder/nowd/CE/loss.pdf");

θ = outerenc() |> gpu
ϕ = outerdec() |> gpu
π = outerclassifier() |> gpu
outerclas = Chain(θ,π)

L_clas = train!(outerclas,
                loader,opt_wd,epochs,logitcrossentropy;
                savecheckpts=true,
                path=path*"wd/classifier/");
L_enc = train!(ϕ,
               loader,opt_wd,epochs,Flux.mse;
               prefn=outerclas[1],
               ignoreY=true,savecheckpts=true,
               path=path*"wd/encoder/");

M_outerclas = Chain(θ,π) |> gpu
L_outerclas = []

train!(M_outerclas,loader,opt,epochs,logitcrossentropy,L_outerclas,
       savecheckpts=true,path=path*"nowd/classifier/");

p = scatter(1:length(L_outerclas), L_outerclas,
            xlabel="batch",ylabel="loss",
            label="outer");
savefig(p,path*"classifier/loss.pdf")

ŷ = M_outerclas(x)
labels = string.(unhot(cpu(y)))[1,:]

grD.pdf(path*"classifier/logits.pdf")
CH.Heatmap(ŷ',"P(label)",col=["white","blue"],
           split=labels,border=true)
grD.dev_off()

p = scatter(1:length(L_outerenc), L_outerenc,
            xlabel="batch",ylabel="loss",
            label="outer");
savefig(p,path*"encoder/loss.pdf")

x,y = first(loader) |> gpu
colorview(Gray,x[:,:,1,1:2])
colorview(Gray,M_outerenc(x[:,:,1,1:2]))

