using MLDatasets, JLD2, CSV

function mnistconv()
    kern = (3,3)
    s = (2,2)
    θ = Chain(Conv(kern,1 => 3,relu,stride=s),
                Conv(kern,3 => 6,relu,stride=s),
                Conv(kern,6 => 9,relu,stride=s),
                Conv((2,2),9 => 12,relu))
    return Chain(θ, x->reshape(x,12,:))
end

function mnistdeconv()
    kern = (3,3)
    s = (2,2)
    ϕ = Chain(ConvTranspose((2,2),12 => 9,relu),
                   ConvTranspose((4,4),9 => 6,relu,stride=s),
                   ConvTranspose(kern,6 => 3,relu,stride=s),
                   ConvTranspose((4,4),3 => 1,relu,stride=s))
    return Chain(x->reshape(x,1,1,12,:), ϕ)
end

function outerenc(m=3)
    kern = (3,3)
    s = (2,2)
    θ_conv = Chain(Conv(kern,1 => 3,relu,stride=s),
                Conv(kern,3 => 6,relu,stride=s),
                Conv(kern,6 => 9,relu,stride=s),
                Conv((2,2),9 => 12,relu))

    θ_mlp = Chain(Dense(12 => 6,relu),
                Dense(6 => m,relu))

    θ = Chain(θ_conv,
              x->reshape(x,12,:),
              θ_mlp)
    return θ
end

function outerclassifier(m=3,k=10)
    π = Chain(Dense(m => 5,relu),
                    Dense(5 => k,relu),
                    softmax)
    return π
end

function outerdec(m=3)
    kern = (3,3)
    s = (2,2)
    ϕ_mlp = Chain(Dense(m => 6,relu),
                  Dense(6 => 12,relu))
    ϕ_deconv = Chain(ConvTranspose((2,2),12 => 9,relu),
                   ConvTranspose((4,4),9 => 6,relu,stride=s),
                   ConvTranspose(kern,6 => 3,relu,stride=s),
                   ConvTranspose((4,4),3 => 1,relu,stride=s))
    ϕ = Chain(ϕ_mlp,
              x->reshape(x,1,1,12,:),
              ϕ_deconv)
    return ϕ
end

function outermodel()
    kern = (3,3)
    s = (2,2)
    θ_conv = Chain(Conv(kern,1 => 3,relu,stride=s),
                Conv(kern,3 => 6,relu,stride=s),
                Conv(kern,6 => 9,relu,stride=s),
                Conv((2,2),9 => 12,relu))

    θ_mlp = Chain(Dense(12 => 6,relu),
                Dense(6 => m,relu))

    θ_outer = Chain(θ_conv,
                    x->reshape(x,12,:),
                    θ_mlp)

    π_outer = Chain(Dense(m => 5,relu),
                    Dense(5 => 10,relu),
                    softmax)

    M_outer = Chain(θ_outer,π_outer)
    return M_outer
end

function trainouter(m::Integer,
                    dat::Flux.DataLoader,
                    opt::Flux.Optimiser,
                    epochs::Integer,
                    path::AbstractString)
    θ = outerenc(m) |> gpu
    ϕ = outerdec(m) |> gpu
    π = outerclassifier(m) |> gpu
    outerclas = Chain(θ,π)

    L_π = train!(outerclas,
                 dat,opt,epochs,logitcrossentropy;
                 savecheckpts=true,
                 path=path*"/classifier/");
    L_ϕ = train!(ϕ,
                 dat,opt,epochs,Flux.mse;
                 prefn=θ,
                 ignoreY=true,
                 savecheckpts=true,
                 path=path*"/encoder/");

    return Dict([(:encoder,θ),
                 (:classifier,π),
                 (:decoder,ϕ),
                 (:L_classifier,L_π),
                 (:L_encoder,L_ϕ)])
end


function combinedouter(m::Integer,
                       dat::Flux.DataLoader,
                       opt_enc::Flux.Optimiser,
                       opt_clas::Flux.Optimiser,
                       epochs::Integer,
                       path::AbstractString)
    θ = outerenc(m) |> gpu
    ϕ = outerdec(m) |> gpu
    π = outerclassifier(m) |> gpu
    M = Chain(θ,ϕ)
    L_ϕ = []
    L_π =[]

    @showprogress map(1:epochs) do i
        map(dat) do (x,y)
            x = gpu(x)
            y = gpu(y)

            push!(L_ϕ, update!(M,x,x,
                               logitcrossentropy,
                               opt_enc))
            push!(L_π,update!(π,θ(x),y,
                              Flux.mse,
                              opt_clas))

            savemodel(θ,path*"/encoder/",string(i))
            savemodel(ϕ,path*"/decoder/",string(i))
            savemodel(π,path*"/classifier/",string(i))
        end
        savemodel(θ,path,"encoder")
        savemodel(ϕ,L_ϕ,path,"decoder")
        savemodel(π,L_π,path,"classifier")
    end
    plotloss(L_ϕ,"MSE");
    plotloss(L_π,"logitCE");

    return Dict([(:encoder,θ),
                 (:classifier,π),
                 (:decoder,ϕ),
                 (:L_classifier,L_π),
                 (:L_encoder,L_ϕ)])
end

function traininner(f::Function,
                    outer::Dict,
                    α::AbstractFloat,
                    dat::Flux.DataLoader,
                    opt::Flux.Optimiser,
                    epochs::Integer,
                    path::AbstractString)
    θ = outer[:encoder]
    π = outer[:classifier]
    ϕ = outer[:decoder]

    loss_clas = (E,x)->logitcrossentropy((E),(outer[:classifier] ∘ θ)(x))
    loss_dec = (E,x)->Flux.mse(E,(ϕ ∘ θ)(x))
    loss_E = (E,x)->Flux.mse(E,θ(x))

    sae = f() |> gpu
    L_classifier = train!(sae,Chain(θ,outer[:classifier]),α,dat,opt,epochs,
                          logitcrossentropy;
                          path=path*"/classifier/L1_L2/")

    sae_L2 = f() |> gpu
    L2_classifier = train!(sae_L2,dat,opt,epochs,
                           loss_clas;
                           prefn=θ,postfn=outer[:classifier],
                           ignoreY=true, savecheckpts=true,
                           path=path*"/classifier/L2/")


    sae_enc = f() |> gpu
    L_encoder = train!(sae_enc,Chain(θ,identity),
                       α,dat,opt,epochs,
                       Flux.mse;
                       path=path*"/encoder/L1_L2/")

    sae_enc_L2 = f() |> gpu
    L2_encoder = train!(sae_enc_L2,
                        dat,opt,epochs,
                        loss_E;
                        prefn=θ,
                        ignoreY=true, savecheckpts=true,
                        path=path*"/encoder/L2/")

    sae_dec = f() |> gpu
    L_decoder = train!(sae_dec,Chain(θ,ϕ),
                       α,dat,opt,epochs,
                       Flux.mse;
                       path=path*"/decoder/L1_L2/")

    sae_dec_L2 = f() |> gpu
    L2_decoder = train!(sae_dec_L2,
                        dat,opt,epochs,
                        loss_dec;
                        prefn=θ,postfn=ϕ,
                        ignoreY=true, savecheckpts=true,
                        path=path*"/decoder/L2/")

    return Dict([(:L_classifier,(sae,L_classifier)),
                 (:L2_classifier,(sae_L2,L2_classifier)),
                 (:L_encoder,(sae_enc,L_encoder)),
                 (:L2_encoder,(sae_enc_L2,L2_encoder)),
                 (:L_decoder,(sae_dec,L_decoder)),
                 (:L2_decoder,(sae_dec_L2,L2_decoder))])
end

function trainsae(m::Integer,d::Integer,
                  outer::Dict,
                  loss::Function,
                  dat::Flux.DataLoader,
                  opt::Flux.Optimiser,
                  epochs::Integer,
                  path::AbstractString)
    sae = SAE(m,d) |> gpu
    L = train!(sae,dat,opt,epochs,loss,path=path)
    return sae,L
end

function trainsaes(m::Integer,
                  d::Integer,
                  outer::Dict,
                  α::AbstractFloat,
                  dat::Flux.DataLoader,
                  opt::Flux.Optimiser,
                  epochs::Integer,
                  path::AbstractString)
    θ = outer[:encoder]
    π = outer[:classifier]
    ϕ = outer[:decoder]

    loss_clas = (E,x)->logitcrossentropy(π(E),(π ∘ θ)(x))
    loss_enc = (E,x)->Flux.mse(ϕ(E),(ϕ ∘ θ)(x))
    loss_E = (E,x)->Flux.mse(E,θ(x))

    sae = SAE(m,d) |> gpu
    L_classifier = train!(sae,Chain(θ,π),α,dat,opt,epochs,
                          logitcrossentropy;
                          path=path*"/classifier/SAE/L1_L2/")

    sae_L2 = SAE(m,d) |> gpu
    L2_classifier = train!(sae_L2,dat,opt,epochs,
                           loss_clas;
                           prefn=θ,postfn=π,
                           ignoreY=true, savecheckpts=true,
                           path=path*"/classifier/SAE/L2/")


    sae_enc = SAE(m,d) |> gpu
    L_encoder = train!(sae_enc,Chain(θ,ϕ),
                       α,dat,opt,epochs,
                       Flux.mse;
                       path=path*"/encoder/SAE/L1_L2/")

    sae_enc_L2 = SAE(m,d) |> gpu
    L2_encoder = train!(sae_enc_L2,
                        dat,opt,epochs,
                        loss_enc;
                        prefn=θ,postfn=ϕ,
                        ignoreY=true, savecheckpts=true,
                        path=path*"/encoder/SAE/L2/")

    sae_dec = SAE(m,d) |> gpu
    L_decoder = train!(sae_dec,Chain(θ,ϕ),
                       α,dat,opt,epochs,
                       Flux.mse;
                       path=path*"/decoder/SAE/L1_L2/")

    sae_dec_L2 = SAE(m,d) |> gpu
    L2_decoder = train!(sae_dec_L2,
                        dat,opt,epochs,
                        Flux.mse;
                        prefn=θ,postfn=ϕ,
                        ignoreY=true, savecheckpts=true,
                        path=path*"/decoder/SAE/L2/")

    return Dict([(:L_classifier,(sae,L_classifier)),
                 (:L2_classifier,(sae_L2,L2_classifier)),
                 (:L_encoder,(sae_enc,L_encoder)),
                 (:L2_encoder,(sae_enc_L2,L2_encoder))])
end

function trainpsae(m::Integer,
                   d::Integer,
                   k::Integer,
                   outer::Dict,
                   α::AbstractFloat,
                   dat::Flux.DataLoader,
                   opt::Flux.Optimiser,
                   epochs::Integer,
                   path::AbstractString)
    θ = outer[:encoder]
    π = outer[:classifier]
    ϕ = outer[:decoder]

    loss_clas = (E,x)->logitcrossentropy(E,(π ∘ θ)(x))
    loss_enc = (E,x)->Flux.mse(E,(ϕ ∘ θ)(x))

    psae = PSAE(m,d,k) |> gpu
    L_classifier = train!(psae,Chain(θ,π),α,dat,opt,epochs,
                          logitcrossentropy;
                          path=path*"/classifier/PSAE/L1_L2/")

    psae_L2 = PSAE(m,d,k) |> gpu
    L2_classifier = train!(psae_L2,dat,opt,epochs,
                           loss_clas;
                           prefn=θ,postfn=π,
                           ignoreY=true, savecheckpts=true,
                           path=path*"/classifier/PSAE/L2/")

    psae_enc = PSAE(m,d,k) |> gpu
    L_encoder = train!(psae_enc,Chain(θ,ϕ),
                       α,dat,opt,epochs,
                       Flux.mse;
                       path=path*"/encoder/PSAE/L1_L2/")

    psae_enc_L2 = PSAE(m,d,k) |> gpu
    L2_encoder = train!(psae_enc_L2,
                        dat,opt,epochs,
                        loss_enc;
                        prefn=θ,postfn=ϕ,
                        ignoreY=true, savecheckpts=true,
                        path=path*"/encoder/PSAE/L2/")
    return Dict([(:L_classifier,(psae,L_classifier)),
                 (:L2_classifier,(psae_L2,L2_classifier)),
                 (:L_encoder,(psae_enc,L_encoder)),
                 (:L2_encoder,(psae_enc_L2,L2_encoder))])
end



function mnistloader(batchsize::Integer)
    dat = MNIST(split=:train)[:]
    target = onehotbatch(dat.targets,0:9)

    m_x,m_y,n = size(dat.features)
    X = reshape(dat.features[:, :, :], m_x, m_y, 1, n)

    loader = Flux.DataLoader((X,target),
                            batchsize=batchsize,
                            shuffle=true)
    return loader
end

function loader(dat::DataType,
                batchsize::Integer)
    X = dat(split=:train)[:]
    target = onehotbatch(X.targets,range(extrema(X.targets)...))
    loader = Flux.DataLoader((X,target),
                             batchsize=batchsize,
                             shuffle=true)
    return loader
end

function loadouter(m=3,path="data/MNIST/")
    θ = outerenc(m) |> gpu
    π = outerclassifier(m) |> gpu
    ϕ = outerdec(m) |> gpu
    M = Chain(θ,π)

    state_M = JLD2.load(path*"/classifier/final.jld2","state")
    Flux.loadmodel!(M,state_M)
    state_ϕ = JLD2.load(path*"/encoder/final.jld2","state")
    Flux.loadmodel!(ϕ,state_ϕ)

    L_π = CSV.File(path*"/classifier/loss.csv").Column1
    L_ϕ = CSV.File(path*"/encoder/loss.csv").Column1

    return Dict([(:encoder,θ),
                 (:classifier,π),
                 (:decoder,ϕ),
                 (:L_classifier,L_π),
                 (:L_encoder,L_ϕ)])
end

function loadsae(m,d,path="data/MNIST/inner/classifier/SAE/L1_L2/")
    sae = SAE(m,d) |> gpu
    state = JLD2.load(path*"/final.jld2","state")
    L = CSV.File(path*"/loss.csv").Column1
    Flux.loadmodel!(sae,state)
    return sae,L
end

function loadpsae(m,d,k,path="data/MNIST/inner/classifier/PSAE/L1_L2/")
    psae = PSAE(m,d,k) |> gpu
    state = JLD2.load(path*"/final.jld2","state")
    Flux.loadmodel!(psae,state)
    L = CSV.File(path*"/loss.csv").Column1
    return psae,L
end

function loadsaes(m,d,path)
    sae,L_clas = loadsae(m,d,path*"/classifier/SAE/L1_L2/")
    sae_L2,L2_clas = loadsae(m,d,path*"/classifier/SAE/L2/")
    sae_enc,L_enc = loadsae(m,d,path*"/encoder/SAE/L1_L2/")
    sae_enc_L2,L2_enc = loadsae(m,d,path*"/encoder/SAE/L2/")

    return Dict([(:L_classifier,(sae,L_clas)),
                 (:L2_classifier,(sae_L2,L2_clas)),
                 (:L_encoder,(sae_enc,L_enc)),
                 (:L2_encoder,(sae_enc_L2,L2_enc))])
end

function loadpsaes(m,d,k,path)
    sae,L_clas = loadpsae(m,d,k,path*"/classifier/PSAE/L1_L2/")
    sae_L2,L2_clas = loadpsae(m,d,k,path*"/classifier/PSAE/L2/")
    sae_enc,L_enc = loadpsae(m,d,k,path*"/encoder/PSAE/L1_L2/")
    sae_enc_L2,L2_enc = loadpsae(m,d,k,path*"/encoder/PSAE/L2/")

    return Dict([(:L_classifier,(sae,L_clas)),
                 (:L2_classifier,(sae_L2,L2_clas)),
                 (:L_encoder,(sae_enc,L_enc)),
                 (:L2_encoder,(sae_enc_L2,L2_enc))])
end

function readcsv(path::AbstractString)
    return CSV.File(path) |> DataFrame
end

function loadmodel(m,path::AbstractString)
    state = JLD2.load(path*"/final.jld2","state")
    L = readcsv(path*"/loss.csv")
    Flux.loadmodel!(m,state)

    #return Dict([(:model,m),(:loss,L),(:path,path)])
    return m,L,path
end   

function loadinner(f::Function,
                   path::String="data/MNIST/inner/classifier/SAE/L1_L2/")
    sae = f() |> gpu
    state = JLD2.load(path*"/final.jld2","state")
    L = CSV.File(path*"/loss.csv")
    Flux.loadmodel!(sae,state)
    return sae,L,path
end

function loadinner(dict::Dict)
    return map(m->loadinner(m[1],m[2]))
end


function loadmodels(f::Function,path::String)

    paths = Dict([(:L_classifier,"/classifier/L1_L2/"),
                 (:L2_classifier,"/classifier/L2/"),
                 (:L_encoder,"/encoder/L1_L2/"),
                 (:L2_encoder,"/encoder/L2/"),
                 (:L_decoder,"/decoder/L1_L2/"),
                 (:L2_decoder, "/decoder/L2/")])
    paths = map(p->path*p,paths)
    return map(p->loadinner(f,p),paths)
end

function innermodels(f::Function,m::Integer,d::Integer,k::Integer,
                     path::AbstractString,σ::Function=relu)
    M = Dict([(:sae,(f(SAE,m,d,σ) |> gpu,path*"/SAE")),
              (:psae, (f(PSAE,m,d,k,σ) |> gpu,path*"/PSAE")),
              (:sparsedict, (f(DictEnc,m,d,k,σ) |> gpu,path*"/sparsedict")),
              (:distenc_eucl, (f(DistEnc,m,d,σ,inveucl) |> gpu,path*"/distenc/eucl")),
              (:distenc_cossim, (f(DistEnc,m,d,σ,cossim) |> gpu,path*"/distenc/cossim")),
              (:distpart_eucl, (f(DistPart,m,d,k,σ,inveucl) |> gpu,path*"/distpart/eucl")),
              (:distpart_cossim, (f(DistPart,m,d,k,σ,cossim) |> gpu,path*"/distpart/cossim"))])
    return M
end

function innermodels(f::Function,
                     f_enc::Function,f_dec::Function,f_clas::Function,
                     m::Integer,d::Integer,k::Integer,
                     path::AbstractString,σ::Function=relu)
    f_sae = ()->SAE(m,d,σ)
    f_dict = ()->DictEnc(SparseDict(d,k),f_clas(),f_dec())
    M = Dict([(:sae,(f(SAE,m,d,σ) |> gpu,path*"/SAE")),
              (:psae, (f(PSAE,f_sae(),f_clas()) |> gpu,path*"/PSAE")),
              (:sparsedict, (f(DictEnc,m,d,k,σ) |> gpu,path*"/sparsedict")),
              (:distenc_eucl, (f(DistEnc,f_enc(),f_dec(),inveucl) |> gpu,path*"/distenc/eucl")),
              (:distenc_cossim, (f(DistEnc,f_enc(),f_dec(),cossim) |> gpu,path*"/distenc/cossim")),
              (:distpart_eucl, (f(DistPart,f_enc(),f_dec(),f_clas(),inveucl) |> gpu,path*"/distpart/eucl")),
              (:distpart_cossim, (f(DistPart,f_enc(),f_dec(),f_clas(),cossim) |> gpu,path*"/distpart/cossim"))])
    return M
end

function loadmodels(M::Dict)
    return map(m->loadmodel(m[1],m[2]),M)
end

    
function mlp(l,f::Function)
    θ = foldl(l[3:length(l)],
              init=Chain(Dense(l[1] => l[2],f))) do layers,d
        d_0 = size(layers[length(layers)].weight)[1]
        return Chain(layers...,Dense(d_0 => d,f))
    end
end

function mlp4x(m::Integer,d::Integer,l::Integer,σ=σ)
    n = maximum([m,d])
    return Chain(Dense(m => 4 * n),
                 map(_->Dense(4 * n => 4*n, σ),1:l)...,
                 Dense(4*n => d))
end
