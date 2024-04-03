using ProgressMeter, MLDatasets, OneHotArrays,
    JLD2,Tables,CSV

function update!(M,x,y,
                 loss::Function,
                 opt::Flux.Optimiser)
    #x = gpu(x)
    #y = gpu(y)
    f = m->loss(m(x),y)
    state = Flux.setup(opt,M)
    l,∇ = Flux.withgradient(f,M)
    Flux.update!(state,M,∇[1])
    return l
end

function update!(M,
                 loss::Function,
                 opt::Flux.Optimiser)
    state = Flux.setup(opt,M)
    l,∇ = Flux.withgradient(loss,M)
    Flux.update!(state,M,∇[1])
    return l
end

function plotloss(L::AbstractArray,ylab::AbstractString,path::AbstractString)
    p = scatter(1:length(L),L,xlabel="batch",ylabel=ylab,label=nothing,
                markershape=:cross,markersize=0.5,margin=10mm)
    savefig(p,path)
end

function plotloss(L::AbstractArray,ylab::AbstractString,path::AbstractString,file::AbstractString)
    p = scatter(1:length(L),L,xlabel="batch",ylabel=ylab,label=nothing,
                markershape=:cross,markersize=0.5,margin=10mm)
    savepath(p,path,file)
end

function writecsv(log::AbstractArray,path::AbstractString)
    Tables.table(log) |> CSV.write(path)
end

function writecsv(log::AbstractArray,path::AbstractString,file::AbstractString)
    mkpath(path)
    Tables.table(log) |> CSV.write(path*"/"*file)
end

function savemodel(M,path)::Nothing
    state = Flux.state(M) |> cpu;
    jldsave(path*".jld2";state)
end

function savemodel(M,path,file)::Nothing
    mkpath(path)
    state = Flux.state(M) |> cpu;
    jldsave(path*"/"*file*".jld2";state)
end

function savemodel(M,log,path,file)::Nothing
    mkpath(path)
    state = Flux.state(M) |> cpu;
    jldsave(path*"/"*file*".jld2";state)
    Tables.table(log) |> CSV.write(path*"/"*file*"_loss.csv")
end

function train!(M,
                loader::Flux.DataLoader,
                opt::Flux.Optimiser,
                epochs::Integer,
                loss::Function,
                log;
                prefn = identity,
                postfn=identity,
                ignoreY=false,
                savecheckpts=false,
                path="")
    if length(path) > 0
        mkpath(path)
    end
    #f = (E,y)->loss(postfn(E),y)
    @showprogress map(1:epochs) do i
        map(loader) do (x,y)
            x = gpu(x)
            y = gpu(y)
            if ignoreY
                y = x
            end
            #E = prefn(x)
            #l = update!(M,E,y,f,opt)
            f = M->loss((postfn ∘ M ∘ prefn)(x),y)
            l = update!(M,f,opt)
            push!(log,l)
        end
        if savecheckpts
            savemodel(M,path*"/"*string(i))
        end
    end
    if length(path) > 0
        savemodel(M,path*"/final")
        Tables.table(log) |> CSV.write(path*"/loss.csv")
        plotloss(log,"loss",path*"/loss.pdf")
    end
end

function train!(M,
                loader::Flux.DataLoader,
                opt::Flux.Optimiser,
                epochs::Integer,
                loss::Function;
                kwargs...)
    log = []
    train!(M,loader,opt,epochs,loss,log;kwargs...)
    return log
end

function train!(M,path::String,
                loss::Function,
                loader::Flux.DataLoader,
                opt::Flux.Optimiser,
                epochs::Integer;
                savecheckpts=true)
    if length(path) > 0
        mkpath(path)
    end
    log = []
    @showprogress map(1:epochs) do i
        map(loader) do (x,y)
            x = gpu(x)
            y = gpu(y)
            f = m->loss(m,x,y)
            l = update!(M,f,opt)
            push!(log,[l...]')
        end
        if savecheckpts
            savemodel(M,path*"/"*string(i))
        end
    end
    log = vcat(log...)
    if length(path) > 0
        savemodel(M,path*"/final")
        #Tables.table(vcat(log...)) |> CSV.write(path*"/loss.csv")
        writecsv(log,path*"/loss.csv")
#        plotloss(log[:,1],"loss",path*"/loss.pdf")
    end
    return log
end

