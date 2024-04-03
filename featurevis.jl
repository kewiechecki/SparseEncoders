function date() :: String
    Dates.format(today(),"yyyy-mm-dd")*"/"
end

function savepath(path::String,file::String,args...;kwargs...) :: Nothing
    mkpath(path)
    save(path*"/"*file,args...;kwargs...)
end

function savepath(p::Plots.Plot,file::String)
    savefig(p,file)
end

function savepath(p::Plots.Plot,path::String,file::String)
    mkpath(path)
    savepath(p,path*"/"*file)
end

function savedate(path::String,args...;kwargs...) :: Nothing
    savepath(date()*"/"*path,args...;kwargs...)
end

function grayimg(x::AbstractArray{<:AbstractFloat,2})
    #x = x .- minimum(x)
    #x = x ./ maximum(x)
    x[x .> 1] .= 1
    colorview(Gray,x[:,:]')
end

function grayimg(x::AbstractArray{<:AbstractFloat,3})
    grayimg(x[:,:,1])
end

function imgbatch(x::AbstractArray{<:AbstractFloat},f::Function=grayimg)
    x = cpu(x)
    f(hcat(eachslice(x,dims=(length ∘ size)(x))...))
end

function imgbatch(M,X::AbstractArray{<:AbstractFloat},
                     f::Function=imgbatch)
    Y = M(X)
    return f(vcat(X,Y))
end

function imgbatch(outer,sae::SparseEncoder,psae::SparseEncoder,X::AbstractArray{<:AbstractFloat},
                  f::Function=grayimg)
    E = outer[:encoder](X)
    X_outer = outer[:decoder](E)
    X_SAE = outer[:decoder](sae(E))
    X_PSAE = outer[:decoder](psae(E))
    img = vcat(X,X_outer,X_SAE,X_PSAE)
    #img = hcat(eachslice(vcat(X,X_outer,X_SAE,X_PSAE),dims=4)...)
    return imgbatch(img)
end

function imgbatch(outer,inner::AbstractVector{<:SparseEncoder},X::AbstractArray{<:AbstractFloat},
                  f::Function=grayimg)
    E = outer[:encoder](X)
    X_outer = outer[:decoder](E)
    X_inner = map(M->outer[:decoder](M(E)),inner)
    img = vcat(X,X_outer,vcat(X_inner...)) |> cpu
    #img = hcat(eachslice(vcat(X,X_outer,X_SAE,X_PSAE),dims=4)...)
    return imgbatch(img)
end

function writeimg(x,path::String,f::Function=grayimg) :: Nothing
    save(path,f(x))
end

function writeimg(x,path::String,file::String,f::Function=grayimg)
    savepath(path,file,f(x))
end

function mappath(f::Function,X,path::String,args...;kwargs...)
    i = 1
    map(X) do x
        f(x,path*string(i),args...;kwargs...)
        i = i + 1;
    end
end

function writeimgs(X,path::String,dims::Integer=4,f::Function=grayimg,format::String="pdf")
    g = (x,path)->writeimg(x,path*"."*format,f)
    mappath(g,eachslice(X,dims=dims),path)
end


function layerimgs(X::AbstractArray{<:AbstractFloat,4},path::String,
                   f::Function=grayimg,format::String="pdf")
    mappath(eachslice(X,dims=4),path) do x,p
        p = p*"/"
        writeimgs(x,p,3,f,format)
    end
end

function trackimg(M,X,path,f::Function=grayimg,format::String="pdf")
    i = 0
    foldr(M.layers,X) do θ,y
        i = i + 1
        z = θ(y)
        layerimgs(z,path*"/layer"*string(i),f,format)f
        return z
    end
end

function centroidimg(M::SparseClassifier,
                     X::AbstractArray,
                     decoder::Chain,
                     f::Function=grayimg)
    C = centroid(M,X)
    return imgbatch(decoder(decode(M,C)),f)
end

function centroidimg(M::Dict,
                     X::AbstractArray,
                     decoder::Chain,
                     f::Function=grayimg)
    map(m->save(m[3]*"/centroid.pdf",
                centroidimg(m[1],X,decoder,f)), M)
end

function featimg(i::Integer,M::SparseEncoder,E::AbstractArray,
                 decoder::Chain,
                 maskfn::Function)
    E = maskfn(E,i)
    X = decode(M,E)
    return decoder(X)
end

function featimg(M::SparseEncoder,X::AbstractArray,
                 decoder::Chain,
                 maskfn::Function)
    E = diffuse(M,X)
    m,n = size(E)
    return mapreduce(i->featimg(i,M,E,decoder,maskfn),vcat,1:m)
end

function featimg(path::String,maskfn::Function,
                 M::SparseEncoder,X::AbstractArray,
                 decoder::Chain,
                 f::Function=grayimg)
    save(path,imgbatch(featimg(M,X,decoder,maskfn),f))
end

function featimg(path::String,
                 M::SparseEncoder,
                 X::AbstractArray,
                 decoder::Chain,
                 f::Function=grayimg)
    files = Dict([("zeroabl",zeroabl),
                  ("meanabl",meanabl),
                  ("zeroabl_inv",(E,i)->zeroabl(E,i,true)),
                  ("meanabl_inv",(E,i)->meanabl(E,i,true))])

    maplab(files) do file,maskfn
        featimg(path*"/"*file*".pdf",
                maskfn,M,X,decoder,f)
    end
end

function featimg(M::Dict,X::AbstractArray,decoder::Chain,
                 f::Function=grayimg)
    map(m->featimg(m[3],m[1],X,decoder,f),M)
end
