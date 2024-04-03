include("Rmacros.jl")

function rplot(path::String,f::RObject,args...;kwargs...)
    return function g(args...;kwargs...)
        grD.pdf(path*".pdf")
        f(args...;kwargs...)
        grD.dev_off()
    end
end

function rplot(path::String,file::String,f::RObject,args...;kwargs...)
    mkpath(path)
    return rplot(path*"/"*file,f,args...;kwargs...)
end

macro rplot(path,expr) 
    return quote
            $(esc(expr))
    end
end

        

function heatmap(path,kwargs...)
    rplot(path,draw)(CH.Heatmap(kwargs...))
end

function channelhm(X::AbstractArray{<:Any,3})
    m,n,c = size(X)
    sp = mapreduce(i->rep("channel"*string(i),n),vcat,1:c)
    X = reshape(X,m,n*c)
    hm = CH.Heatmap(X',"embedding",col=["white","red"],
                    split=sp,
                    cluster_columns=false,cluster_rows=false,
                    height=rgrid.unit(n*c,"mm"),
                    width=rgrid.unit(m,"mm"));
    CH.draw(hm)
end

function channelhm(X,path::String)
    m,n,c = size(X)
    sp = mapreduce(i->rep("channel"*string(i),vcat,n),1:c)
    X = reshape(X,m,n*c)
    hm = CH.Heatmap(X',"embedding",col=["white","red"],
                    split=sp,
                    cluster_columns=false,cluster_rows=false)
    rplot(path,draw)(hm)
end

function channelhm(X::AbstractArray{<:Any,3},path::String,file::String)
    mkpath(path)
    m,n,c = size(X)
    sp = mapreduce(i->rep("channel"*string(i),vcat,n),1:c)
    X = reshape(X,m,n*c)
    hm = CH.Heatmap(X',"embedding",col=["white","red"],
                    split=sp,
                    cluster_columns=false,cluster_rows=false)
    rplot(path,file,draw)(hm)
end
function Kheatmap(M,x,y) 
    ŷ = M(x)
    labels = string.(unhot(cpu(y)))[1,:]
    return CH.Heatmap(ŷ',"P(label)",col=["white","blue"],
                      split=labels,border=true)
end

function Eheatmap(M,x,y)
    ŷ = M(x)
    labels = string.(unhot(cpu(y)))[1,:]
    return CH.Heatmap(ŷ',"P(label)",col=["white","red"],
                      split=labels,border=true)
end

function drawheatmap(H,out)
    grD.pdf(out)
    CH.draw(H)
    grD.dev_off()
end

function plotlogits(M,x,y,out)
    ŷ = M(x)
    labels = string.(unhot(cpu(y)))[1,:]

    grD.pdf(out)
    CH.Heatmap(ŷ',col=["white","blue"],"P(label)",
            split=labels,border=true)
    grD.dev_off()
end

function covheatmap(X,lab,out)
    Y = cov(X')
    grD.pdf(out)
    CH.Heatmap(Y,
               cluster_columns=false,cluster_rows=false)
    grD.dev_off()
end
