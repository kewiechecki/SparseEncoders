using RCall

@rimport base
@rimport stats
@rimport ComplexHeatmap as CH
@rimport circlize
@rimport grDevices as grD
@rimport grid as rgrid

macro rsource()
    R"""
    f <- list.files("R","\\.R$",full.names=T)
    sapply(f,source)
    """
end

@rsource

function rpdf(file)
    R"""
pdf($file)
    """
end

function rdevoff()
    R"""
dev.off()
"""
end

macro rpdf(file, expr)
    rpdf(file)
    expr
    rdevoff()
end

macro leiden(G)
    q = R"""
    library(leiden)
    leiden($G)
    """
    return rcopy(q) |> Array{Int}
end

macro Heatmap(args,out)
    rpdf(out)
    R"""
    draw(do.call(Heatmap,$(args)))
    """
    rdevoff()
end

macro HeatmapScale(breaks,cols,args,out)
    return quote
        breaks = $(esc(breaks))
        cols = $(esc(cols))
        args = $(esc(args))
        out = $(esc(out))
        R"""
        library(ComplexHeatmap)
        library(circlize)
        col <- colorRamp2($(breaks),$(cols))
        args <- $(args)
        args$col <- col

        pdf($(out))
        draw(do.call(Heatmap,args))
        dev.off()
        """
    end
end


macro Rfn(f,argnames,args)
    R"""
    argnames <- names(formals(args($(f))))
    """
    @rget argnames
    return quote
        f = $(esc(f))
        argnames = $(esc(argnames))
        args = $(esc(args))
        R"""
        args <- $(args)
        argnames <- $(argnames)
        names(args) <- argnames
        do.call($f,args)

        """
    end 
end

function clusthyper(out,cond,clust;kwargs...)
    args = [cond,clust,out]
    rsendargs(args,kwargs...)
    R"""
args[[1]] <- as.data.frame(args[[1]])
do.call(clusthyper,args)
    """
end

function rsendargs(args;kwargs...)
    args = map(1:length(args)) do i
        ("arg"*string(i),args[i])
    end |> Dict

    kwargs = Dict([kwargs...])
    @rput args
    @rput kwargs
    R"""
names(args) <- NULL
args <- append(args,kwargs)
    """
end

function heatmap(out, dat, mid; kwargs...)
    rsendargs([dat],kwargs...)
    R"""
args$col <- col.z(args[[1]],mid=$mid)
pdf($(out))
draw(do.call(Heatmap,args))
dev.off()
    """
end

function heatmap(out, X; kwargs...)
    rsendargs([X]; kwargs...)

    if minimum(X) >= 0
        mid = (median ∘ not0)(X)
    else
        mid = 0
    end
    @rput mid

    rpdf(out)
    R"""
args$col <- col.z(args[[1]],mid=$mid)
draw(do.call(Heatmap,args))
    """
    rdevoff()
end

function combinedheatmap(out,dict;kwargs...)
    mat = vcat(values(dict)...)
    colsp = mapreduce(vcat,zip(values(dict),keys(dict))) do (M,lab)
        m,n = size(M)
        return rep(lab,m)
    end
    heatmap(out*".pdf",mat',column_split = colsp;kwargs...)
end

function batchhyper(dict)
    @rput dict
    R"""
batchsize <- length(dict[[1]])
hyper <- sapply(dict,function(cond) lapply(dict, function(clust) condhyper(1:batchsize,cond,clust)))
xlab <- sapply(hyper[,1],function(x) colnames(x$log2OR))
ylab <- sapply(hyper[1,],function(x) row.names(x$log2OR))
    """
    @rget hyper
    @rget xlab
    @rget ylab
    return hyper,xlab,ylab
end


function combinedhyper(out,dict;kwargs...)

    hyper,xlab,ylab = batchhyper(dict)
    xlab = vcat(values(xlab)...)
    ylab = vcat(values(ylab)...)
                
    clust = dictcat(hyper)
    sp = vcat(repkey_clust(dict)...)
    
    rsendargs([clust[:log2OR],clust[:FDR],clust[:q]],
              split = sp, column_split=sp,border = true,
              #column_title = "cluster",row_title = "enriched in";
              kwargs...)
    @rput xlab
    @rput ylab
    R"""
row.names(args[[1]]) <- ylab
colnames(args[[1]]) <- xlab
pdf($out)
do.call(dotplot,args)
dev.off()
    """
end
    

function rcall(f,args,kwargs...)
    rsendargs(args, kwargs...)
    R"""
res <- do.call($f,args)
    """
    @rget res
    return res
end
