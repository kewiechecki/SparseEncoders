using RCall

@rimport stats
@rimport ComplexHeatmap as CH
@rimport circlize
@rimport grDevices as grD
@rimport grid as rgrid

function colorfn(M::AbstractArray,col::String,quant=0.99)
    return circlize.colorRamp2([0,stats.quantile(M,quant)[1]],
                               ["white",col])
end

function colorfn(M::AbstractArray,col::Tuple{String,String},quant=0.99)
    return circlize.colorRamp2([stats.quantile(M,1-quant)[1],0,
                                stats.quantile(M,quant)[1]],
                               [col[1],"white",col[2]])
end

function KEheatmap(out::String,
                   layout::AbstractMatrix,
                   hmvals::Dict,
                   cols::Dict,colorkey::Dict,lgd::Dict,
                   cclust::AbstractVector{Symbol},rclust::AbstractVector{Symbol},
                   lab_c::AbstractVector{String},
                   lab_r::AbstractVector{String})
    rowsp = mapreduce(x->rep(x[1],x[2]),
                      vcat,
                      zip(lab_r,map(i->size(hmvals[i])[2],layout[1,:])))
    colsp = mapreduce(x->rep(x[1],x[2]),
                      vcat,
                      zip(lab_c,map(i->size(hmvals[i])[1],layout[:,1])))
        
    colvals = maplab(cols) do key,col
        vals = (collect ∘ keys)(colorkey)[values(colorkey) .== key]
        return mapreduce(i->hcat(hmvals[i]...),hcat,vals)
    end
    
    colfns = mapkey(colorfn,colvals,cols)
    #colfns = mapkey((M,c)->circlize.colorRamp2([stats.quantile(M,0.01)[1],
    #                                            stats.quantile(M,0.99)[1]],
    #                                        ["white",c]),
    #                colvals,cols)

    f = (key,val)->colfns[colorkey[key]](val) |> rcopy
    hmfill = maplab(f,hmvals)

    hclust = map(stats.hclust ∘ stats.dist ∘ transpose,hmvals)
                 
    ord = map(x->rcopy(x[:order]),hclust)

    rsel = [foldl((x,i)->begin
        y = ord[i] .+ length(x)
        return cat(x,y,dims=1)
        end,rclust,init=[])...]

    csel = [foldl((x,i)->begin
        y = ord[i] .+ length(x)
        return cat(x,y,dims=1)
        end,cclust,init=[])...]

    mat = mapreduce(vcat,eachrow(layout)) do row
        hcat(select(hmvals,row)...)
    end
    colmat = mapreduce(vcat,eachrow(layout)) do row
        hcat(select(hmfill,row)...)
    end
    
    @rput mat
    @rput colmat
    @rput rsel
    @rput csel
    @rput rowsp
    @rput colsp
    @rput lgd
    @rput colfns
    
R"""
library(ComplexHeatmap)
library(circlize)
cellfn <- function(j,i,x,y,width,height,fill){
  grid.rect(x=x,y=y,height=height,width=width,gp = gpar(fill = colmat[i, j], col = NA))
}

lgd <- mapply(function(colfn,lab) Legend(col_fun=colfn,title=lab),colfns,lgd,SIMPLIFY=F)

hm <- Heatmap(mat,col=colmat,
                cell_fun=cellfn,
                split=rowsp,column_split=colsp,
                row_order=rsel, column_order=csel,
                cluster_rows=FALSE, cluster_columns=FALSE,
                cluster_row_slices=FALSE, cluster_column_slices=FALSE,
                show_heatmap_legend=FALSE,border=TRUE);
pdf($out)
draw(hm,annotation_legend_list=lgd)
dev.off()
"""
end
    
function KEheatmap(out::String,E::AbstractMatrix,metric::Function,Dlab::String)
    P = metric(E)
    D = metric(E')
    
    cols = Dict([(:E,"red"),
                 (:D,"darkgreen")])
    lgd = Dict([(:E,"embedding"),
                (:D,Dlab)])

    hmvals = Dict([(:P,P), (:E,E'), (:ET,E), (:D,D)])

    colorkey = Dict([(:P,:D), (:E,:E), (:ET,:E), (:D,:D)])

    layout = [:P :E; :ET :D]
    clust = [:P,:E]
    lab_c = ["(1) D","(2) E"]
    lab_r = ["(1) D","(2) E^T"]

    KEheatmap(out,layout,hmvals,cols,colorkey,lgd,clust,clust,lab_c,lab_r)
end


function KEheatmap(out::String,K::AbstractMatrix,E::AbstractMatrix)
    k,n = size(K)
    d,_ = size(E)
    
    P = pwak(K)
    Ksum = sum(K,dims=2)

    C = K * E' ./ Ksum

    Ê = (P * E')'
    Ĉ = K * Ê' ./ Ksum

    topleftfill = zeros(k,k)
    bottomrightfill = zeros(d,d)
    
    cols = Dict([(:E,"red"),
                (:K,"blue"),
                (:P,"black")])
    lgd = Dict([(:E,"embedding"),
                (:K,"P(cluster)"),
                (:P,"pairwise weight")])


    hmvals = Dict([(:topleftfill,topleftfill),
                (:K,K),
                (:C,C),
                (:KT,K'),
                (:P,P),
                (:E,E'),
                (:Ĉ,Ĉ'),
                (:Ê,Ê),
                (:bottomrightfill,bottomrightfill)])

    colorkey = Dict([(:topleftfill,:K),
                (:K,:K),
                (:C,:E),
                (:KT,:K),
                (:P,:P),
                (:E,:E),
                (:Ĉ,:E),
                (:Ê,:E),
                (:bottomrightfill,:E)])

    layout = [:topleftfill :K :C;
            :KT :P :E;
            :Ĉ :Ê :bottomrightfill]
    clust = [:KT,:P,:E]
    lab_c = ["(1) K^T","(2) PWAK(K)","(3) KE^T"]
    lab_r = ["(1) K","(2) PWAK(K)","(3) KÊ^T"]

    KEheatmap(out,layout,hmvals,cols,colorkey,lgd,clust,clust,lab_c,lab_r)
end

function KEheatmap(out::String,
                   K::AbstractMatrix,
                   E::AbstractMatrix,
                   P::AbstractMatrix,
                   Ê::AbstractMatrix,
                   C::AbstractMatrix,
                   Ĉ::AbstractMatrix,
                   metric::Function,
                   Plab::String,
                   Dlab::String,
                   lab_c::AbstractArray{String},
                   lab_r::AbstractArray{String})
    D_C = metric(C')
    D_E = metric(E')
    
    cols = Dict([(:E,"red"),
                 (:K,"blue"),
                 (:P,"black"),
                 (:D,"darkgreen")])
    lgd = Dict([(:E,"embedding"),
                (:K,"P(cluster)"),
                (:P,Plab),
                (:D,Dlab)])

    hmvals = Dict([(:D_C,D_C),
                (:K,K),
                (:C,C),
                (:KT,K'),
                (:P,P),
                (:E,E'),
                (:Ĉ,Ĉ'),
                (:Ê,Ê),
                (:D_E,D_E)])

    colorkey = Dict([(:D_C,:D),
                (:K,:K),
                (:C,:E),
                (:KT,:K),
                (:P,:P),
                (:E,:E),
                (:Ĉ,:E),
                (:Ê,:E),
                (:D_E,:D)])

    layout = [:D_C :K :C;
            :KT :P :E;
            :Ĉ :Ê :D_E]

    clust = [:KT,:P,:E]

    KEheatmap(out,layout,hmvals,cols,colorkey,lgd,clust,clust,lab_c,lab_r)
end

function KEheatmap(out::String,
                   K::AbstractMatrix,
                   E::AbstractMatrix,
                   P::AbstractMatrix,
                   C::AbstractMatrix,
                   metric::Function,
                   Plab::String,
                   Dlab::String)
    Ksum = sum(K,dims=2)

    C = K * E' ./ Ksum

    Ê = (P * E')'
    Ĉ = K * Ê' ./ Ksum

    lab_c = ["(1) K^T","(2) PWAK(K)","(3) KE^T"]
    lab_r = ["(1) K","(2) PWAK(K)","(3) KÊ^T"]
    KEheatmap(out,K,E,P,Ê,C,Ĉ,metric,Plab,Dlab,lab_c,lab_r)
end

function KEheatmap(out::String,
                   K::AbstractMatrix,
                   E::AbstractMatrix,
                   P::AbstractMatrix,
                   metric::Function,
                   Plab::String,
                   Dlab::String)
    Ksum = sum(K,dims=2)

    C = K * E' ./ Ksum

    Ê = (P * E')'
    Ĉ = K * Ê' ./ Ksum

    lab_c = ["(1) K^T","(2) PWAK(K)","(3) KE^T"]
    lab_r = ["(1) K","(2) PWAK(K)","(3) KÊ^T"]
    KEheatmap(out,K,E,P,Ê,C,Ĉ,metric,Plab,Dlab,lab_c,lab_r)
end

function KEheatmap(M::SAE,X::AbstractArray,path::String,file::String)
    E = encode(M,X)
    KEheatmap(path*"/"*file,E,cossim,"cosine similarity")
end

function KEheatmap(M::PSAE,X::AbstractArray,path::String,file::String)
    K = cluster(M,X)
    E = encode(M,X)
    P = pwak(K)
    KEheatmap(path*"/"*file,K,E,P,cossim,
              "pairwise weight","cosine similarity")
end

function KEheatmap(M::DistPart,X::AbstractArray,path::String,file::String)
    K = cluster(M,X)
    E = encode(M,X)
    P = kern(M,X)
    KEheatmap(path*"/"*file,K,E,P,cossim,
              "pairwise weight","cosine similarity")
end

function KEheatmap(M::DistEnc,X::AbstractArray,path::String,file::String)
    E = encode(M,X)
    P = kern(M,X)
    KEheatmap(path*"/"*file,E,P,cossim,
              "pairwise weight","cosine similarity")
end

function KEheatmap(M::DictEnc,X::AbstractArray,path::String,file::String)
    K = cluster(M,X)
    C = M.dict.dict'
    E = M.dict(K)
    P = cossim(E)

    D_C = cossim(C')
    D_E = cossim(E')
    
    cols = Dict([(:E,("blue","red")),
                 (:K,"blue"),
                 (:D,("darkgoldenrod","darkgreen"))])
    lgd = Dict([(:E,"embedding"),
                (:K,"P(cluster)"),
                (:D,"cosine similarity")])

    hmvals = Dict([(:D_C,D_C),
                (:K,K),
                (:C,C),
                (:KT,K'),
                (:P,P),
                (:E,E'),
                (:CT,C'),
                (:ET,E),
                (:D_E,D_E)])

    colorkey = Dict([(:D_C,:D),
                (:K,:K),
                (:C,:E),
                (:KT,:K),
                (:P,:D),
                (:E,:E),
                (:CT,:E),
                (:ET,:E),
                (:D_E,:D)])

    layout = [:D_C :K :C;
            :KT :P :E;
            :CT :ET :D_E]

    clust = [:CT,:P,:E]
    lab_c = ["(1) K^T","(2) CK","(3) C"]
    lab_r = ["(1) K","(2) CK","(3) C"]

    KEheatmap(path*"/"*file,
              layout,hmvals,
              cols,colorkey,lgd,
              clust,clust,lab_c,lab_r)
end

function KEheatmap(inner::Dict,X::AbstractArray)
    map(m->KEheatmap(m[1],X,m[3],"KE.pdf"),inner)
end

function KEheatmap(M::DistEnc,X::AbstractArray,path::String,file::String)
    E = encode(M,X)
    P = kern(M,X)
    Ê = (P * E')'

    D = cossim(E')
    
    cols = Dict([(:E,"red"),
                 (:P,"black"),
                 (:D,"darkgreen")])
    lgd = Dict([(:E,"embedding"),
                (:P,"pairwise weight"),
                (:D,"cosine similarity")])

    hmvals = Dict([(:P,P),
                   (:E,E'),
                   (:Ê,Ê),
                   (:D,D)])

    colorkey = Dict([(:P,:P),
                     (:E,:E),
                     (:Ê,:E),
                     (:D,:D)])

    layout = [:P :E;
              :Ê :D]

    clust = [:P,:E]
    lab_c = ["(1) D","(2) E"]
    lab_r = ["(1) D","(3) Ê"]

    KEheatmap(path*"/"*file,
              layout,hmvals,
              cols,colorkey,lgd,
              clust,clust,lab_c,lab_r)
end
