using Flux,Functors
import Base.size
import Base.mapreduce

struct EncoderBlock
    heads::Parallel
end
@functor EncoderBlock

function EncoderBlock(f::Union{Function,DataType},h::Integer,connection,args...;kwargs...)
    layers = Parallel(connection,map(_->f(args...;kwargs...),1:h)...)
    return EncoderBlock(layers)
end

function (M::EncoderBlock)(x)
    return M.heads(x)
end

function heads(M::EncoderBlock)
    return M.heads.layers
end

function size(M::EncoderBlock)
    return length(M.heads.layers)
end

function conn(M::EncoderBlock)
    return M.heads.connection
end

function map(f::Function,M::EncoderBlock,args...;kwargs...)
    return map(m->f(m,args...;kwargs...),heads(M))
end

function mapreduce(f::Function,g::Function,M::EncoderBlock,
                   args...;kwargs...)
    return reduce(g,map(f,M,args...;kwargs...))
end

function mapreduce(f::Function,M::EncoderBlock,args...;kwargs...)
    return reduce(conn(M),map(f,M,args...;kwargs...))
end

function encode(M::EncoderBlock,x)
    return mapreduce(encode,vcat,M,x)
end

function diffuse(M::EncoderBlock,x)
    return encode(M,x)
end

function L1(M::EncoderBlock,α,x)
    c = diffuse(M,x)
    return α * L1(c)
end

function L1_normcos(M::EncoderBlock,α,x)
    c = diffuse(M,x)
    return α * L1_normcos(c)
end
