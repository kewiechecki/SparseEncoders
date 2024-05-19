using Flux, Functors, CUDA, LinearAlgebra, ProgressMeter, MLDatasets, JLD2, CSV, MLDatasets, StatsPlots, OneHotArrays, Images, Dates, Distributions, DataFrames
using Tables
using RCall

using Flux: logitcrossentropy
using Plots.PlotMeasures
import Base.map
import ProgressMeter.update!

include("SAE.jl")
include("PSAE.jl")
include("EncoderBlock.jl")

include("distfns.jl")
include("lossfns.jl")
include("DistEnc.jl")
include("SparseDict.jl")

include("trainingfns.jl")
include("models.jl")
include("auxfns.jl")

include("loadmodels.jl")

include("featurevis.jl")
include("plotfns.jl")

include("Rmacros.jl")
include("KEheatmap.jl")

include("defaults.jl")
