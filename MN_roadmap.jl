cd(@__DIR__); include("helper.jl")
using MultiscaleGraphSignalTransforms, Plots, LightGraphs, JLD2
G = JLD2.load("datasets/MN_MutGauss.jld2", "G")["G"]

# N = 2640, M = 3302
