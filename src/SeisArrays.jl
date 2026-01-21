#!/usr/local/bin julia
# coding=utf-8

module SeisArrays
    
    using LinearAlgebra
    using Statistics
    using FFTW
    using Contour

    export SeisArray2D
    export zlcc
    export tcwals
    
    include("types.jl")
    include("utils.jl")
    include("signals.jl")
    include("contours.jl")

    # Zero Lag Cross Correlation
    include("zlcc.jl")

    # Time-Closure Weighted Adaptive Likelihood Slowness
    include("tcwals.jl")

end
