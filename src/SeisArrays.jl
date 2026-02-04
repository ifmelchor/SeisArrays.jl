#!/usr/local/bin julia
# coding=utf-8

module SeisArrays
    
    using LinearAlgebra
    using Statistics
    using FFTW
    using Contour
    using Interpolations

    export SeisArray2D
    export zlcc
    export trias
    
    include("types.jl")
    include("utils.jl")
    include("signals.jl")
    include("contours.jl")

    # Zero Lag Cross Correlation
    include("zlcc.jl")

    # TRIad-based Adaptive Slowness
    include("trias.jl")

end
