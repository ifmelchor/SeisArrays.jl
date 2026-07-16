#!/usr/local/bin julia
# coding=utf-8

module SeisArrays
    
    using LinearAlgebra
    using Statistics
    using FFTW
    using Contour
    using Interpolations

    export SeisArray2D, TriangleDef

    # GCC (Generalized Cross-Correlation)
    export init_wsgcc, gcc_delay, delay_matrix_gcc
    
    # Metodos de analisis de array
    export zlcc
    export init_triads, trias
    
    # Archivos Base
    include("types.jl")
    include("topology.jl")
    include("slowness_grids.jl")

    # Utilidades
    include("signals.jl")
    include("filter.jl")
    # include("cf.jl") 

    # Motores de Correlación y Delays
    include("gcc.jl")
    # include("lcc.jl")

    # Zero Lag Cross Correlation
    include("zlcc.jl")

    # TRIad-based Adaptive Slowness
    include("trias.jl")

end
