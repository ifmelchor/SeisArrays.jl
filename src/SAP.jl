#!/usr/local/bin julia
# coding=utf-8

__precompile__()

module SAP

    using LinearAlgebra
    using Statistics
    using Contour

    abstract type AbstractBase end

    function _ccmap! end
    function rolling_bandpower end
    
    export rolling_bandpower
    export zlcc, zlcc_stack, array_transfunc
    export AbstractBase, BaseCuda

    include("types.jl")

    include("utils.jl")

    include("array_resp.jl")

    include("cc8mre.jl")

end
