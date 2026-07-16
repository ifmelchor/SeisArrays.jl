#!/usr/local/bin julia
# coding=utf-8

abstract type AbstractSeisArray{T} end


struct SeisArray2D{T<:Real} <: AbstractSeisArray{T}
    xcoord::Vector{T} # coordenadas UTM (eastern) en km
    ycoord::Vector{T} # coordenadas UTM (northing) en km
    
    # Datos (Muestras x Sensores)
    data::Matrix{T}

    # frecuencia de muestreo
    fs::Float64

    function SeisArray2D(x::Vector{T}, y::Vector{T}, data::Matrix{T}, fs::Real) where T

        if length(x) != length(y)
            error("Las coordenadas X e Y deben tener la misma longitud.")
        end

        n_sensors = length(x)
        if size(data, 2) != n_sensors
            error("La matriz de datos tiene $(size(data, 2)) columnas, pero hay $n_sensors coordenadas.")
        end

        new{T}(x, y, data, Float64(fs))
    end
end


struct TriangleDef{T<:AbstractFloat}
    i::Int          # i < j < k por construcción
    j::Int
    k::Int
    # métricas geométricas precalculadas
    dmin::T
    dmax::T
    dx_ij::T
    dx_jk::T
    dx_ki::T
    dy_ij::T
    dy_jk::T
    dy_ki::T
end


