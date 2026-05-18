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



struct ZLCC_WS_CPU{T<:AbstractFloat, R<:AbstractRange{T}}
    data::Matrix{T}
    dx::Vector{T}
    dy::Vector{T}
    citer::Vector{Tuple{Int, Int}}
    
    lwin::Int
    nsta::Int
    slomax2::T

    # Grillas
    s_grid::R
    s_grid_c::R
    s_grid_f::R

    # tamaños
    nite :: Int
    nite_c :: Int
    nite_f :: Int
    
    # Mapas
    ccmap::Matrix{T}
    ccmap_c::Matrix{T}
    ccmap_f::Matrix{T}
    
    # Buffers de trabajo
    benergy::Vector{T}
    beam::Vector{T}
    taper::Vector{Float64}
    fft_buf::Vector{ComplexF64}
end
