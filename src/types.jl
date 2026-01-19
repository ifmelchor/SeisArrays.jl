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


struct ZLCC_WS_CPU{T<:Real}
    data::Matrix{T}
    dx::Vector{T}
    dy::Vector{T}
    citer::Vector{Tuple{Int, Int}}
    
    lwin::Int
    nsta::Int
    slomax_sq::T

    # Grillas
    sx::Vector{T}
    sy::Vector{T}
    sx2::Vector{T}
    sy2::Vector{T}
    
    # Mapas
    ccmap::Matrix{T}
    ccmap2::Matrix{T}
    
    # Buffers de trabajo
    energy_bufs::Vector{Vector{T}}
    beam::Vector{T}
end


struct FSGCC_ws{P1, P2}
    # Parámetros
    fs::Float64        # Sampling rate
    n::Int             # Tamaño de la señal
    n_fft::Int         # Tamaño con Zero-Padding
    n_up::Int          # Tamaño interpolado
    B::Int
    n_gamma::Float64
    
    # Buffers temporales
    buf_s1::Vector{Float64}
    buf_s2::Vector{Float64}
    window::Vector{Float64} # Ventana de Hann pre-calculada
    
    # Buffers Frecuenciales
    n_freq::Int
    S1::Vector{ComplexF64}
    S2::Vector{ComplexF64}
    G12::Vector{ComplexF64}
    G12_smooth::Vector{ComplexF64}
    G11_smooth::Vector{Float64}
    G22_smooth::Vector{Float64}
    
    # mascara de banda de frequencia
    mask::Vector{Float64}
    
    # Buffer de interpolación (Upsampling)
    n_freq_up::Int
    C_up_freq::Vector{ComplexF64}
    C_up_time::Vector{Float64} # Resultado final en tiempo
    
    # Planes de FFT (Lo más costoso de crear)
    plan_fwd::P1
    plan_inv::P2
end


struct ValidTrios{T<:AbstractFloat}
    x1::Vector{T}
    y1::Vector{T}
    x2::Vector{T}
    y2::Vector{T}
    x3::Vector{T}
    y3::Vector{T}
    dt1::Vector{T}
    dt2::Vector{T}
    dt3::Vector{T}
end


struct ThreadBuffers{T<:AbstractFloat}
    fsgcc_ws    :: FSGCC_ws 
    dt          :: Vector{T}
    cc          :: Vector{T}
    trio_flags  :: BitVector
    trio_error  :: Vector{T}
    trio_cc_avg :: Vector{T}
    trio_w      :: Vector{T}
    vt          :: ValidTrios
    like_map    :: Matrix{T}
end


struct TriangleDef
    # indicides de los pares
    p1_idx::Int
    p2_idx::Int
    p3_idx::Int

    # signo del delay
    s1::Float64
    s2::Float64
    s3::Float64

    sta_triad::Tuple{Int, Int, Int}
end
