#!/usr/local/bin julia
# coding=utf-8


function _biquad!(x, y, coef)
    ndata = length(x)
    
    y[1] = coef[1]*x[1]
    y[2] = coef[1]*x[2] + coef[2]*x[1] + coef[4]*y[1]
    
    @inbounds for j in 3:ndata
        y[j] = coef[1]*x[j] + coef[2]*x[j-1] + coef[3]*x[j-2] + coef[4]*y[j-1] + coef[5]*y[j-2]
    end
end


function _biquad_coef(amort, fc, fs, lowpass)
    a = tan(pi * fc / fs)
    d = 1 + 2 * amort * a + a * a
    a0 = lowpass ? (a * a / d) : (1 / d)
    a1 = lowpass ? (2 * a0) : (-2 * a0)
    a2 = a0
    b1 = -(2 * a * a - 2) / d
    b2 = -(1 - 2 * amort * a + a * a) / d
    return (a0, a1, a2, b1, b2)
end


function filter!(S::AbstractSeisArray{T}, fmin::Real, fmax::Real) where T

    ntime, _ = size(S.data)
    
    temp_buf = Vector{T}(undef, ntime)

    _filter!(S.data, S.fs, T(fmin), T(fmax), temp_buf)

    return S 
end


function _filter!(data::AbstractMatrix{T}, fs::Real, fmin::Real, fmax::Real, temp_buf::AbstractVector{T}; filter_type::Symbol=:butt) where T
  
    ntime, nsta = size(data)

    if filter_type == :butt
        coef_h = T.(_biquad_coef(1/sqrt(2), fmax, fs, true))
        coef_l = T.(_biquad_coef(1/sqrt(2), fmin, fs, false))
    
    elseif filter_type == :fb2
        coef_h = T.(_biquad_coef(0.47, fmax, fs, true))
        coef_l = T.(_biquad_coef(0.47, fmin, fs, false))
    
    else
        error("Filtro no reconocido: :$filter_type. Usa :butt o :fb2")
    end

    @inbounds @views for i in 1:nsta
        col = data[:, i]
        # Forward
        _biquad!(col, temp_buf, coef_h)
        _biquad!(temp_buf, col, coef_l)
        # reverse
        reverse!(col)
        # backward
        _biquad!(col, temp_buf, coef_h)
        _biquad!(temp_buf, col, coef_l)
        # reverse
        reverse!(col)
    end
end


function mbf_filter!(U::AbstractArray{T, 3}, data::AbstractMatrix{T}, fs::Real, fmin::Real, fmax::Real, f_central::AbstractVector{T}, temp_buf::AbstractVector{T}) where T<:Real
    
    nband = length(f_central)
    
    @inbounds for b in 1:nband
        # Frecuencias de corte
        if b == 1
            f_low = fmin / sqrt(f_central[2]/fmin)
        else
            f_low = sqrt(f_central[b-1] * f_central[b])
        end

        if b == nband
            f_high = fmax * sqrt(fmax/f_central[end-1])
        else
            f_high = sqrt(f_central[b] * f_central[b+1])
        end
        
        # Crear una copia de los datos originales
        U_slice = @view U[:, :, b]
        copyto!(U_slice, data)
        
        # (forward-backward)
       _filter!(U_slice, fs, T(f_low), T(f_high), temp_buf)
    end
    
    return U
end


function mbf_filter(data::AbstractMatrix{T}, fs::Real, fmin::Real, fmax::Real, nband::Int) where T<:Real
    
    ntime, nsta = size(data)
    
    f_central = T.(exp10.(range(log10(fmin), log10(fmax), length=nband)))
    
    U = Array{T, 3}(undef, ntime, nsta, nband)
    
    temp_buf = Vector{T}(undef, ntime)
    
    mbf_filter!(U, data, fs, fmin, fmax, f_central, temp_buf)
    
    return U, f_central
end


function _gaussian_recursive_filter!(x::AbstractVector{T}, sigma::Real) where T
    
    N = length(x)

    # Coeficientes Young & van Vliet (1995)
    q = sigma > 0.5 ? 0.98711*sigma - 0.96330 : 
        3.97156 - 4.14554 * sqrt(1.0 - 0.26891*sigma)
    
    b0 = 1.57825 + 2.44413*q + 1.4281*q^2 + 0.422205*q^3
    b1 = 2.44413*q + 2.85619*q^2 + 1.26661*q^3
    b2 = -(1.4281*q^2 + 1.26661*q^3)
    b3 = 0.422205*q^3

    B0 = 1.0 - (b1 + b2 + b3)/b0
    A1 = -b1/b0; A2 = -b2/b0; A3 = -b3/b0

    @inbounds begin
        # === PASO CAUSAL (1 → N) ===
        x[1] = B0 * x[1]
        N >= 2 && (x[2] = B0*x[2] + A1*x[1])
        N >= 3 && (x[3] = B0*x[3] + A1*x[2] + A2*x[1])
        for i in 4:N
            x[i] = B0*x[i] + A1*x[i-1] + A2*x[i-2] + A3*x[i-3]
        end

        # === PASO ANTICAUSAL (N → 1) ===
        x[N] = B0 * x[N]
        N >= 2 && (x[N-1] = B0*x[N-1] + A1*x[N])
        N >= 3 && (x[N-2] = B0*x[N-2] + A1*x[N-1] + A2*x[N])
        for i in N-3:-1:1
            x[i] = B0*x[i] + A1*x[i+1] + A2*x[i+2] + A3*x[i+3]
        end
    end
end