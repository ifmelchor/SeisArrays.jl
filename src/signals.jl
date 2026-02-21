#!/usr/local/bin julia
# coding=utf-8


"""
    Filter signal
"""
function _fb2_inplace!(input, output, coef)
    ndata = length(input)
    output[1], output[2] = input[1], input[2]
    @inbounds for j in 3:ndata
        output[j] = coef[1]*input[j] + coef[2]*input[j-1] + coef[3]*input[j-2] + coef[4]*output[j-1] + coef[5]*output[j-2]
    end
end



function _get_fb2_coefs(fc, fs, lowpass, T)
    a = tan(pi * fc / fs)
    amort = 0.47
    d = 1 + 2 * amort * a + a * a
    
    a0 = lowpass ? (a * a / d) : (1 / d)
    a1 = lowpass ? (2 * a0) : (-2 * a0)
    a2 = a0
    b1 = -(2 * a * a - 2) / d
    b2 = -(1 - 2 * amort * a + a * a) / d

    return (T(a0), T(a1), T(a2), T(b1), T(b2))
end


function filter!(S::AbstractSeisArray{T}, fmin::Real, fmax::Real) where T

    _filter!(S.data, S.fs, T(fmin), T(fmax))

    return S 
end


function _filter!(data::AbstractMatrix{T}, fs::Real, fmin::T, fmax::T) where T
  
    ntime, nsta = size(data)

    coef_h = _get_fb2_coefs(fmax, fs, true, T)
    coef_l = _get_fb2_coefs(fmin, fs, false, T)

    temp_buf = zeros(T, ntime)

    @inbounds @views for i in 1:nsta
        col = data[:, i]
        # Forward
        _fb2_inplace!(col, temp_buf, coef_h)
        _fb2_inplace!(temp_buf, col, coef_l)
        # reverse
        reverse!(col)
        # backward
        _fb2_inplace!(col, temp_buf, coef_h)
        _fb2_inplace!(temp_buf, col, coef_l)
        # reverse
        reverse!(col)
    end
    return nothing
end



function smooth_spectrum!(out::AbstractVector, inp::AbstractVector, B::Int)
    # Implementación manual de convolución (moving average)
    
    n = length(inp)
    half_B = (B - 1) ÷ 2
    inv_B  = 1.0 / B
    
    @inbounds for i in 1:n
        acc = zero(eltype(out))
        for k in -half_B:half_B
            idx = i + k
            if 1 <= idx <= n
                acc += inp[idx]
            end
        end
        out[i] = acc * inv_B 
    end
end



function cc_overlap(s_ref::AbstractVector, s_mov::AbstractVector, lag::Int, N::Int, min_overlap::Int=50)

    # 1. Definir los índices de intersección (Overlap)
    # Si lag > 0: s_mov se desplaza a la derecha (sus primeros datos salen, entran ceros imaginarios)
    # Comparamos la cola de s_ref con la cabeza de s_mov
    
    if lag >= 0
        # s_ref: desde (1 + lag) hasta N
        # s_mov: desde 1 hasta (N - lag)
        range_ref = (1+lag):N
        range_mov = 1:(N-lag)
    else # lag < 0
        # s_mov se desplaza a la izquierda
        # s_ref: desde 1 hasta (N + lag)  (recuerda que lag es negativo)
        # s_mov: desde (1 - lag) hasta N
        range_ref = 1:(N+lag)
        range_mov = (1-lag):N
    end
    
    # Pocos samples = Correlación falsa
    len_overlap = length(range_ref)
    if len_overlap < min_overlap
        return 0.0
    end

    # recorta la seccion
    v_ref = @view s_ref[range_ref]
    v_mov = @view s_mov[range_mov]
    
    # Calcular Pearson sobre las vistas
    num = 0.0
    sq_ref = 0.0
    sq_mov = 0.0
    @inbounds @simd for i in 1:len_overlap
        x = v_ref[i]
        y = v_mov[i]
        num    = muladd(x, y, num)
        sq_ref = muladd(x, x, sq_ref)
        sq_mov = muladd(y, y, sq_mov)
    end
    
    den = sqrt(sq_ref * sq_mov)
    
    if den > 1e-12
        return num / den
    else
        return 0.0
    end
end


function haning_windows(lwin::Int, ::Type{T}=Float64) where {T<:AbstractFloat}

    windows = Vector{Vector{T}}(undef, lwin)
    
    @inbounds for n in 1:lwin
        win = zeros(T, n)
        if n == 1
            win[1] = one(T)
        else
            fac = T(2π) / (n - 1)  # ← Paréntesis para precisión
            for i in 1:n
                win[i] = T(0.5) * (one(T) - cos(fac * (i - 1)))
            end
        end
        windows[n] = win
    end

    return windows
end
