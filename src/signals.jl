#!/usr/local/bin julia
# coding=utf-8

"""
    sigmoid
"""
function sigmoid(x::T; k::T=T(10.0), x0::T=T(0.0)) where {T<:Real}
    return T(1) / (T(1) + exp(-k * (x - x0)))
end


function sort3_median(a::T, b::T, c::T) where {T<:AbstractFloat}
    (a <= b <= c || c <= b <= a) && return b
    (b <= a <= c || c <= a <= b) && return a
    return c
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


function pearson_overlap(sref::AbstractVector{T}, smov::AbstractVector{T}, lag::Int, N::Int, min_overlap::Int=50) where {T<:Real}

    # Si lag > 0: s_mov se desplaza a la derecha (sus primeros datos salen, entran ceros imaginarios)
    
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
        return zero(T)
    end

    # recorta la seccion
    v_ref = @view sref[range_ref]
    v_mov = @view smov[range_mov]
    
    # Calcular Pearson sobre las vistas
    num    = zero(T)
    sq_ref = zero(T)
    sq_mov = zero(T)
    
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
        return zero(T)
    end
end

