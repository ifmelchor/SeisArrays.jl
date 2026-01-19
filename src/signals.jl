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