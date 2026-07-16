#!/usr/local/bin julia
# coding=utf-8

abstract type AbstractCFType end

"""
    RMSCF(; T_decay=0.5)

Characteristic Function based on Recursive RMS Envelope.
Suitable for emergent signals.

# Fields
- `T_decay::Float64`: Reference decay time [s] for the central frequency band.
"""
Base.@kwdef struct RMSCF <: AbstractCFType 
    T_decay::Float64 = 0.5 
end

"""
    KurtosisCF(; T_decay=3.0, sigma_min=1e-8, order=4, transform=false)

Characteristic Function based on Recursive Kurtosis

# Fields
- `T_decay::Float64`: Reference decay time [s] for scaling with frequency.
- `sigma_min::Float64`: Floor for variance estimation to avoid numerical instability.
- `order::Int`: Order of the HOS moment (default=4 for kurtosis).
- `transform::Bool`: If true, return CF with positive derivative + Gaussian smoothing.
"""
Base.@kwdef struct KurtosisCF <: AbstractCFType 
    T_decay::Float64 = 3.0
    sigma_min::Float64 = 1e-8
    order::Int = 4
    transform::Bool = false
end


function kurtosis_cf!(cf_out::AbstractVector{T}, u::AbstractVector{T}, dt::Real, T_decay::Real; sigma_min::Real=1e-10, order::Int=4, transform::Bool=false) where T<:Real
    
    N = length(u)

    # Constante de decaimiento
    C = clamp(dt / T_decay, eps(), 1.0)
    invC = 1.0 - C
    power = order / 2
    
    # Inicializar variables
    μ = zero(T); var = zero(T); hos = zero(T)
    n_win = min(ceil(Int, 1/C), N)
    @inbounds for i in 1:n_win
        μ = C * u[i] + invC * μ
        var_temp = C * (u[i] - μ)^2 + invC * var
        var = var_temp > sigma_min ? var_temp : sigma_min
        k_inst = (u[i] - μ)^order / (var^power + eps())
        hos = C * k_inst + invC * hos
    end
    
    # Bucle principal
    @inbounds for i in 1:N
        μ = C * u[i] + invC * μ
        var_temp = C * (u[i] - μ)^2 + invC * var
        var = var_temp > sigma_min ? var_temp : sigma_min
        
        k_inst = (u[i] - μ)^order / (var^power + eps())
        hos = C * k_inst + invC * hos
        cf_out[i] = hos
    end
    
    if !transform
        return (μ, var, hos)
    end
    
    # Derivada positiva
    @inbounds @simd for i in N:-1:2
        cf_out[i] = max(0.0, cf_out[i] - cf_out[i-1])
    end
    cf_out[1] = 0.0
    
    # Convolución Gaussiana recursiva
    sigma = T_decay / 2.0
    _gaussian_recursive_filter!(cf_out, sigma/dt)
    
    return (μ, var, hos)
end


function recursive_rms!(signal::AbstractVector{T}, rms_signal::AbstractVector{T}, C_WIN::Real; mean_sq::Real=zero(T), memory_sample::Int=-1, initialize::Bool=true) where T<:Real
    """
    Recursive RMS envelope calculation following Poiata et al. 2016
    Implements Eq. (8) of the paper: CF_env(t) = sqrt(C·u(t)² + (1-C)·CF_env(t-1)²)
    """

    npts = length(signal)
    
    # C truncates: n_win = (int) 1/C_WIN
    n_win = max(1, trunc(Int, 1.0 / C_WIN))
    _rms = T(mean_sq)
    
    # Convert C's 0-based memory_sample to Julia's 1-based index
    mem_idx = (memory_sample < 0 || memory_sample >= npts) ? npts : memory_sample + 1
    
    # Pre-warming (initialization) exactly as in C
    if initialize && n_win > 0
        sum_sq = zero(T)
        limit = min(n_win, npts)
        @inbounds for j in 1:limit
            sum_sq += signal[j]^2
        end
        _rms = sqrt(sum_sq / limit)
    end
    
    @inbounds for i in 1:npts
        # Recursive RMS: rms_new = sqrt(C * x² + (1-C) * rms_old²)
        _rms = sqrt(C_WIN * signal[i]^2 + (1.0 - C_WIN) * _rms^2)
        rms_signal[i] = _rms

        if i == mem_idx
            mean_sq = _rms
        end
    end
    
    return rms_signal, mean_sq
end


"""
    mbf_cf(data, fs, fmin, fmax, nband, method::AbstractCFType; 
           operator=:max, normalize=true)

Unified Multiband Filter Characteristic Function computation.

# Arguments
- `data::AbstractMatrix`: Input signals (ntime × nsta)
- `fs::Real`: Sampling frequency [Hz]
- `fmin`, `fmax::Real`: Frequency range [Hz]
- `nband::Int`: Number of frequency bands
- `method::AbstractCFType`: CF method (`RMSCF()` or `KurtosisCF()`)

# Keywords
- `operator::Symbol`: Broad-band composition: `:max` (Eq. 9) or `:rms` (Eq. 10)
- `normalize::Bool`: Normalize each band/station CF to [0, 1]

# Returns
- `cf_out::Matrix{Float64}`: Broad-band CF (ntime × nsta)
- `f_central::Vector{Float64}`: Central frequencies of each band [Hz]
"""
function mbf_cf(data::AbstractMatrix{T}, fs::Real, fmin::Real, fmax::Real, nband::Int, method::AbstractCFType; operator::Symbol=:max, normalize::Bool=true) where T<:Real
    
    ntime, nsta = size(data)
    
    # MBF decomposition
    U, f_central = mbf_filter(data, fs, fmin, fmax, nband)
    
    # Allocate output for band-wise CFs
    cf_bands = Array{Float64, 3}(undef, ntime, nsta, nband)
    cf_out   = Matrix{Float64}(undef, ntime, nsta)
    
    # Dispatch to type-specific band calculation
    _calculate_bands!(cf_bands, U, fs, f_central, method)
    
    # Optional normalization per band/station
    if normalize
        _normalize_bands!(cf_bands)
    end
    
    # Broad-band composition
    _compose_broadband!(cf_out, cf_bands, operator)
    
    return cf_out, f_central
end


function _calculate_bands!(cf_bands::AbstractArray{Float64, 3}, U::AbstractArray{Float64, 3}, fs::Real, f_central::AbstractVector{Float64}, method::RMSCF)
    
    ntime, nsta, nband = size(cf_bands)
    
    # Reference frequency for scaling
    idx_center = cld(nband, 2)
    f_center_ref = f_central[idx_center]
    
    # Reusable buffer for RMS computation
    rms_buf = Vector{Float64}(undef, ntime)
    
    @inbounds for b in 1:nband
        # Frequency-dependent T_decay: lower freq → longer memory
        T_decay = method.T_decay * (f_center_ref / f_central[b])
        C_WIN = (1.0 / fs) / T_decay
        
        for s in 1:nsta
            cf_bands[:, s, b], _ = recursive_rms!(U[:, s, b], rms_buf, C_WIN; initialize=true, memory_sample=-1)
        end
    end
    
    return nothing
end


function _calculate_bands!(cf_bands::AbstractArray{Float64, 3}, U::AbstractArray{Float64, 3}, fs::Real, f_central::AbstractVector{Float64}, method::KurtosisCF)
    
    ntime, nsta, nband = size(cf_bands)
    dt = 1.0 / fs

    # Frecuencia de referencia (banda central, Eq. 32 de Poiata)
    f_idx = cld(nband, 2)
    f_ref = f_central[f_idx]
    
    @inbounds @views for b in 1:nband
        # Escalado dinámico del tiempo de decaimiento
        T_decay = method.T_decay * (f_ref / f_central[b])
        
        for s in 1:nsta
            # Compute kurtosis CF
            kurtosis_cf!(cf_bands[:, s, b], U[:, s, b], dt, T_decay; sigma_min=method.sigma_min, order=method.order, transform=method.transform)
        end
    end
    
    return nothing
end


function _normalize_bands!(cf_bands::AbstractArray{Float64, 3})
    ntime, nsta, nband = size(cf_bands)
    @inbounds @views for b in 1:nband, s in 1:nsta
        m = maximum(cf_bands[:, s, b])
        if m > eps(Float64)
            inv_m = 1.0 / m
            @simd for t in 1:ntime
                cf_bands[t, s, b] *= inv_m
            end
        end
    end
end


function _compose_broadband!(cf_out::AbstractMatrix{Float64}, cf_bands::AbstractArray{Float64, 3}, operator::Symbol)
    ntime, nsta, nband = size(cf_bands)

    if operator == :max
        # Poiata Eq. 9: maximum operator
        @inbounds for s in 1:nsta
            
            @simd for t in 1:ntime
                cf_out[t, s] = cf_bands[t, s, 1]
            end
            
            for b in 1:nband
                @simd for t in 1:ntime
                    val = cf_bands[t, s, b]
                    cf_out[t, s] = ifelse(val > cf_out[t, s], val, cf_out[t, s])
                end
            end
        end

    elseif operator == :rms
        # Poiata Eq. 10: RMS operator
        inv_n = 1.0 / nband

        @inbounds for s in 1:nsta
            
            @simd for t in 1:ntime
                cf_out[t, s] = 0.0
            end

            for b in 1:nband
                @simd for t in 1:ntime
                    val = cf_bands[t, s, b]
                    cf_out[t, s] += val * val
                end
            end

            @simd for t in 1:ntime
                cf_out[t, s] = sqrt(cf_out[t, s] * inv_n)
            end

        end

    else
        error("operator must be :max or :rms, got :$operator")
    end
end
