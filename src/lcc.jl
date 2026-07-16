#!/usr/local/bin julia
# coding=utf-8

"""
Calcula la Correlación Cruzada Local (Local CC) entre dos señales
"""

struct LCCParams{T<:Real}
    # --- CF Kurtosis ---
    T_decay   :: T      # constante de decaimiento [s]
                        # controla la resolución temporal de la CF
    sigma_min :: T      # floor de varianza — evita división por cero
    order     :: Int    # orden HOS: 4=kurtosis, 3=skewness
    transform :: Bool   # true → derivada positiva + gaussiana
                        # false → kurtosis cruda sin transformar

    # --- Filtro gaussiano LCC ---
    sigma_gauss :: T    # ancho del gaussiano [s]
                        # controla la localidad temporal de LCC

    # --- Umbral de validación ---
    min_peak :: T       # coherencia mínima aceptable en la banda óptima
                        # equivalente a psr_th en GCC
end


struct LCC_ws{T<:Real, R<:AbstractRange}
    fs::Float64            # Sampling rate
    lwin::Int              # Tamaño de la ventana (muestras)
    global_nlags::Int      # Máximo lag posible global

    cc_buffer::Matrix{T}   # Buffer 2D preasignado: (lwin, global_nlags)
    cf1::Vector{T}         # Buffer para la Broadband CF de la estación 1
    cf2::Vector{T}         # Buffer para la Broadband CF de la estación 2

    sigma_samples::T

    lmax_per_pair::Vector{Int}
    nlags_per_pair::Vector{Int}
    time_lags_per_pair::Vector{R} # Especialización limpia del rango
end


function init_lcc_ws(lwin::Int, fs::Float64, dd::Vector{T}, slowmax::T, params::LCCParams{T}) where T<:Real
    
    tau_max_global = maximum(dd) * slowmax
    global_lmax    = round(Int, tau_max_global * fs)
    global_nlags   = 2 * global_lmax
    
    cc_buffer = zeros(T, lwin, global_nlags)
    cf1       = zeros(T, lwin)
    cf2       = zeros(T, lwin)
    
    lmax_per_pair  = [round(Int, d * slowmax * fs) for d in dd]
    nlags_per_pair = 2 .* lmax_per_pair
    
    time_lags_per_pair = [
        range(T(-lmax/fs), stop=T((lmax-1)/fs), length=2*lmax)
        for lmax in lmax_per_pair
    ]

    sigma_samples = params.sigma_gauss * fs
    
    return LCC_ws(fs, lwin, global_nlags, cc_buffer, cf1, cf2, sigma_samples, lmax_per_pair, nlags_per_pair, time_lags_per_pair)
end


function lcc_delay(cc::AbstractMatrix{T}, cc_time_lags::AbstractRange) where T<:Real

    npts, nlags = size(cc)
    dt_lag = step(cc_time_lags)

    # === LCC_MAX — máximo sobre tiempo para cada lag (Poiata Eq. 28) ===
    peak_val, max_cart = findmax(cc)
    t_peak, l_peak = max_cart.I
    dt_opt = cc_time_lags[l_peak]
    sigma = Inf
    
    # Interpolación parabólica sub-muestra e Incertidumbre
    if 1 < l_peak < nlags
        @inbounds begin
            alpha = cc[t_peak, l_peak - 1]
            beta  = peak_val
            gamma = cc[t_peak, l_peak + 1]
        end
        
        denom = alpha - 2*beta + gamma
        if abs(denom) > eps(Float64)
            # Ajuste del pico por interpolación
            p = 0.5 * (alpha - gamma) / denom  
            dt_opt += p * dt_lag
            peak_val = beta - (alpha - gamma)^2 / (8.0 * denom)
            
            # Incertidumbre por curvatura
            d2 = denom / (dt_lag^2)
            sigma = d2 < -eps(Float64) ? sqrt(-peak_val / d2) : Inf
        end
    end
    
    return dt_opt, sigma, t_peak, l_peak, peak_val
end


function local_cc!(cc::AbstractMatrix{T}, signal1::AbstractVector{T}, signal2::AbstractVector{T}, fs::Real, max_time_lag::Real, sigma_samples::Real) where T<:Real

    npts = length(signal1)
    lmax = round(Int, max_time_lag * fs)

    @inbounds for l in -lmax:(lmax - 1)
        l_f = fld(l, 2)
        l_g = cld(l, 2)
        col = l + lmax + 1
        
        # === Borde Inicial ===
        for n in 1:lmax
            i11 = n - l_f; i22 = n + l_g
            i12 = n - l_g; i21 = n + l_f
            cc[n, col] = T(0.5) * (
                (1 ≤ i11 ≤ npts ? signal1[i11] : zero(T)) * (1 ≤ i22 ≤ npts ? signal2[i22] : zero(T)) +
                (1 ≤ i12 ≤ npts ? signal1[i12] : zero(T)) * (1 ≤ i21 ≤ npts ? signal2[i21] : zero(T))
            )
        end

        # === Nucleo (¡A toda velocidad con SIMD!) ===
        @simd for n in (lmax + 1):(npts - lmax)
            cc[n, col] = T(0.5) * (
                signal1[n - l_f] * signal2[n + l_g] + 
                signal1[n - l_g] * signal2[n + l_f]
            )
        end

        # === Borde Final ===
        for n in (npts - lmax + 1):npts
            i11 = n - l_f; i22 = n + l_g
            i12 = n - l_g; i21 = n + l_f
            cc[n, col] = T(0.5) * (
                (1 ≤ i11 ≤ npts ? signal1[i11] : zero(T)) * (1 ≤ i22 ≤ npts ? signal2[i22] : zero(T)) +
                (1 ≤ i12 ≤ npts ? signal1[i12] : zero(T)) * (1 ≤ i21 ≤ npts ? signal2[i21] : zero(T))
            )
        end
        
        # === FILTRADO GAUSSIANO ===
        if sigma_samples >= 0.5
            _gaussian_recursive_filter!(@view(cc[:, col]), sigma_samples)
        end
    end
    
    return nothing
end


function delay_matrix_lcc(data::AbstractMatrix{T}, fs::Real, max_time_lag::Real; sigma::Real=0.0, s_th::Real=Inf) where T<:Real
    
    lwin = size(data, 1)
    N    = size(data, 2)
    
    # Matrices de salida
    DT    = fill(NaN, N, N)
    CC    = fill(NaN, N, N)
    SIGMA = fill(NaN, N, N)
    
    # Pre-allocations
    lmax = round(Int, max_time_lag * fs)
    nlags = 2 * lmax
    cc_buffer = zeros(T, lwin, nlags)
    cc_time_lags = range(-lmax / fs, (lmax - 1) / fs, length=nlags)
    dt_lag = step(cc_time_lags)
    sigma_samples = sigma * fs
    
    @views @inbounds for i in 1:N, j in i+1:N

        local_cc!(cc_buffer, data[:, j], data[:, i], fs, max_time_lag, sigma_samples)
        
        # Extraemos valores
        dt_opt, sigma, t_peak, l_peak, peak_val = lcc_delay(cc_buffer, cc_time_lags, dt_lag)
        
        # Filtro de calidad
        if sigma <= s_th
            lag_int = l_peak - (lmax + 1)
            cc_val = pearson_overlap(data[:, j], data[:, i], lag_int, lwin)
            DT[j, i]    = dt_opt
            DT[i, j]    = -dt_opt
            CC[i, j]    = cc_val
            CC[j, i]    = cc_val
            SIGMA[i, j] = sigma
            SIGMA[j, i] = sigma
        end
    end
    
    return DT, CC, SIGMA
end