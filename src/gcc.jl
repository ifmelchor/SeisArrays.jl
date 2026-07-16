#!/usr/local/bin julia
# coding=utf-8


struct WSGCC{T<:AbstractFloat, P1, P2}
    fs::T              # Sampling rate
    n::Int             # Tamaño de la señal
    n_fft::Int         # Tamaño con Zero-Padding
    n_up::Int          # Tamaño interpolado
    B::Int             # Ancho de la ventana de suavizado
    exc_th::T          # Tiempo de exclusion
    method::Symbol     # Metodo de ponderacion
    r_eff::T           # Grados de libertad efectivo
    
    # Buffers temporales
    buf_s1::Vector{T}
    buf_s2::Vector{T}
    window::Vector{T} # Ventana de Hann pre-calculada
    
    # Buffers Frecuenciales
    n_freq::Int
    S1::Vector{Complex{T}}
    S2::Vector{Complex{T}}
    G12::Vector{Complex{T}}
    G12_smooth::Vector{Complex{T}}
    G11_smooth::Vector{T}
    G22_smooth::Vector{T}
    
    # mascara de banda de frequencia
    mask::Vector{T}
    
    # Buffer de interpolación (Upsampling)
    n_freq_up::Int
    C_up_freq::Vector{Complex{T}}
    C_up_time::Vector{T} # Resultado final en tiempo
    
    # Planes de FFT (Lo más costoso de crear)
    plan_fwd::P1
    plan_inv::P2
end

"""
    init_wsgcc(n, fs, fmin, fmax, B, upsample, df_taper, method)

    Inicializa el workspace para FS-GCC.

    # Parámetros
    - `n`: Longitud de la ventana de análisis en muestras
    - `fs`: Frecuencia de muestreo en Hz
    - `fmin`, `fmax`: Banda de análisis en Hz
    - `B`: Ancho de suavizado espectral en bins
    - `upsample`: Factor de interpolación
    - `df_taper`: Ancho del taper coseno en Hz
    - `method`: `:ml`, `:coh`, `:scot`, o `:phat`
"""
function init_wsgcc(n::Int, fs::T, fmin::T, fmax::T, B::Int, upsample::Int, df_taper::T, method::Symbol) where {T<:AbstractFloat}

    fmin < fmax   || throw(ArgumentError("fmin ($fmin) debe ser < fmax ($fmax)"))
    fmin >= 0     || throw(ArgumentError("fmin debe ser ≥ 0"))
    fmax <= fs/2  || throw(ArgumentError("fmax ($fmax) debe ser ≤ fs/2 ($(fs/2))"))
    B > 0         || throw(ArgumentError("B debe ser > 0"))
    isodd(B)      || throw(ArgumentError("B debe ser impar (recibido: $B)"))
    upsample >= 1 || throw(ArgumentError("upsample debe ser ≥ 1"))
    df_taper >= 0 || throw(ArgumentError("df_taper debe ser ≥ 0"))

    if method ∉ (:ml, :coh, :scot, :phat)
        throw(ArgumentError("Método no soportado: :$method. Usa :ml, :coh, :scot, o :phat"))
    end

    n_fft = nextpow(2, 2*n - 1)
    n_up = n_fft * upsample
    n_freq = (n_fft ÷ 2) + 1
    n_freq_up = (n_up ÷ 2) + 1
    exc_th = T(0.5)/(fmax-fmin)

    # ventana de hanning
    t = range(zero(T), stop=T(n-1), length=n)
    w_time = T(0.5) .* (one(T) .- cos.(T(2π) .* t ./ T(n-1)))

    # Factor de corrección por ventana temporal
    # Para Hann --> 2/3 (Harris, 1978)
    r_eff = T(B * n) / T(1.5 * n_fft)
    
    # Máscara de frecuencia
    freqs = rfftfreq(n_fft, fs) # solo positivas
    mask  = zeros(T, n_freq)
    @inbounds for i in 1:n_freq
        f = freqs[i]
        if f >= fmin && f <= fmax
            if f < fmin + df_taper
                mask[i] = T(0.5) * (one(T) - cos(T(π) * (f - fmin) / df_taper))
            elseif f > fmax - df_taper
                mask[i] = T(0.5) * (one(T) - cos(T(π) * (fmax - f) / df_taper))
            else
                mask[i] = one(T)
            end
        end
    end
    
    # Buffers
    buf_s1 = zeros(T, n_fft)
    buf_s2 = zeros(T, n_fft)
    S1 = zeros(Complex{T}, n_freq)
    S2 = zeros(Complex{T}, n_freq)
    G12 = zeros(Complex{T}, n_freq)
    G12_smooth = zeros(Complex{T}, n_freq)
    G11_smooth = zeros(T, n_freq)
    G22_smooth = zeros(T, n_freq)
    C_up_freq = zeros(Complex{T}, n_freq_up)
    C_up_time = zeros(T, n_up)

    # planes FFT
    plan_fwd = plan_rfft(buf_s1; flags=FFTW.MEASURE)
    plan_inv = plan_irfft(C_up_freq, n_up; flags=FFTW.MEASURE)

    return WSGCC{T, typeof(plan_fwd), typeof(plan_inv)}(
        fs, n, n_fft, n_up, B, exc_th, method, r_eff,
        buf_s1, buf_s2, w_time,
        n_freq, S1, S2, G12, G12_smooth, G11_smooth, G22_smooth,
        mask,
        n_freq_up, C_up_freq, C_up_time,
        plan_fwd, plan_inv
    )
end


function _psr(idx::Int, max_corr::T, ws::WSGCC{T}) where {T<:AbstractFloat}
    
    n_up = ws.n_up
    second_peak = T(1e-12)

    exclusion  = round(Int, ws.exc_th * ws.fs * (T(n_up) / T(ws.n)))
    excl_start = mod1(idx - exclusion, n_up)
    excl_end   = mod1(idx + exclusion, n_up)
    
    if excl_start < excl_end
        # CASO A: La exclusión es un bloque central [ ... XXX ... ]
        
        # Tramo Izquierdo
        @inbounds for i in 1:(excl_start - 1)
            val = abs(ws.C_up_time[i])
            if val > second_peak
                second_peak = val
            end
        end
        
        # Tramo Derecho
        @inbounds for i in (excl_end + 1):n_up
            val = abs(ws.C_up_time[i])
            if val > second_peak
                second_peak = val
            end
        end
        
    else
        # CASO B: La exclusión da la vuelta [ XX ... XX ]
        @inbounds for i in (excl_end + 1):(excl_start - 1)
            val = abs(ws.C_up_time[i])
            if val > second_peak
                second_peak = val
            end
        end
    end

    psr_db = T(20) * log10(abs(max_corr) / second_peak)

    return psr_db
end


function _gcc_delay_core!(ws::WSGCC{T}, s1_in::AbstractVector{T}, s2_in::AbstractVector{T}) where {T<:AbstractFloat}
    
    # Media y tapering
    m1 = mean(s1_in)
    m2 = mean(s2_in)

    @inbounds @simd for i in 1:ws.n
        ws.buf_s1[i] = (s1_in[i] - m1) * ws.window[i]
        ws.buf_s2[i] = (s2_in[i] - m2) * ws.window[i]
    end

    ws.buf_s1[ws.n+1:end] .= zero(T)
    ws.buf_s2[ws.n+1:end] .= zero(T)
    
    # FFT con planes precomputados
    mul!(ws.S1, ws.plan_fwd, ws.buf_s1)
    mul!(ws.S2, ws.plan_fwd, ws.buf_s2)
    
    # Espectro cruzado
    @. ws.G12 = ws.S1 * conj(ws.S2)
    
    # Suavizado sub-bandas
    smooth_spectrum!(ws.G12_smooth, ws.G12, ws.B)
    smooth_spectrum!(ws.G11_smooth, abs2.(ws.S1), ws.B)
    smooth_spectrum!(ws.G22_smooth, abs2.(ws.S2), ws.B)
    
    norm_accum      = zero(T)
    crlb_num        = zero(T)
    crlb_den        = zero(T)
    coherence_accum = zero(T)
    max_power       = -one(T)
    dfreq           = zero(T)
    df              = ws.fs / ws.n_fft
    
    @inbounds @simd for i in 1:ws.n_freq
        denom2 = max(zero(T), ws.G11_smooth[i] * ws.G22_smooth[i])
        denom  = sqrt(denom2) + T(1e-12)
        num    = abs(ws.G12_smooth[i])
        gamma  = num / denom
        gamma2 = clamp(gamma^2, zero(T), one(T) - T(1e-6))

        if ws.mask[i] > 0
            f_k    = (i - 1) * df

            # Peso según método (Tabla 1, Brennan et al. 2007)
            if ws.method === :ml
                w_gps = gamma2 / max(one(T) - gamma2, T(1e-4))
            
            elseif ws.method === :coh
                w_gps = gamma2
            
            elseif ws.method === :scot
                w_gps = gamma
            
            else  # :phat
                w_gps = one(T)
            end
            
            # # CRLB: σ² = Σ|Wi|²ωi²(1-γ²)/γ² / (2r·(Σ|Wi|ωi²)²)
            ω_k = T(2π) * f_k
            crlb_num += w_gps^2 * ω_k^2 * (one(T) - gamma2) / gamma2
            crlb_den += w_gps * ω_k^2 
            coherence_accum += gamma * ws.mask[i]

            if num > max_power
                max_power = num
                dfreq     = f_k
            end
        end
        
        # Ponderación GCC para IFFT (Tabla 1, columna Cg(ω))
        if ws.method === :ml
            # Maximum Likelihood: γ²/[(1-γ²)|G12|]
            w_num = max(one(T) - gamma2, T(1e-4))
            w     = gamma2 / ((w_num * num) + T(1e-12))
        elseif ws.method === :coh
            # Coherence: γ²/|G12|
            w = gamma2 / (num + T(1e-12))
        elseif ws.method === :scot
            # Smoothed Coherence Transform: γ/|G12|
            w = gamma / (num + T(1e-12))
        else  # :phat
            # PHAT: 1/|G12|
            w = one(T) / (num + T(1e-12))
        end
        
        ws.G12_smooth[i] = (ws.G12_smooth[i] * w) * ws.mask[i]
        norm_accum      += ws.mask[i]
    end

    coherence = (norm_accum > zero(T)) ? (coherence_accum / norm_accum) : zero(T)

    # CRLB según Brennan et al. 2007, ec. (23)
    # σ² = crlb_num / (2r · crlb_den²)
    if crlb_den > T(1e-12)
        sigma = sqrt(crlb_num / (T(2) * ws.r_eff * crlb_den^2))
    else
        sigma = typemax(T)
    end

    # Zero-padding + IFFT
    fill!(ws.C_up_freq, complex(zero(T), zero(T)))
    copyto!(ws.C_up_freq, 1, ws.G12_smooth, 1, ws.n_freq)
    mul!(ws.C_up_time, ws.plan_inv, ws.C_up_freq)
    
    # Búsqueda del pico
    max_corr_raw, idx = findmax(ws.C_up_time)
    theoretical_max   = (norm_accum * T(2.0)) / ws.n_up

    cc_val = if theoretical_max > T(1e-6)
        clamp(max_corr_raw / theoretical_max, T(-1.0), T(1.0))
    else
        zero(T)
    end
 
    # Métricas finales
    psr = _psr(idx, cc_val, ws)
    lag = (idx <= ws.n_up ÷ 2) ? (idx - 1) : (idx - 1 - ws.n_up)
    delay = T(lag) / (ws.fs * (T(ws.n_up) / T(ws.n_fft)))

    return (; delay, psr, sigma, coherence, dfreq)
end


function gcc_delay(s1_in::AbstractVector{T}, s2_in::AbstractVector{T}, fs::T, fmin::T, fmax::T, B::Int, upsample::Int, df_taper::T; method::Symbol=:ml) where {T<:AbstractFloat}
    ws = init_wsgcc(length(s1_in), fs, fmin, fmax, B, upsample, df_taper, method)
    return _gcc_delay_core!(ws, s1_in, s2_in)
end


function smooth_spectrum!(out::AbstractVector{T}, inp::AbstractVector{T}, B::Int) where {T<:Number}

    n = length(inp)
    half_B = (B - 1) ÷ 2

    # Suma inicial para i=1
    s = zero(T)
    upper = min(n, 1 + half_B)
    @inbounds @simd for k in 1:upper
        s += inp[k]
    end
    out[1] = s / T(upper)

    @inbounds for i in 2:n
        # Elemento que entra por la derecha
        new_idx = i + half_B
        if new_idx <= n
            s += inp[new_idx]
        end

        # Elemento que sale por la izquierda
        old_idx = i - half_B - 1
        if old_idx >= 1
            s -= inp[old_idx]
        end

        # Conteo de elementos en la ventana
        lo = max(1, i - half_B)
        hi = min(n, i + half_B)
        count = hi - lo + 1
        out[i] = s / T(count)
    end
end


function delay_matrix_gcc(ws::WSGCC{T}, data::AbstractMatrix{T}; psr_th::T=5.0) where {T<:AbstractFloat}
    
    lwin  = size(data, 1)
    N     = size(data, 2)
    mdelay  = fill(NaN, N, N)
    msigma  = fill(NaN, N, N)
    mcorr   = fill(NaN, N, N)
    mfreq   = fill(NaN, N, N)
    
    @views @inbounds for i in 1:N, j in i+1:N
        res = _gcc_delay_core!(ws, data[:,j], data[:,i])
        if res.psr >= psr_th
            mdelay[j,i] =  res.delay
            mdelay[i,j] = -res.delay
            msigma[i,j] =  res.sigma
            msigma[j,i] =  res.sigma
            mcorr[i,j] =  res.coherence
            mcorr[j,i] =  res.coherence
            mfreq[i,j] = res.dfreq
            mfreq[j,i] = res.dfreq
        end
    end
    
    return return (; mdelay, msigma, mcorr, mfreq)
end