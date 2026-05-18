#!/usr/local/bin julia
# coding=utf-8


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


"""
    init_fsgcc(n, fs, fmin, fmax, B, n_gamma, upsample, df_taper)

Inicializa el workspace para la estimación de delay via FS-GCC
(Frequency-Sliding Generalized Cross-Correlation).

Parámetros
----------
n        : Int
    Longitud de la ventana de análisis en muestras.

fs       : Real
    Frecuencia de muestreo en Hz.

fmin     : Real
    Frecuencia mínima de la banda de análisis en Hz.

fmax     : Real
    Frecuencia máxima de la banda de análisis en Hz.

B        : Int
    Ancho de la ventana de suavizado espectral en bins de frecuencia.
    Controla el promediado sub-banda de Cobos et al. (2020).
    Valores típicos: 3-11. Mayor B → más suavizado, menor varianza 
    espectral, menor resolución frecuencial.

n_gamma  : Real
    Exponente de ponderación de la coherencia (γ^n_gamma).
    n_gamma=0 → sin ponderación (correlación cruzada clásica).
    n_gamma=1 → SCOT.
    n_gamma=2 → ponderación por coherencia al cuadrado (Faerman et al. 2022).

upsample : Int
    Factor de interpolación espectral via zero-padding antes de la IFFT.
    Resolución temporal efectiva: 1/(upsample·fs).
    Valores típicos: 4-16. Mayor upsample → mayor precisión sub-sample,
    mayor costo computacional.

df_taper : Real
    Ancho del taper coseno en los bordes de la banda [fmin, fmax] en Hz.
    Suaviza la transición de la máscara espectral para evitar efectos 
    de Gibbs en la correlación. df_taper=0 → máscara rectangular.

Retorna
-------
ws : FSGCC_ws
    Workspace pre-alocado con todos los buffers y planes FFTW listos
    para uso en compute_delay!.
"""
function init_fsgcc(n::Int, fs::Real, fmin::Real, fmax::Real, B::Int, n_gamma::Real, upsample::Int, df_taper::Real)

    n_fft = nextpow(2, 2*n - 1)
    n_up = n_fft * upsample
    n_freq = (n_fft ÷ 2) + 1
    n_freq_up = (n_up ÷ 2) + 1
    
    # ventana de hanning
    w_time = 0.5 .* (1.0 .- cos.(2π .* (0:n-1) ./ (n-1)))
    
    # Pre-calcular mascara de frecuencia
    freqs = rfftfreq(n_fft, fs) # solo positivas
    mask = zeros(n_freq)
    @inbounds for i in 1:n_freq
        f = freqs[i]
        if f >= fmin && f <= fmax
            if f < fmin + df_taper
                mask[i] = 0.5 * (1 - cos(pi * (f - fmin) / df_taper))
            elseif f > fmax - df_taper
                mask[i] = 0.5 * (1 - cos(pi * (fmax - f) / df_taper))
            else
                mask[i] = 1.0
            end
        end
    end
    
    # Buffers con Zero-Padding implícito
    b_s1 = zeros(Float64, n_fft)
    b_s2 = zeros(Float64, n_fft)

    # Buffers Frecuenciales
    b_s1_f = zeros(ComplexF64, n_freq)
    b_s2_f = zeros(ComplexF64, n_freq)
    b_g12_f = zeros(ComplexF64, n_freq)
    b_g12_s = zeros(ComplexF64, n_freq)
    b_g1_s = zeros(Float64, n_freq)
    b_g2_s = zeros(Float64, n_freq)

    # Buffers Upsampling
    b_f_up = zeros(ComplexF64, n_freq_up)
    b_C_up = zeros(Float64, n_up)

    # planes fftw
    plan_fwd = plan_rfft(b_s1; flags=FFTW.MEASURE)
    plan_inv = plan_irfft(b_f_up, n_up; flags=FFTW.MEASURE)

    ws = FSGCC_ws(fs, n, n_fft, n_up, B, Float64(n_gamma), b_s1, b_s2, w_time, 
        n_freq, b_s1_f, b_s2_f, b_g12_f, b_g12_s, b_g1_s, b_g2_s, mask, 
        n_freq_up, b_f_up, b_C_up, 
        plan_fwd, plan_inv)

    return ws
end


function compute_psr(idx::Int, max_corr::AbstractFloat, ws::FSGCC_ws; exc_th::Real=0.1)
    
    second_peak = 1e-12

    exclusion  = Int(round(exc_th * ws.fs * (ws.n_up / ws.n)))
    excl_start = mod1(idx - exclusion, ws.n_up)
    excl_end   = mod1(idx + exclusion, ws.n_up)
    
    if excl_start < excl_end
        # CASO A: La exclusión es un bloque central [ ... XXX ... ]
        
        # Tramo Izquierdo
        @inbounds for i in 1:(excl_start - 1)
            val = abs(ws.C_up_time[i])
            if val > second_peak; second_peak = val; end
        end
        
        # Tramo Derecho
        @inbounds for i in (excl_end + 1):ws.n_up
            val = abs(ws.C_up_time[i])
            if val > second_peak; second_peak = val; end
        end
        
    else
        # CASO B: La exclusión da la vuelta [ XX ... XX ]
        @inbounds for i in (excl_end + 1):(excl_start - 1)
            val = abs(ws.C_up_time[i])
            if val > second_peak; second_peak = val; end
        end
    end

    psr_db = 20 * log10(max_corr / second_peak)

    return psr_db
end



function compute_delay!(ws::FSGCC_ws, s1_in::AbstractVector{T}, s2_in::AbstractVector{T}) where {T<:AbstractFloat}
    
    m1 = mean(s1_in)
    m2 = mean(s2_in)

    @inbounds @simd for i in 1:ws.n
        ws.buf_s1[i] = (s1_in[i] - m1) * ws.window[i]
        ws.buf_s2[i] = (s2_in[i] - m2) * ws.window[i]
    end

    # println("Energía e1: $e1, e2: $e2, norm: $norm_factor")

    ws.buf_s1[ws.n+1:end] .= 0.0
    ws.buf_s2[ws.n+1:end] .= 0.0
    
    # FFT (RFFT) usando mul!
    mul!(ws.S1, ws.plan_fwd, ws.buf_s1)
    mul!(ws.S2, ws.plan_fwd, ws.buf_s2)
    
    # Espectro Cruzado
    @. ws.G12 = ws.S1 * conj(ws.S2)
    
    # Suavizado (Sub-bandas) - Cobos et al 2020
    smooth_spectrum!(ws.G12_smooth, ws.G12, ws.B)
    smooth_spectrum!(ws.G11_smooth, abs2.(ws.S1), ws.B)
    smooth_spectrum!(ws.G22_smooth, abs2.(ws.S2), ws.B)
    
    # Peso Gamma-PHAT
    norm_accum = zero(T)
    fisher_accum = zero(T)
    df = ws.fs / ws.n_fft
    @inbounds @simd for i in 1:ws.n_freq
        denom = sqrt(ws.G11_smooth[i] * ws.G22_smooth[i]) + 1e-12
        num   = abs(ws.G12_smooth[i])
        gamma = num / denom # Esto es |G12| / sqrt(G11*G22)

        if ws.mask[i] > 0
            f_k   = (i - 1) * df
            gamma2 = clamp(gamma^2, zero(T), one(T) - T(1e-6))
            # el factor 2 tiene en cuenta el rango de frecuencias
            fisher_accum += T(2) * (2π * f_k)^2 * gamma2 / (one(T) - gamma2) * df
        end
        
        # Ponderación PHAT adaptativa
        w = (gamma ^ ws.n_gamma) / (num + 1e-12)
        
        # Aplicar peso y máscara de una vez
        ws.G12_smooth[i] = (ws.G12_smooth[i] * w) * ws.mask[i]
        norm_accum += ws.mask[i]
    end

    # CRLB de Knapp & Carter
    T_obs = ws.n / ws.fs
    sigma_eff = (T_obs * fisher_accum > T(1e-12)) ? one(T) / sqrt(T(T_obs * fisher_accum)) : typemax(T)
    
    # Interpolación (Zero Padding correcto para RFFT)
    # Limpiamos el buffer de upsampling
    fill!(ws.C_up_freq, 0.0 + 0.0im)

    # Copiamos la parte positiva del espectro
    copyto!(ws.C_up_freq, 1, ws.G12_smooth, 1, ws.n_freq)
    
    # irfft -> salida temporal
    mul!(ws.C_up_time, ws.plan_inv, ws.C_up_freq)
    
    # Buscamos el maximo
    max_corr_raw, idx = findmax(ws.C_up_time)

    # factor de normalizacion
    # Multiplicamos por 2.0 por la simetría de la FFT
    # Dividimos por ws.n_up porque la IFFT distribuye la energía en N puntos.
    theoretical_max = (norm_accum * 2.0) / ws.n_up

    if theoretical_max > 1e-6
        cc_val = max_corr_raw / theoretical_max
        cc_val = clamp(cc_val, -1.0, 1.0)
    else
        cc_val = 0.0
    end
 
    # Peak-Sidelobe-Ratio (PSR)
    # Rango de exclusión circular (100 ms)
    psr_db = compute_psr(idx, cc_val, ws)

    # calcula delay
    lag = (idx <= ws.n_up ÷ 2) ? (idx - 1) : (idx - 1 - ws.n_up)
    delay = lag / (ws.fs * (ws.n_up / ws.n_fft))

    return delay, psr_db, sigma_eff
end


function delay_matrix(ws::FSGCC_ws, data::AbstractMatrix, fs::Real; psr_th::Real=5.0)
    
    lwin  = size(data, 1)
    N     = size(data, 2)
    DT    = fill(NaN, N, N)
    CC    = fill(NaN, N, N)
    SIGMA = fill(NaN, N, N)
    
    @views @inbounds for i in 1:N, j in i+1:N
        delay, psr_db, sigma_ij = compute_delay!(ws, data[:,j], data[:,i])
        if psr_db >= psr_th
            lag = round(Int, delay * fs)
            cc_val     = cc_overlap(data[:,j], data[:,i], lag, lwin)
            DT[j,i]    = delay
            DT[i,j]    = -delay
            CC[i,j]    =  cc_val
            CC[j,i]    =  cc_val
            SIGMA[i,j] =  sigma_ij
            SIGMA[j,i] =  sigma_ij
        end
    end
    
    return DT, CC, SIGMA
end