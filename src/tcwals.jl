#!/usr/local/bin julia
# coding=utf-8

# GNU GPL v2 licenced to I. Melchor and J. Almendros 08/2022
 # TC-WALS (Time-Closure Weighted Adaptive Likelihood Slowness)

function tcwals(data::AbstractArray, x::AbstractVector, y::AbstractVector, fs::Real, args...; kwargs...)
    
    SA = SeisArray2D(x, y, data, fs)
    
    return tcwals(SA, args...; kwargs...)
end

function tcwals(S::SeisArray2D, lwin::Int, nadv::T, fmin::T, fmax::T; slowmax::T=2.0, slowint::T=0.02, ccerr::T=0.95, ratio_max::T=0.25, max_tce::T=0.5, min_cc::T=0.5, min_trio::Int=5, psr_th::T=5.0, upsample::Int=20, B_PHAT::Int=5, g_PHAT::T=2.0, df_taper_PHAT::T=0.2, score_min::T=5.0, gamma_L::T=2.0, lambda_L::Union{T, Nothing}=nothing, stack::Bool=false, baz_th::Real=20.0, baz_lim::Union{Vector{<:Real}, Nothing}=nothing) where {T<:Real}

    npts, nsta = size(S.data)
    filter!(S, fmin, fmax)

    # definición del mapa de lentitud
    s_grid  = -slowmax:slowint:slowmax
    slomax2 = slowmax*slowmax
    nite    = size(s_grid, 1)

    # configuración de ventanas de analisis
    step = round(Int, lwin * nadv)
    nwin = div(npts - lwin, Int(step)) + 1

    # geometría de pares y triadas
    pairs, trios = init_triads(nsta)
    n_pairs = length(pairs)
    n_trios = length(trios)
    dx, dy  = cross_pair_dist(S, pairs)

    @assert n_trios > min_trio

    # inicializa la funcion peso para la Likelihood
    eps = 1/(S.fs*upsample)
    Teff2 = compute_Teff(trios, s_grid, dx, dy, eps)

    # inicialización de Buffers
    n_threads = Threads.nthreads()
    thread_buffers = [
        init_buffers(n_pairs, n_trios, nite, lwin, S.fs, fmin, fmax, B_PHAT, g_PHAT, upsample, df_taper_PHAT) 
    for _ in 1:n_threads
    ]

    # diccionario de salida
    dout = Dict{String, Any}()
    dout["time_s"]  = fill(NaN, nwin)
    dout["n_trios"] = fill(NaN, nwin)
    dout["cc_avg"]  = fill(NaN, nwin)
    dout["likemap"] = fill(Float32(NaN), (nwin, nite, nite))
    dout["lambda"]  = fill(NaN, nwin)
    dout["lmax"] = fill(NaN, nwin)
    dout["trios"] = [Tuple{Int, Int, Int}[] for _ in 1:nwin]
    dout["sx"] = fill(NaN, nwin)
    dout["sy"] = fill(NaN, nwin)
    dout["ratio"]   = fill(NaN, nwin)
    dout["baz"]     = fill(NaN, (nwin,3))
    dout["slow"]    = fill(NaN, (nwin,3))
    dout["baz_width"]  = fill(NaN, nwin)
    dout["slow_width"] = fill(NaN, nwin)
    dout["rms"]   = fill(NaN, nwin)

    @views Threads.@threads for nk in 1:nwin
        # define la ventana
        n0 = round(Int, 1 + lwin * nadv * (nk - 1))
        window_data = S.data[n0:n0+lwin-1, :]

        # inicia el buffer
        tid = Threads.threadid()
        buf = thread_buffers[tid]

        fill!(buf.cc, -2.0)
        fill!(buf.trio_flags, true)

        # calcula las correlaciones de las triadas
        cc_avg = 0.0
        @inbounds for t in eachindex(trios)
            trio = trios[t]
            for p in (trio.p1_idx, trio.p2_idx, trio.p3_idx)
                if buf.cc[p] <= -1.5
                    i, j = pairs[p]

                    signal_i = window_data[:, i]
                    signal_j = window_data[:, j]

                    # calcula delay usando FS-PHAT
                    delay  = compute_delay!(buf.fsgcc_ws, signal_j, signal_i, psr_th)
                    
                    # calcula coeficiente correlacion en el tiempo
                    lag = round(Int, delay * S.fs)
                    cc_val = cc_overlap(signal_j, signal_i, lag, lwin)
                    # si la correlacion fue mala, cc_val => 0.0

                    buf.dt[p] = delay
                    buf.cc[p] = cc_val
                end

                if buf.cc[p] < min_cc
                    buf.trio_flags[t] = false
                    break
                end
            end

            if buf.trio_flags[t]
                # Calcular cierre: dt1 + dt2 + dt3 ≈ 0
                dt1 = buf.dt[trio.p1_idx]
                dt2 = buf.dt[trio.p2_idx]
                dt3 = buf.dt[trio.p3_idx]
                closure = (dt1 * trio.s1) + (dt2 * trio.s2) + (dt3 * trio.s3)

                if abs(closure) <= max_tce
                    buf.trio_error[t] = closure

                    # calcula la correlacion promedio
                    c1  = buf.cc[trio.p1_idx]
                    c2  = buf.cc[trio.p2_idx]
                    c3  = buf.cc[trio.p3_idx]
                    cc123 = (c1 + c2 + c3) / 3.0
                    buf.trio_cc_avg[t] = cc123
                    cc_avg += cc123
                    # println(t, "  CCavg: ", buf.trio_cc_avg[t], "  TCE: ", closure)
                else
                    buf.trio_flags[t] = false
                end
            end
        end

        n_valid_trios  = count(buf.trio_flags)

        if n_valid_trios >= min_trio
            # calculamos el mapa de misfits
            misfitmap!(buf, trios, dx, dy, s_grid, nite, slomax2, Teff2, gamma_L)
            best_misfit = minimum(buf.like_map)
            if best_misfit < 2.0 # s
                dout["time_s"][nk]  = (n0 - 1) / float(S.fs)
                dout["n_trios"][nk] = n_valid_trios
                dout["cc_avg"][nk]  = cc_avg/n_valid_trios
                dout["lambda"][nk]  = best_misfit
                dout["likemap"][nk, :, :] .= buf.like_map
                # save trios
                @inbounds for t in eachindex(trios)
                    if buf.trio_flags[t]
                        push!(dout["trios"][nk], trios[t].sta_triad)
                    end
                end
            end
        end
    end

    mask = findall(!isnan, dout["time_s"])

    if isempty(mask)
        return nothing
    end

    # calcula el lambda para la likelihood
    if isnothing(lambda_L) || lambda_L <= zero(T)
        s_noise_floor = quantile(dout["lambda"][mask], 0.2)
        lambda_global = 1.0 / max(s_noise_floor, 1e-4)
        println("Auto-Lambda calculado: ", lambda_global)
    else
        lambda_global = lambda_L
    end

    # define los buffers para las estaciones
    station_mask = [falses(nsta) for _ in 1:n_threads]
    station_lags = [zeros(Int, nsta) for _ in 1:n_threads]

    @views Threads.@threads for nk in mask
        likemap = dout["likemap"][nk, :, :]
        @. likemap = exp(-lambda_global * likemap)

        likemax = maximum(likemap)

        if likemax > 0.1
            is_good, ratio, s_c, slobnd, bazbnd = uncertainty_contour(s_grid, s_grid, likemap, likemax*ccerr, ratio_max)
            dout["lmax"][nk]  = likemax
            dout["ratio"][nk] = ratio

            if is_good
                # guardamos el resultado final
                dout["sx"][nk] = s_c[1]
                dout["sy"][nk] = s_c[2]
                dout["baz"][nk,1] = bazbnd[1]
                dout["baz"][nk,2] = bazbnd[2]
                dout["baz"][nk,3] = bazbnd[3]
                dout["slow"][nk,1] = slobnd[1]
                dout["slow"][nk,2] = slobnd[2]
                dout["slow"][nk,3] = slobnd[3]
                dout["baz_width"][nk] = bazbnd[4]
                dout["slow_width"][nk] = slobnd[4]

                # calculamos el beam power de las estaciones activas
                n0 = round(Int, 1 + lwin * nadv * (nk - 1))
                window_data = S.data[n0:n0+lwin-1, :]

                # fill station mask
                tid = Threads.threadid()
                buf = thread_buffers[tid]
                stamask = station_mask[tid]
                lagmask = station_lags[tid]
                fill!(stamask, false)
                @inbounds for (i,j,k) in dout["trios"][nk]
                    stamask[i] = true
                    stamask[j] = true
                    stamask[k] = true
                end

                dout["rms"][nk] = beam_power(window_data, S.xcoord, S.ycoord, lwin, S.fs, s_c[1], s_c[2], stamask, lagmask)
            end
        end
    end

    mask = findall(!isnan, dout["sx"])

    if stack
        stack_dout = wals_stack(dout, mask, s_grid, baz_th, baz_lim, ccerr, ratio_max)
    end

    final_dout = Dict{String, Any}()
    for key in keys(dout)
        val = dout[key]
        if ndims(val) == 1
            final_dout[key] = val[mask]
        elseif ndims(val) == 2
            final_dout[key] = val[mask, :]
        elseif ndims(val) == 3
            final_dout[key] = val[mask, :, :]
        end
    end

    # limpia la memoria
    thread_buffers = nothing
    GC.gc()

    if stack
        return final_dout, stack_dout
    else
        return final_dout
    end
end


function misfitmap!(buf::ThreadBuffers, triangles::Vector{TriangleDef}, dx, dy, s_grid::AbstractVector{T}, nite::Int, slomax2::T, teff2::AbstractArray{T,3}, gamma::T=2.0) where {T<:AbstractFloat}

    # inicializamos la funcion peso
    k = 1
    @inbounds for t in eachindex(triangles)
        if buf.trio_flags[t]
            trio = triangles[t]

            # Copia de coordenadas
            buf.vt.x1[k] = dx[trio.p1_idx]
            buf.vt.y1[k] = dy[trio.p1_idx]
            buf.vt.x2[k] = dx[trio.p2_idx]
            buf.vt.y2[k] = dy[trio.p2_idx]
            buf.vt.x3[k] = dx[trio.p3_idx]
            buf.vt.y3[k] = dy[trio.p3_idx]

            # Copia de delays observados
            buf.vt.dt1[k] = buf.dt[trio.p1_idx]
            buf.vt.dt2[k] = buf.dt[trio.p2_idx]
            buf.vt.dt3[k] = buf.dt[trio.p3_idx]

            # Pre-cálculo de términos constantes del peso
            buf.vt.w_base[k] = buf.trio_cc_avg[t]^gamma 
            buf.vt.err_sq[k] = buf.trio_error[t]^2

            # Guardar ID original
            buf.vt.id[k] = t
            k += 1
        end
    end
    nvtrios = k-1

    @inbounds for j in 1:nite
        sy  = s_grid[j]
        sy2 = sy * sy

        for i in 1:nite
            sx = s_grid[i]

            if (sx*sx + sy2) > slomax2
                buf.like_map[i, j] = typemax(T)
                continue
            end

            R_sum = zero(T)
            W_sum = zero(T)
            @simd ivdep for k in 1:nvtrios
                
                # delay teoricos
                dt_t1 = sx * buf.vt.x1[k] + sy * buf.vt.y1[k]
                dt_t2 = sx * buf.vt.x2[k] + sy * buf.vt.y2[k]
                dt_t3 = sx * buf.vt.x3[k] + sy * buf.vt.y3[k]
                    
                # residuos norma L1
                e_trio = abs(dt_t1 - buf.vt.dt1[k]) + 
                         abs(dt_t2 - buf.vt.dt2[k]) + 
                         abs(dt_t3 - buf.vt.dt3[k])

                # peso
                tidx = buf.vt.id[k]
                w_val = buf.vt.w_base[k] * exp(-buf.vt.err_sq[k] * teff2[tidx,i,j])

                W_sum += w_val
                R_sum += w_val * e_trio
            end

            misfit_val = (W_sum > 1e-6) ? (R_sum / W_sum) : typemax(T)
            buf.like_map[i, j] = misfit_val
        end
    end
end


function compute_Teff(triangles::Vector{TriangleDef}, s_grid::AbstractVector{T}, dx, dy, eps::T) where {T<:AbstractFloat}

    nite  = length(s_grid)
    ntrio = length(triangles)
    Teff2 = zeros(nite, nite, ntrio)
    @inbounds for t in eachindex(triangles)
        trio = triangles[t]

        for j in 1:nite
            sy  = s_grid[j]
            
            @simd ivdep for i in 1:nite
                sx = s_grid[i]

                dt1 = abs(sx * dx[trio.p1_idx] + sy * dy[trio.p1_idx])
                dt2 = abs(sx * dx[trio.p2_idx] + sy * dy[trio.p2_idx])
                dt3 = abs(sx * dx[trio.p3_idx] + sy * dy[trio.p3_idx])
                app = (dt1 + dt2 + dt3) / 2.0

                Teff2[i,j,t] = 1.0 / (app+eps)^2
            end
        end
    end

    return permutedims(Teff2, (3, 1, 2))
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


function compute_delay!(ws::FSGCC_ws, s1_in::AbstractVector{T}, s2_in::AbstractVector{T}, psr_th::Real) where {T<:AbstractFloat}
    
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
    @inbounds @simd for i in 1:ws.n_freq
        denom = sqrt(ws.G11_smooth[i] * ws.G22_smooth[i]) + 1e-12
        num   = abs(ws.G12_smooth[i])
        gamma = num / denom # Esto es |G12| / sqrt(G11*G22)
        
        # Ponderación PHAT adaptativa
        w = (gamma ^ ws.n_gamma) / (num + 1e-12)
        
        # Aplicar peso y máscara de una vez
        ws.G12_smooth[i] = (ws.G12_smooth[i] * w) * ws.mask[i]
        norm_accum += ws.mask[i]
    end
    
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
    # Multiplicamos por 2.0 por la simetría de la FFT (frecuencias negativas implícitas)
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

    if psr_db < psr_th
        delay  = 0.0
    else
        # calcula delay
        lag = (idx <= ws.n_up ÷ 2) ? (idx - 1) : (idx - 1 - ws.n_up)
        delay = lag / (ws.fs * (ws.n_up / ws.n_fft))
    end

    return delay
end


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


function init_buffers(n_pairs, n_trios, nite, lwin, fs, fmin, fmax, B, n_gamma, upsample, df_taper)
    # Inicializa FS-PHAT
    ws = init_fsgcc(lwin, fs, fmin, fmax, B, n_gamma, upsample, df_taper)
    
    dt = zeros(n_pairs)
    cc = zeros(n_pairs)

    trio_flags  = falses(n_trios)
    trio_error  = zeros(n_trios)
    trio_cc_avg = zeros(n_trios)
    trio_w = zeros(n_trios)
    like_map = zeros(nite, nite)

    # esto es para el simd
    vt = ValidTrios(
        zeros(n_trios), zeros(n_trios), # x1, y1
        zeros(n_trios), zeros(n_trios), # x2, y2
        zeros(n_trios), zeros(n_trios), # x3, y3
        zeros(n_trios), # dt1
        zeros(n_trios), # dt2
        zeros(n_trios), # dt3
        zeros(n_trios),
        zeros(n_trios),
        zeros(Int, n_trios)
    )

    buff = ThreadBuffers(ws, dt, cc, trio_flags, trio_error, trio_cc_avg, trio_w, vt, like_map)
    return buff
end


function beam_power(data::AbstractArray{T}, xSta::AbstractVector{T}, ySta::AbstractVector{T}, lwin::Int, fsem::Real, sx::Real, sy::Real, station_mask::BitVector, station_lags::AbstractVector{Int}) where {T<:AbstractFloat}

    # xsta y ysta tienen que ser coordenadas en relacion al centro

    n_active = count(station_mask)
    sum_x = 0.0
    sum_y = 0.0
    @inbounds for k in eachindex(station_mask)
        if station_mask[k]
            sum_x += xSta[k]
            sum_y += ySta[k]
        end
    end

    ref_x = sum_x / n_active
    ref_y = sum_y / n_active

    min_lag = 0
    max_lag = 0
    first_pass = true

    @inbounds for k in eachindex(station_mask)
        if station_mask[k]
            dx = xSta[k] - ref_x
            dy = ySta[k] - ref_y

            delay_sec = -(sx * dx + sy * dy)
            lag = round(Int, delay_sec * fsem)
            station_lags[k] = lag
            
            if first_pass
                min_lag = lag
                max_lag = lag
                first_pass = false
            else
                if lag < min_lag; min_lag = lag; end
                if lag > max_lag; max_lag = lag; end
            end
        end
    end

    t0 =    1 + max(0, -min_lag)
    t1 = lwin - max(0,  max_lag)

    if t1 <= t0
        return 0.0
    end

    power_sum = 0.0
    @inbounds for t in t0:t1
        beam_amp = 0.0
        for k in eachindex(station_mask)
            if station_mask[k]
                beam_amp += data[t + station_lags[k], k]
            end
        end
        power_sum += beam_amp^2
    end

    norm_factor = (t1 - t0 + 1) * (n_active^2)
    
    return power_sum / norm_factor
end


function wals_stack(dout::Dict, mask::Vector{<:Integer}, s_grid::AbstractVector{T}, baz_th::Real, baz_lim::Union{Vector{<:Real}, Nothing}=nothing, ccerr::Real=0.95, ratio_max::Real=0.05) where {T<:AbstractFloat}

    raw_maac = dout["lmax"][mask]
    raw_time = dout["time_s"][mask]
    raw_rms  = dout["rms"][mask]
    raw_baz  = dout["baz"][mask,2]
    raw_slow = dout["slow"][mask,2]
    raw_baz_width = dout["baz_width"][mask]
    raw_lmap  = dout["likemap"][mask,:,:]

    N_total = length(dout["time_s"])

    if length(raw_time) == 0
        return nothing
    end
    
    # Calcular máscara de Stacking
    cond_baz_w = raw_baz_width .<= baz_th

    if baz_lim !== nothing
        bmin, bmax = baz_lim
        if bmin <= bmax
            cond_baz_lim = (raw_baz .>= bmin) .& (raw_baz .<= bmax)
        else
            cond_baz_lim = (raw_baz .>= bmin) .| (raw_baz .<= bmax)
        end
        mask_stack = cond_baz_w .& cond_baz_lim
    else
        mask_stack = cond_baz_w
    end

    nidx = findall(mask_stack)
    count_stack = length(nidx)
    prct_nidx = 100 * count_stack / N_total

    stack_out = Dict{String, Any}()
    stack_out["prct_nidx"] = prct_nidx
    stack_out["nidx"]      = nidx
    stack_out["time_s"]    = raw_time[nidx]

    # statistics
    stack_out["stats"] = Dict{String, Any}()
    rms_vec  = raw_rms[nidx]
    slow_vec = raw_slow[nidx]
    baz_vec  = raw_baz[nidx]

    w = raw_maac[nidx] # usamos el Lmax como peso
    sum_w  = sum(w)
    slow_avg = sum(slow_vec .* w) / sum_w
    stack_out["stats"]["slow_avg"] = slow_avg
    slow_var = sum(w .* (slow_vec .- slow_avg).^2) / sum_w
    stack_out["stats"]["slow_std"] = sqrt(slow_var)
    baz_avg, baz_std = circular_stats(baz_vec, w)
    stack_out["stats"]["baz_avg"]  = baz_avg
    stack_out["stats"]["baz_std"]  = baz_std
    stack_out["stats"]["rms_mean"] = sum(rms_vec .* w) / sum_w

    # slowmap Weighted Stack
    raw_maps = raw_lmap[nidx, :, :]
    w_stack = dropdims(sum(raw_maps .* reshape(w, :, 1, 1), dims=1), dims=1) ./ sum_w
    stack_out["stackmap"] = w_stack

    lmax = maximum(w_stack)
    stack_out["lmax"] = lmax

    is_good, ratio, s_c, slobnd, bazbnd = uncertainty_contour(s_grid, s_grid, w_stack, lmax * ccerr, ratio_max)
    stack_out["ratio"] = ratio

    stack_out["sx"]  = nothing
    stack_out["sy"]  = nothing
    stack_out["baz"]   = nothing
    stack_out["slow"]  = nothing
    stack_out["baz_width"]  = nothing
    stack_out["slow_width"] = nothing

    if is_good
        # save apparent slowness
        stack_out["sx"]  = s_c[1]
        stack_out["sy"]  = s_c[2]
        stack_out["baz"] = [bazbnd[1], bazbnd[2], bazbnd[3]]
        stack_out["baz_width"] = bazbnd[4]
        stack_out["slow"] = [slobnd[1], slobnd[2], slobnd[3]]
        stack_out["slow_width"] = slobnd[4]
    end

    return stack_out
end