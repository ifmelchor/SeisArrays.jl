#!/usr/local/bin julia
# coding=utf-8

# GNU GPL v2 licenced to I. Melchor and J. Almendros 08/2022
 # TRIAS (TRIad-based Adaptive Slowness)

function trias(data::AbstractArray, x::AbstractVector, y::AbstractVector, fs::Real, args...; kwargs...)
    
    SA = SeisArray2D(x, y, data, fs)
    
    return trias(SA, args...; kwargs...)
end

function trias(S::SeisArray2D, lwin::Int, nadv::T, fmin::T, fmax::T; slowmax::T=2.5, slowint_c::T=0.1, slowint_f::T=0.01, slowfw::T=0.5, ratio_max::T=0.25, tol_tce::T=0.7, min_cc::T=0.5, psr_th::T=5.0, error_max::T=0.5) where {T<:Real}

    npts, nsta = size(S.data)

    # incia pares y triadas
    pairs, dx, dy, dd, trios = init_triads(S)
    n_pairs = length(pairs)
    n_trios = length(trios)

    # prepara los datos
    filter!(S, fmin, fmax)

    # slowness maximo
    slomax2 = slowmax*slowmax

    # definición de los mapas de lentitud 
    # (coarser)
    s_grid_c  = -slowmax:slowint_c:slowmax
    nite_c    = size(s_grid_c, 1)
    # (finer)
    rfine  = -slowfw:slowint_f:slowfw
    nite_f  = size(rfine, 1)
    # (full)
    s_grid = -slowmax:slowint_f:slowmax
    nite   = size(s_grid, 1)

    # configuración de ventanas de analisis
    step = round(Int, lwin * nadv)
    nwin = div(npts - lwin, Int(step)) + 1

    # define algunos parametros para el GCC
    B_FS  = 5
    g_GCC = 2.0 # --> SCOT method (gamma)
    df_taper_FS = 0.2
    upsample  = 20

    # resolucion mínima
    sigma_min = 1/(S.fs)

    # inicialización de Buffers
    n_threads = Threads.nthreads()
    thread_buffers = [
        init_buffers(nsta, n_pairs, n_trios, nite, nite_f, nite_c, lwin, S.fs, fmin, fmax, B_FS, g_GCC, upsample, df_taper_FS) 
    for _ in 1:n_threads
    ]

    # diccionario de salida
    dout = Dict{String, Any}()
    dout["time_s"]  = fill(NaN, nwin)
    dout["n_trios"] = fill(0, (nwin,3))
    dout["misfit"]  = Vector{Union{Matrix{Float32}, Nothing}}(undef, nwin)
    dout["trios"] = [Vector{Any}() for _ in 1:nwin]
    fill!(dout["misfit"], nothing)

    # estimacion del vector lentitud aparente
    dout["sx"] = fill(NaN, nwin)
    dout["sy"] = fill(NaN, nwin)
    dout["s_ratio"] = fill(NaN, nwin)
    dout["s_circ"]  = fill(NaN, nwin)
    dout["s_radii"]  = fill(NaN, nwin)
    dout["beam"]   = fill(NaN, nwin)
    dout["fpeak"]  = fill(NaN, nwin)
    dout["baz"]   = fill(NaN, (nwin,3))
    dout["slow"]  = fill(NaN, (nwin,3))
    dout["baz_width"]  = fill(NaN, nwin)
    dout["slow_width"] = fill(NaN, nwin)

    # metricas de calidad
    dout["delta_s"] = fill(NaN, nwin)
    dout["kappa"]   = fill(NaN, nwin)
    dout["cc_avg"]  = fill(NaN, nwin)
    dout["dte_avg"] = fill(NaN, nwin)
    dout["D_avg"]   = fill(NaN, nwin)
    dout["eta2"]    = fill(NaN, nwin)

    @views Threads.@threads for nk in 1:nwin
        # inicia el buffer
        tid = Threads.threadid()
        buf = thread_buffers[tid]
        fill!(buf.cc, -2.0)

        # define la ventana
        n0 = round(Int, 1 + lwin * nadv * (nk - 1))
        window_data = S.data[n0:n0+lwin-1, :]

        # ------------------------------------
        # 1. Establece triadas POTENCIALES
        # ------------------------------------
        k_vt   = 1
        @inbounds for t in eachindex(trios)
            trio = trios[t]
            is_valid_trio = true

            for p in (trio.p1_idx, trio.p2_idx, trio.p3_idx)
                if buf.cc[p] <= -1.5
                    i, j = pairs[p]
                    dist = dd[p]

                    signal_i = window_data[:, i]
                    signal_j = window_data[:, j]

                    # calcula delay usando FS-PHAT
                    delay  = compute_delay!(buf.fsgcc_ws, signal_j, signal_i, psr_th)

                    if  abs(delay) > (dist*slowmax)
                        # evita aliasing
                        cc_val = 0
                    else
                        # calcula coeficiente correlacion en el tiempo
                        lag = round(Int, delay * S.fs)
                        cc_val = cc_overlap(signal_j, signal_i, lag, lwin)
                        # si la correlacion fue mala, cc_val => 0.0
                    end

                    buf.dt[p] = delay
                    buf.cc[p] = cc_val
                end

                if buf.cc[p] < min_cc
                    is_valid_trio = false
                    break
                end
            end

            # 2. Validación de Cierre y Llenado de vt
            if is_valid_trio
                dt1 = buf.dt[trio.p1_idx]
                dt2 = buf.dt[trio.p2_idx]
                dt3 = buf.dt[trio.p3_idx]

                # Calcula la metrica de cierre
                # epislon = dt1 + dt2 + dt3 ≈ 0
                closure = (dt1 * trio.s1) + (dt2 * trio.s2) + (dt3 * trio.s3)

                # limite fisico
                max_tce = trio.dmax * slowmax
                if abs(closure) <= max_tce * tol_tce
                    # calcula la correlacion promedio
                    c1    = buf.cc[trio.p1_idx]
                    c2    = buf.cc[trio.p2_idx]
                    c3    = buf.cc[trio.p3_idx]
                    cc123 = (c1 + c2 + c3) / 3.0

                    # guarda valores en buffer
                    # buf.trio_error[t] = closure
                    # buf.trio_cc_avg[t] = cc123
                    # println(t, "  CCavg: ", buf.trio_cc_avg[t], "  TCE: ", closure)

                    # guardamos para el misfitmap
                    buf.vt.t[k_vt] = t
                    buf.vt.x1[k_vt] = dx[trio.p1_idx]
                    buf.vt.y1[k_vt] = dy[trio.p1_idx]
                    buf.vt.x2[k_vt] = dx[trio.p2_idx]
                    buf.vt.y2[k_vt] = dy[trio.p2_idx]
                    buf.vt.x3[k_vt] = dx[trio.p3_idx]
                    buf.vt.y3[k_vt] = dy[trio.p3_idx]
                    buf.vt.dt1[k_vt] = dt1
                    buf.vt.dt2[k_vt] = dt2
                    buf.vt.dt3[k_vt] = dt3
                    buf.vt.cc[k_vt]  = cc123
                    buf.vt.tce2[k_vt] = closure^2
                    k_vt += 1
                end
            end
        end
        nvalid0  = k_vt-1

        save = false
        if nvalid0 >= 1
            # calculamos el mapa de misfits (coarser)
            misfitmap!(buf.coarser_map, buf, nvalid0, s_grid_c, s_grid_c, slomax2)
            
            # buscamos el minimo
            idx = argmin(buf.coarser_map)
            sx0 = s_grid_c[idx[1]]
            sy0 = s_grid_c[idx[2]]

            # --------------------------------------
            # 1.2 FILTRA POR TRIADAS TEMP COHERENTES
            # --------------------------------------
            nvalid1 = clean_triads_temp!(sx0, sy0, buf, trios, nvalid0, error_max)

            if nvalid1 >= 1
                if nvalid0 != nvalid1
                    # Re-calcula el grid grueso
                    misfitmap!(buf.coarser_map, buf, nvalid1, s_grid_c, s_grid_c, slomax2)
                    idx = argmin(buf.coarser_map)
                    sx0 = s_grid_c[idx[1]]
                    sy0 = s_grid_c[idx[2]]
                end
                
                # CALCULO DEL POWER BEAM Y FPEAK
                powbeam, fpeak = beam_analysis(window_data, S.xcoord, S.ycoord, lwin, S.fs, sx0, sy0, buf)

                # ------------------------------------------
                # 1.3 FILTRA POR TRIADAS ESPACIAL COHERENTES
                # ------------------------------------------
                nvalid2, quality_metrics = clean_triads_space!(sx0, sy0, buf, trios, nvalid1, fpeak)

                if nvalid2 >= 1
                    save = true
                    error_avg, cc_avg, D_mean = quality_metrics

                    # ----------------------------------
                    # 1.4 LENTITUD FROM GRID INTERPOLADO
                    # ----------------------------------
                    if nvalid1 != nvalid2
                        # Re-calcula el grid grueso
                        misfitmap!(buf.coarser_map, buf, nvalid2, s_grid_c, s_grid_c, slomax2)
                        idxf = argmin(buf.coarser_map)
                        sxf = s_grid_c[idxf[1]]
                        syf = s_grid_c[idxf[2]]
                    else
                        sxf, syf = sx0, sy0
                    end

                    # diferencia de lentitud (coarser)
                    dsx = sx0 - sxf
                    dsy = sy0 - syf
                    delta_s = sqrt(dsx*dsx + dsy*dsy)
                    
                    # finer grid + interpolacion
                    rfx = rfine .+ sxf 
                    rfy = rfine .+ syf
                    misfitmap!(buf.finer_map, buf, nvalid2, rfx, rfy, slomax2)
                    interpolate_grids!(buf.like_map, buf.coarser_map, buf.finer_map, s_grid, s_grid_c, rfx, rfy)

                    # replace NaN to Inf
                    replace!(buf.like_map, NaN => Inf)

                    # calcula la solucion final
                    idxf = argmin(buf.like_map)
                    sxf = s_grid[idxf[1]]
                    syf = s_grid[idxf[2]]

                    # -----------------------------------
                    # 1.5 CALCULA CONSISTENCIA DE TRIADAS
                    # -----------------------------------
                    sum_sq_err = 0.0
                    @inbounds for k in 1:nvalid2
                        sx_local, sy_local = slowness_triad(buf.vt, k)
                        dsx = sx_local - sxf
                        dsy = sy_local - syf
                        triad_con = dsx*dsx + dsy*dsy
                        buf.vt.s_cons[k] = sqrt(triad_con)
                        sum_sq_err += triad_con
                    end
                    sigma_s = sqrt(sum_sq_err / nvalid2)
                    slow_f = sqrt(sxf*sxf + syf*syf)
                    kappa = exp(-sigma_s/slow_f)
                end
            end
        end

        if save
            # ---------------------------
            # 2 GUARDA RESULTADOS BASICOS
            # ---------------------------
            dout["time_s"][nk]  = (n0 - 1) / float(S.fs)
            dout["n_trios"][nk,1] = nvalid0
            dout["n_trios"][nk,2] = nvalid1
            dout["n_trios"][nk,3] = nvalid2
            
            # trios info (filtro temporal)
            @inbounds for k in 1:nvalid1
                t_orig = buf.vt.t[k]
                trio = trios[t_orig]
                trios_stats = (trio.sta_triad, buf.vt.cc[k], sqrt(buf.vt.tce2[k]), buf.vt.spatial_res[k], buf.vt.s_cons[k])
                push!(dout["trios"][nk], trios_stats)
            end
            
            dout["misfit"][nk] = copy(buf.like_map)
            dout["fpeak"][nk] = fpeak
            dout["beam"][nk]  = powbeam
            dout["delta_s"][nk] = delta_s
            dout["sx"][nk] = sxf
            dout["sy"][nk] = syf

            # ---------------------------
            # 2.1 METRICAS DE CALIDAD
            # ---------------------------
            dout["kappa"][nk] = kappa
            dout["cc_avg"][nk]  = cc_avg
            dout["dte_avg"][nk] = error_avg
            dout["D_avg"][nk] = D_mean

            # eta2
            eta2 = kappa * sigmoid(-error_avg; x0=-1.) * sigmoid(D_mean; x0=0.5)
            dout["eta2"][nk] = eta2

            # ----------------------
            # 3 PSEUDO-VEROSIMILITUD
            # ----------------------
            @. buf.like_map = exp(-sqrt(nvalid2 * eta2) * buf.like_map / sigma_min)
            
            # nivel de una desviacion estandar
            level = maximum(buf.like_map) / ℯ

            # ----------------------
            # 3.1 INCERTIDUMBRE
            # ----------------------
            is_good, uncert = uncertainty_contour(s_grid, s_grid, buf.like_map, level, ratio_max)
            dout["s_ratio"][nk] = uncert[1]
            dout["s_circ"][nk]  = uncert[2] # circularidad del contorno de íncertidumbre

            if is_good
                radii, slobnd, bazbnd = uncert[3], uncert[4], uncert[5]
                dout["s_radii"][nk] = radii
                # existe un unico contorno claro de solucion
                # BAZ min, BAZ central, BAZ max
                dout["baz"][nk,1] = bazbnd[1]
                dout["baz"][nk,2] = bazbnd[2]
                dout["baz"][nk,3] = bazbnd[3]
                dout["baz_width"][nk] = bazbnd[4]

                # SLOW min, SLOW central, SLOW max
                dout["slow"][nk,1] = slobnd[1]
                dout["slow"][nk,2] = slobnd[2]
                dout["slow"][nk,3] = slobnd[3]
                dout["slow_width"][nk] = slobnd[4]
            end
        end
    end

    mask = findall(!isnan, dout["time_s"])

    if isempty(mask)
        return nothing
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

    return final_dout
end


function misfitmap!(like_map::AbstractMatrix{T}, buf::ThreadBuffers, nvtrios::Int, s_grid_x::AbstractVector{T}, s_grid_y::AbstractVector{T}, slomax2::T) where {T<:AbstractFloat}

    nx = length(s_grid_x)
    ny = length(s_grid_y)

    inv_2 = T(0.5)
    sigma_val = buf.sigma
    min_weight = T(1e-6)

    fill!(like_map, zero(T))

    @inbounds for j in 1:ny
        sy  = s_grid_y[j]
        sy2 = sy * sy

        for i in 1:nx
            sx = s_grid_x[i]

            if (sx*sx + sy2) > slomax2
                like_map[i, j] = typemax(T)
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

                # tiempo maximo de la triada
                tmax = inv_2 * (abs(dt_t1) + abs(dt_t2) + abs(dt_t3))
                tmax += sigma_val
                tmax *= tmax

                # funcion peso
                w_val = buf.vt.cc[k] * exp(-buf.vt.tce2[k] / tmax)

                W_sum += w_val
                R_sum += w_val * e_trio
            end

            like_map[i, j] = R_sum / W_sum
        end
    end
end


function clean_triads_temp!(best_sx, best_sy, buf, trios, n_valid_trios::Int, error_max::T) where {T<:Real}
    
    k_vt_new = 1
    inv_2 = T(0.5)
    sigma_val = buf.sigma
    stamask = buf.station_mask

    # Reseteamos la máscara de estaciones al inicio
    fill!(stamask, false)

    @inbounds for k in 1:n_valid_trios
        # Extraemos las distancias de los lados de la tríada
        x1, y1 = buf.vt.x1[k], buf.vt.y1[k]
        x2, y2 = buf.vt.x2[k], buf.vt.y2[k]
        x3, y3 = buf.vt.x3[k], buf.vt.y3[k]
        
        # Calculamos el tiempo de tránsito direccional para cada lado
        t1 = abs(x1 * best_sx + y1 * best_sy)
        t2 = abs(x2 * best_sx + y2 * best_sy)
        t3 = abs(x3 * best_sx + y3 * best_sy)
        tau_max = inv_2 * (t1 + t2 + t3)
        
        closure_error = sqrt(buf.vt.tce2[k])
        time_error = closure_error / (tau_max + sigma_val)
        
        # Si pasa el filtro, la mantenemos en el buffer
        if time_error <= error_max
            t_orig = buf.vt.t[k]
            trio = trios[t_orig]

            # activa las estaciones para el power beam
            for sta_idx in trio.sta_triad
                stamask[sta_idx] = true
            end

            # Desplazamos los datos en el buffer
            if k_vt_new != k
                buf.vt.t[k_vt_new]  = t_orig
                buf.vt.x1[k_vt_new] = x1
                buf.vt.y1[k_vt_new] = y1
                buf.vt.x2[k_vt_new] = x2
                buf.vt.y2[k_vt_new] = y2
                buf.vt.x3[k_vt_new] = x3
                buf.vt.y3[k_vt_new] = y3
                buf.vt.dt1[k_vt_new] = buf.vt.dt1[k]
                buf.vt.dt2[k_vt_new] = buf.vt.dt2[k]
                buf.vt.dt3[k_vt_new] = buf.vt.dt3[k]
                buf.vt.cc[k_vt_new] = buf.vt.cc[k]
                buf.vt.tce2[k_vt_new] = buf.vt.tce2[k]
            end

            k_vt_new += 1
        end
    end

    return k_vt_new-1
end


function clean_triads_space!(best_sx, best_sy, buf, trios, n_valid_trios::Int, f_peak::T) where {T<:Real}

    k_vt_new = 1
    inv_2 = T(0.5)
    sigma_val = buf.sigma

    error_avg = T(0.0)
    cc_avg    = T(0.0)
    D_avg     = T(0.0)

    @inbounds for k in 1:n_valid_trios
        t_orig = buf.vt.t[k]
        trio = trios[t_orig]
        
        # tau_max con best_s
        t1 = abs(buf.vt.x1[k] * best_sx + buf.vt.y1[k] * best_sy)
        t2 = abs(buf.vt.x2[k] * best_sx + buf.vt.y2[k] * best_sy)
        t3 = abs(buf.vt.x3[k] * best_sx + buf.vt.y3[k] * best_sy)

        tau_max = inv_2 * (t1 + t2 + t3)
        D_norm = tau_max * f_peak

        # FILTRO ESPACIAL
        if D_norm > 0.5
            D_avg += D_norm

            closure_error = sqrt(buf.vt.tce2[k])
            error_avg += closure_error/(tau_max + sigma_val)
            cc_avg    += buf.vt.cc[k]

            # Desplazamos los datos en el buffer
            if k_vt_new != k
                buf.vt.t[k_vt_new]  = t_orig
                buf.vt.x1[k_vt_new] = buf.vt.x1[k]
                buf.vt.y1[k_vt_new] = buf.vt.y1[k]
                buf.vt.x2[k_vt_new] = buf.vt.x2[k]
                buf.vt.y2[k_vt_new] = buf.vt.y2[k]
                buf.vt.x3[k_vt_new] = buf.vt.x3[k]
                buf.vt.y3[k_vt_new] = buf.vt.y3[k]
                buf.vt.dt1[k_vt_new] = buf.vt.dt1[k]
                buf.vt.dt2[k_vt_new] = buf.vt.dt2[k]
                buf.vt.dt3[k_vt_new] = buf.vt.dt3[k]
                buf.vt.cc[k_vt_new] = buf.vt.cc[k]
                buf.vt.tce2[k_vt_new] = buf.vt.tce2[k]
                buf.vt.spatial_res[k_vt_new] = D_norm
            end
            k_vt_new += 1
        end
    end

    n_valid_trios2 = k_vt_new-1

    if n_valid_trios2 >= 1
        return n_valid_trios2, (error_avg/n_valid_trios2, cc_avg/n_valid_trios2, D_avg/n_valid_trios2)
    else
        return n_valid_trios2, (T(NaN), T(NaN), T(NaN))
    end
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


function init_buffers(nsta, n_pairs, n_trios, nite, nite_f, nite_c, lwin, fs, fmin, fmax, B, n_gamma, upsample, df_taper)

    T = Float64

    # Inicializa FS-PHAT
    ws = init_fsgcc(lwin, fs, fmin, fmax, B, n_gamma, upsample, df_taper)

    # Buffers de correlación
    dt = zeros(T, n_pairs)
    cc = zeros(T, n_pairs)

    # Buffers de tríos
    trio_error  = zeros(T, n_trios)
    trio_cc_avg = zeros(T, n_trios)
    trio_w = zeros(T, n_trios)

    # Buffers de likelihood
    like_map = zeros(T, nite, nite)
    finer_map   = zeros(T, nite_f, nite_f)
    coarser_map = zeros(T, nite_c, nite_c)

    # sigma teorico
    sigma = one(T)/fs

    # beam analysis
    station_mask = falses(nsta)
    station_lags = zeros(Int, nsta)
    windows = haning_windows(lwin, T)
    beam = zeros(T, lwin)
    beam_window_fft = zeros(Complex{T}, lwin)

    vt = ValidTrios(
        Vector{Int}(undef, n_trios), # t
        Vector{T}(undef, n_trios), Vector{T}(undef, n_trios), # x1, y1
        Vector{T}(undef, n_trios), Vector{T}(undef, n_trios), # x2, y2
        Vector{T}(undef, n_trios), Vector{T}(undef, n_trios), # x3, y3
        Vector{T}(undef, n_trios), # dt1
        Vector{T}(undef, n_trios), # dt2
        Vector{T}(undef, n_trios), # dt3
        Vector{T}(undef, n_trios), # cc
        Vector{T}(undef, n_trios), # tce2
        fill(T(NaN), n_trios), # slow_consistency
        fill(T(NaN), n_trios) # spatial_res
    )

    buff = ThreadBuffers(ws, dt, cc, trio_error, trio_cc_avg, trio_w, vt, like_map, finer_map, coarser_map, sigma, station_mask, station_lags, windows, beam, beam_window_fft)
    
    return buff
end


function beam_analysis(data::AbstractArray{T}, xSta::AbstractVector{T}, ySta::AbstractVector{T}, lwin::Int, fsem::Real, sx::Real, sy::Real, buf::ThreadBuffers) where {T<:AbstractFloat}

    station_mask = buf.station_mask
    station_lags = buf.station_lags
    beam = buf.beam
    fft_ws = buf.beam_window_fft

    # Calcula el centroide del subarray
    sum_x, sum_y, n_active = 0.0, 0.0, 0
    @inbounds for k in eachindex(station_mask)
        if station_mask[k]
            sum_x += xSta[k]
            sum_y += ySta[k]
            n_active += 1
        end
    end
    ref_x, ref_y = sum_x / n_active, sum_y / n_active

    min_lag, max_lag = typemax(Int), typemin(Int)
    @inbounds for k in eachindex(station_mask)
        if station_mask[k]
            dx = xSta[k] - ref_x
            dy = ySta[k] - ref_y
            lag = round(Int, -(sx*dx + sy*dy) * fsem)
            station_lags[k] = lag
            min_lag = ifelse(lag < min_lag, lag, min_lag)
            max_lag = ifelse(lag > max_lag, lag, max_lag)
        end
    end

    t0 =    1 + max(0, -min_lag)
    t1 = lwin - max(0,  max_lag)

    if t1 < t0
        return 0.0, 0.0
    end

     # Limpia solo la porción útil del buffer
    n_samples = t1 - t0 + 1
    @inbounds for idx in 1:n_samples
        beam[idx] = zero(T)
    end

    # suma amplitudes
    @inbounds for k in eachindex(station_mask)
        if station_mask[k]
            lag = station_lags[k]
            for idx in 1:n_samples
                beam[idx] += data[t0 + idx - 1 + lag, k]
            end
        end
    end

    # calcula potenia
    power_sum = 0.0
    @inbounds for idx in 1:n_samples
        val = beam[idx]
        power_sum += val*val
    end
    norm_factor = T(n_active * n_active) * T(n_samples)
    beam_power = power_sum / norm_factor

     # Aplicar el FFT usando tappering
    taper_window = buf.beam_window[n_samples]
    @inbounds for i in 1:n_samples
        fft_ws[i] = beam[i] * taper_window[i] + 0im
    end
    fft!(view(fft_ws, 1:n_samples))

    # Encontrar pico
    max_power = 0.0
    idx_max = 1
    half_n = div(n_samples, 2) + 1

    @inbounds for i in 1:half_n
        power = abs2(fft_ws[i])
        if power > max_power
            max_power = power
            idx_max = i
        end
    end

    fpeak = (idx_max - 1) * fsem / n_samples

    return power_sum/norm_factor, fpeak
end


function slowness_triad(vt::ValidTrios, k::Int)
    # Diferencias de coordenadas
    dx1 = vt.x1[k]; dy1 = vt.y1[k]
    dx2 = vt.x2[k]; dy2 = vt.y2[k]
    dx3 = vt.x3[k]; dy3 = vt.y3[k]
    
    # Delays
    dt1 = vt.dt1[k]
    dt2 = vt.dt2[k]
    dt3 = vt.dt3[k]

    # Términos para Mínimos Cuadrados (M'M)s = M'd
    # M = [dx1 dy1; dx2 dy2; dx3 dy3]
    mtm_11 = dx1*dx1 + dx2*dx2 + dx3*dx3
    mtm_12 = dx1*dy1 + dx2*dy2 + dx3*dy3
    mtm_22 = dy1*dy1 + dy2*dy2 + dy3*dy3
    mtd_1  = dx1*dt1 + dx2*dt2 + dx3*dt3
    mtd_2  = dy1*dt1 + dy2*dt2 + dy3*dt3
    det = mtm_11 * mtm_22 - mtm_12 * mtm_12
    
    inv_det = 1.0 / det
    sx = inv_det * ( mtm_22 * mtd_1 - mtm_12 * mtd_2 )
    sy = inv_det * (-mtm_12 * mtd_1 + mtm_11 * mtd_2 )
    
    return sx, sy
end

