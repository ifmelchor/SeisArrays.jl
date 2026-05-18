#!/usr/local/bin julia
# coding=utf-8

# GNU GPL v2 licenced to I. Melchor and J. Almendros 08/2022
 # TRIAS (TRIad-based Adaptive Slowness)

struct TriadParams{T<:AbstractFloat, J<:Integer}
    t::Vector{J}
    x1::Vector{T}
    y1::Vector{T}
    x2::Vector{T}
    y2::Vector{T}
    x3::Vector{T}
    y3::Vector{T}
    dt1::Vector{T}
    dt2::Vector{T}
    dt3::Vector{T}
    cc_avg::Vector{T}
    closure::Vector{T}
    sigma::Vector{T}
    apert_eff::Vector{T} # apertura efectiva [km]
    sx::Vector{T}      # local sx component [s/km]
    sy::Vector{T}      # local sy component [s/km]
    delta_s2:: Vector{T} # cuadrado de la distancia euclidiana con respecto a global sx, sy 
end


struct ThreadBuffers{T<:AbstractFloat}
    fsgcc_ws    :: FSGCC_ws 
    dt          :: Vector{T}
    cc          :: Vector{T}
    sigma       :: Vector{T}

    trios       :: TriadParams
    
    like_map    :: Matrix{T}
    finer_map   :: Matrix{T}
    coarser_map :: Matrix{T}

    station_mask::BitVector
    station_lags::Vector{Int}
    beam_window::Vector{Vector{T}}
    beam::Vector{T}
    beam_window_fft::Vector{Complex{T}}
end


function init_buffers(nsta, n_pairs, n_trios, nite, nite_f, nite_c, lwin, fs, fmin, fmax, B, n_gamma, upsample, df_taper)

    T = Float64

    # Inicializa FS-PHAT
    ws = init_fsgcc(lwin, fs, fmin, fmax, B, n_gamma, upsample, df_taper)

    # Buffers de correlación y delays
    dt = zeros(T, n_pairs)
    cc = zeros(T, n_pairs)
    sigma = zeros(T, n_pairs)

    # Buffers de tríos
    triad = TriadParams(
        Vector{Int}(undef, n_trios), # t
        Vector{T}(undef, n_trios), Vector{T}(undef, n_trios), # x1, y1
        Vector{T}(undef, n_trios), Vector{T}(undef, n_trios), # x2, y2
        Vector{T}(undef, n_trios), Vector{T}(undef, n_trios), # x3, y3
        Vector{T}(undef, n_trios), # dt1
        Vector{T}(undef, n_trios), # dt2
        Vector{T}(undef, n_trios), # dt3
        Vector{T}(undef, n_trios), # cc_avg
        Vector{T}(undef, n_trios), # closure
        Vector{T}(undef, n_trios), # sx local
        Vector{T}(undef, n_trios), # sy local
        Vector{T}(undef, n_trios), # aperture effective
        Vector{T}(undef, n_trios)  # distancia euclidiana a sx,sy global
    )

    # Buffers de likelihood
    like_map = zeros(T, nite, nite)
    finer_map   = zeros(T, nite_f, nite_f)
    coarser_map = zeros(T, nite_c, nite_c)

    # beam analysis
    station_mask = falses(nsta)
    station_lags = zeros(Int, nsta)
    windows = haning_windows(lwin, T)
    beam = zeros(T, lwin)
    beam_window_fft = zeros(Complex{T}, lwin)

    buff = ThreadBuffers(ws, dt, cc, sigma, triad, like_map, finer_map, coarser_map, station_mask, station_lags, windows, beam, beam_window_fft)
    
    return buff
end


function trias(data::AbstractArray, x::AbstractVector, y::AbstractVector, fs::Real, args...; kwargs...)
    
    SA = SeisArray2D(x, y, data, fs)
    
    return trias(SA, args...; kwargs...)
end


function trias(S::SeisArray2D, lwin::Int, nadv::T, fmin::T, fmax::T; slowmax::T=2.5, slowint_c::T=0.1, slowint_f::T=0.01, slowfw::T=0.5, min_cc::T=0.5, psr_th::T=5.0, C_max::T=0.5) where {T<:Real}

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
    sigma_min = 1/(upsample*S.fs)

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
    dout["sx"] = fill(NaN, nwin)
    dout["sy"] = fill(NaN, nwin)
    dout["s_ratio"] = fill(NaN, nwin)
    dout["s_circ"]  = fill(NaN, nwin)
    dout["s_radii"] = fill(NaN, nwin)
    dout["beam_pow"]   = fill(NaN, nwin)
    dout["beam_peak"] = fill(NaN, nwin)
    dout["fpeak"]  = fill(NaN, nwin)
    dout["baz"]   = fill(NaN, (nwin,3))
    dout["slow"]  = fill(NaN, (nwin,3))
    dout["baz_width"]  = fill(NaN, nwin)
    dout["slow_width"] = fill(NaN, nwin)
    dout["CC_trios"] = fill(NaN, nwin)
    dout["kappa"]    = fill(NaN, nwin)
    dout["C_avg"]    = fill(NaN, nwin)
    dout["D_avg"]    = fill(NaN, nwin)
    dout["S_med"]    = fill(NaN, nwin)
    dout["eta2"]     = fill(NaN, nwin)

    @views Threads.@threads for nk in 1:nwin
        # inicia el buffer
        tid = Threads.threadid()
        buf = thread_buffers[tid]
        fill!(buf.cc, -2.0)

        # define la ventana
        n0 = round(Int, 1 + lwin * nadv * (nk - 1))
        window_data = S.data[n0:n0+lwin-1, :]

        # ---------------------------------
        # 1. Establece triadas POTENCIALES
        # ---------------------------------
        k_vt   = 1 # identificacion de triada
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
                    delay_ij, psr_ij, sigma_ij  = compute_delay!(buf.fsgcc_ws, signal_j, signal_i)

                    if psr_ij < psr_th || abs(delay_ij) > (dist * slowmax)
                        # primer filtro de validacion
                        # el delay tiene que ser fisicamente valido y estadísticamente claro
                        buf.cc[p] = 0.0
                        is_valid_trio = false
                        break
                    end
                    
                    # calcula coeficiente correlacion en el tiempo
                    lag = round(Int, delay_ij * S.fs)
                    cc_ij = cc_overlap(signal_j, signal_i, lag, lwin)
                    
                    # segundo filtro de validacion
                    if cc_ij >= min_cc
                        # guarda parametros
                        buf.dt[p]    = delay_ij
                        buf.cc[p]    = cc_ij
                        buf.sigma[p] = sigma_min + sigma_ij
                    else
                        buf.cc[p] = 0.0
                        is_valid_trio = false
                        break
                    end

                elseif buf.cc[p] == 0.0
                    is_valid_trio = false
                    break
                end
            end

            # Validación de Cierre y Llenado de vt
            if is_valid_trio
                dt1 = buf.dt[trio.p1_idx]
                dt2 = buf.dt[trio.p2_idx]
                dt3 = buf.dt[trio.p3_idx]

                # Calcula la metrica de cierre
                # epislon = dt1 + dt2 + dt3 ≈ 0
                closure = (dt1 * trio.s1) + (dt2 * trio.s2) + (dt3 * trio.s3)

                # limite fisico
                max_tce = trio.dmax * slowmax
                if abs(closure) <= max_tce #* tol_tce
                    # calcula la correlacion promedio
                    c1    = buf.cc[trio.p1_idx]
                    c2    = buf.cc[trio.p2_idx]
                    c3    = buf.cc[trio.p3_idx]
                    ccavg = (c1 + c2 + c3) / 3.0

                    # propagacion de errores
                    s1 = buf.sigma[trio.p1_idx]
                    s2 = buf.sigma[trio.p2_idx]
                    s3 = buf.sigma[trio.p3_idx]
                    s_closure = sqrt(s1^2 + s2^2 + s3^2)

                    # carga los parametros de las triadas
                    buf.trios.t[k_vt] = t # id

                    # geometrica de la triada
                    buf.trios.x1[k_vt] = dx[trio.p1_idx]
                    buf.trios.y1[k_vt] = dy[trio.p1_idx]
                    buf.trios.x2[k_vt] = dx[trio.p2_idx]
                    buf.trios.y2[k_vt] = dy[trio.p2_idx]
                    buf.trios.x3[k_vt] = dx[trio.p3_idx]
                    buf.trios.y3[k_vt] = dy[trio.p3_idx]

                    # delays
                    buf.trios.dt1[k_vt] = dt1
                    buf.trios.dt2[k_vt] = dt2
                    buf.trios.dt3[k_vt] = dt3

                    # correlacion media
                    buf.trios.cc_avg[k_vt]  = ccavg

                    # tiempo de clausura e incertidumbre
                    buf.trios.closure[k_vt] = closure
                    buf.trios.sigma[k_vt]   = s_closure

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
            nvalid1 = clean_triads_temp!(sx0, sy0, buf, trios, nvalid0, C_max)

            if nvalid1 >= 1
                if nvalid0 != nvalid1
                    # Re-calcula el grid grueso
                    misfitmap!(buf.coarser_map, buf, nvalid1, s_grid_c, s_grid_c, slomax2)
                    idx = argmin(buf.coarser_map)
                    sx0 = s_grid_c[idx[1]]
                    sy0 = s_grid_c[idx[2]]
                end
                
                # CALCULO DEL POWER BEAM Y FPEAK
                powbeam, beampeak, fpeak = beam_analysis(window_data, S.xcoord, S.ycoord, lwin, S.fs, sx0, sy0, buf)

                # ------------------------------------------
                # 1.3 FILTRA POR TRIADAS ESPACIAL COHERENTES
                # ------------------------------------------
                nvalid2, quality_metrics = clean_triads_space!(sx0, sy0, buf, trios, nvalid1, fpeak)

                if nvalid2 >= 1
                    save = true
                    C_avg, cc_avg, D_avg, S_med = quality_metrics

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
                    @inbounds for k in 1:nvalid2
                        dsx = buf.trios.sx[k] - sxf
                        dsy = buf.trios.sy[k] - syf
                        # agrega cuadrado de la distancia euclidiana
                        buf.trios.delta_s2[k] = dsx*dsx + dsy*dsy
                    end

                    sigma_s = median([buf.trios.delta_s2[k] for k in 1:nvalid2])
                    slow_f  = sxf*sxf + syf*syf
                    kappa   = exp(-sigma_s / slow_f)
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
            
            # guarda info de las triadas validas
            @inbounds for k in 1:nvalid2
                t_orig = buf.trios.t[k]
                trio = trios[t_orig]
                trios_stats = (trio.sta_triad, buf.trios.cc_avg[k], buf.trios.closure[k], buf.trios.sigma[k], buf.trios.apert_eff[k], buf.trios.delta_s2[k])
                push!(dout["trios"][nk], trios_stats)
            end
            
            dout["sx"][nk] = sxf
            dout["sy"][nk] = syf
            dout["misfit"][nk] = copy(buf.like_map)
            
            dout["fpeak"][nk] = fpeak
            dout["beam_peak"][nk] = beampeak
            dout["beam_pow"][nk]  = powbeam

            # ---------------------------
            # 2.1 METRICAS DE CALIDAD
            # ---------------------------
            dout["CC_trios"][nk] = cc_avg
            dout["C_avg"][nk]    = C_avg
            dout["D_avg"][nk]    = D_avg
            dout["kappa"][nk]    = kappa
            dout["S_med"][nk]    = S_med

            # eta2
            eta2 = kappa * sigmoid(-C_avg; x0=-1.) * sigmoid(D_avg; x0=0.5)
            dout["eta2"][nk] = eta2

            # ----------------------
            # 3 PSEUDO-VEROSIMILITUD
            # ----------------------
            @. buf.like_map = exp(-sqrt(nvalid2 * eta2) * buf.like_map / S_med)
            
            # nivel de una desviacion estandar
            level = maximum(buf.like_map) / ℯ

            # ----------------------
            # 3.1 INCERTIDUMBRE
            # ----------------------
            uncert = uncertainty_contour(s_grid, s_grid, buf.like_map, level)

            dout["s_ratio"][nk] = uncert[1]
            dout["s_circ"][nk]  = uncert[2] # circularidad del contorno de íncertidumbre
            dout["s_radii"][nk] = uncert[3]
            
            # SLOW min, SLOW central, SLOW max
            slobnd = uncert[4]
            dout["slow"][nk,1] = slobnd[1]
            dout["slow"][nk,2] = slobnd[2]
            dout["slow"][nk,3] = slobnd[3]
            dout["slow_width"][nk] = slobnd[4]

            # BAZ min, BAZ central, BAZ max
            bazbnd = uncert[5]
            dout["baz"][nk,1] = bazbnd[1]
            dout["baz"][nk,2] = bazbnd[2]
            dout["baz"][nk,3] = bazbnd[3]
            dout["baz_width"][nk] = bazbnd[4]
        end
    end

    mask = findall(x -> !isnan(x), dout["time_s"])

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
                dt_t1 = sx * buf.trios.x1[k] + sy * buf.trios.y1[k]
                dt_t2 = sx * buf.trios.x2[k] + sy * buf.trios.y2[k]
                dt_t3 = sx * buf.trios.x3[k] + sy * buf.trios.y3[k]
                    
                # residuos norma L1
                e_trio = abs(dt_t1 - buf.trios.dt1[k]) + 
                         abs(dt_t2 - buf.trios.dt2[k]) + 
                         abs(dt_t3 - buf.trios.dt3[k])

                # tiempo maximo de la triada
                tmax = inv_2 * (abs(dt_t1) + abs(dt_t2) + abs(dt_t3))
                tmax += buf.trios.sigma[k]

                # funcion peso
                clos  = buf.trios.closure[k]
                w_val = buf.trios.cc_avg[k] * exp(-clos*clos/(tmax*tmax))

                W_sum += w_val
                R_sum += w_val * e_trio
            end

            like_map[i, j] = R_sum / W_sum
        end
    end
end


function clean_triads_temp!(best_sx::T, best_sy::T, buf::ThreadBuffers, trios, n_valid_trios::Int, C_max::T) where {T<:Real}
    
    k_vt_new = 1
    inv_2 = T(0.5)
    stamask = buf.station_mask

    # Reseteamos la máscara de estaciones al inicio
    fill!(stamask, false)

    @inbounds for k in 1:n_valid_trios
        # Extraemos las distancias de los lados de la tríada
        x1, y1 = buf.trios.x1[k], buf.trios.y1[k]
        x2, y2 = buf.trios.x2[k], buf.trios.y2[k]
        x3, y3 = buf.trios.x3[k], buf.trios.y3[k]
        
        # Calculamos el tiempo de tránsito direccional para cada lado
        t1 = abs(x1 * best_sx + y1 * best_sy)
        t2 = abs(x2 * best_sx + y2 * best_sy)
        t3 = abs(x3 * best_sx + y3 * best_sy)
        tau_max = inv_2 * (t1 + t2 + t3)
        tau_max += buf.trios.sigma[k]
        clos = buf.trios.closure[k]
        C_triad = clos*clos/(tau_max*tau_max)
        
        # Si pasa el filtro, la mantenemos en el buffer
        if C_triad <= C_max
            t_orig = buf.trios.t[k]
            trio   = trios[t_orig]

            # activa las estaciones para el power beam
            for sta_idx in trio.sta_triad
                stamask[sta_idx] = true
            end

            # Desplazamos los datos en el buffer
            if k_vt_new != k
                buf.trios.t[k_vt_new]  = t_orig
                buf.trios.x1[k_vt_new] = x1
                buf.trios.y1[k_vt_new] = y1
                buf.trios.x2[k_vt_new] = x2
                buf.trios.y2[k_vt_new] = y2
                buf.trios.x3[k_vt_new] = x3
                buf.trios.y3[k_vt_new] = y3
                buf.trios.dt1[k_vt_new] = buf.trios.dt1[k]
                buf.trios.dt2[k_vt_new] = buf.trios.dt2[k]
                buf.trios.dt3[k_vt_new] = buf.trios.dt3[k]
                buf.trios.cc_avg[k_vt_new]  = buf.trios.cc_avg[k]
                buf.trios.closure[k_vt_new] = buf.trios.closure[k]
                buf.trios.sigma[k_vt_new]   = buf.trios.sigma[k]
            end
            k_vt_new += 1
        end
    end

    return k_vt_new-1
end


function clean_triads_space!(best_sx::T, best_sy::T, buf::ThreadBuffers, trios, n_valid_trios::Int, f_peak::T) where {T<:Real}

    k_vt_new = 1
    inv_2 = T(0.5)

    D_triad_avg  = T(0.0) # aperture
    C_triad_avg  = T(0.0) # closure
    cc_triad_avg = T(0.0) # correlation

    @inbounds for k in 1:n_valid_trios
        t_orig = buf.trios.t[k]
        
        # tau_max con best_s
        t1 = abs(buf.trios.x1[k] * best_sx + buf.trios.y1[k] * best_sy)
        t2 = abs(buf.trios.x2[k] * best_sx + buf.trios.y2[k] * best_sy)
        t3 = abs(buf.trios.x3[k] * best_sx + buf.trios.y3[k] * best_sy)
        tau_max = inv_2 * (t1 + t2 + t3)

        # apertura efectiva de la kth-triada
        apert_eff = tau_max * f_peak

        # FILTRO ESPACIAL
        nyquist_min = T(0.5)
        if apert_eff > nyquist_min
            D_triad_avg += apert_eff

            clos = buf.trios.closure[k]
            sig  = buf.trios.sigma[k]
            C_triad = clos*clos / (tau_max+sig)^2
            C_triad_avg += C_triad

            cc_triad_avg += buf.trios.cc_avg[k]

            # Desplazamos los datos en el buffer
            if k_vt_new != k
                buf.trios.t[k_vt_new]  = t_orig
                buf.trios.x1[k_vt_new] = buf.trios.x1[k]
                buf.trios.y1[k_vt_new] = buf.trios.y1[k]
                buf.trios.x2[k_vt_new] = buf.trios.x2[k]
                buf.trios.y2[k_vt_new] = buf.trios.y2[k]
                buf.trios.x3[k_vt_new] = buf.trios.x3[k]
                buf.trios.y3[k_vt_new] = buf.trios.y3[k]
                buf.trios.dt1[k_vt_new] = buf.trios.dt1[k]
                buf.trios.dt2[k_vt_new] = buf.trios.dt2[k]
                buf.trios.dt3[k_vt_new] = buf.trios.dt3[k]
                buf.trios.cc_avg[k_vt_new]  = buf.trios.cc_avg[k]
                buf.trios.closure[k_vt_new] = buf.trios.closure[k]
                buf.trios.sigma[k_vt_new]   = buf.trios.sigma[k]
            end
            
            # agrega la apertura
            buf.trios.apert_eff[k_vt_new] = apert_eff

            # y calcula sx,sy de la triada (local)
            slowness_triad!(k_vt_new, buf.trios)
            
            k_vt_new += 1
        end
    end

    n_valid_trios2 = k_vt_new-1

    if n_valid_trios2 >= 1
        C_triad_avg /= n_valid_trios2
        D_triad_avg /= n_valid_trios2
        cc_triad_avg /= n_valid_trios2

        # incertidumbre típica del estimador de delay
        S_triads = median(view(buf.trios.sigma, 1:n_valid_trios2))
        return n_valid_trios2, (C_triad_avg, cc_triad_avg, D_triad_avg, S_triads)

    else
        return n_valid_trios2, (T(NaN), T(NaN), T(NaN), T(NaN))
    end
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

    # construye el beam
    @inbounds for k in eachindex(station_mask)
        if station_mask[k]
            lag = station_lags[k]
            for idx in 1:n_samples
                beam[idx] += data[t0 + idx - 1 + lag, k]
            end
        end
    end

    # calcula potenia del beam
    power_sum = 0.0
    @inbounds @simd for idx in 1:n_samples
        val = beam[idx]
        power_sum = muladd(val, val, power_sum)
    end
    norm_factor = T(n_active * n_active) * T(n_samples)
    beam_power = power_sum / norm_factor

     # Peak value and prepare fft_buf
    max_abs = 0.0
    taper_window = buf.beam_window[n_samples]
    @inbounds for i in 1:n_samples
        val = beam[i]
        max_abs = max(max_abs, abs(val))
        fft_ws[i] = Complex{T}(val * taper_window[i])
    end
    beam_peak = max_abs / n_active

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

    return beam_power, beam_peak, fpeak
end


function slowness_triad!(k::Int, vt::TriadParams)
    
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
    vt.sx[k] = inv_det * ( mtm_22 * mtd_1 - mtm_12 * mtd_2 )
    vt.sy[k] = inv_det * (-mtm_12 * mtd_1 + mtm_11 * mtd_2 )

    return
end

