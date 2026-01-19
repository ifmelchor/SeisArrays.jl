#!/usr/local/bin julia
# coding=utf-8

# GNU GPL v2 licenced to I. Melchor and J. Almendros 08/2022
using Base.Threads


function relse_absolute(data::AbstractMatrix{T}, xStaUTM::AbstractVector{T}, yStaUTM::AbstractVector{T}, fsem::J, lwin::J, nadv::T, fqband::Vector{T}; cc_th::T=0.6, n_min::J=1, nite::J=81, rms_th::T=0.1, slomax::T=2.0, sloint::T=0.02, ratio_max::T=0.05, save_maps::Bool=false) where {T<:Real, J<:Integer}

    # cc_th  :: umbral de correlacion cruzada
    # n_min  :: numero mínimo de triangulaciones
    # rms_th :: umbral de error en la estimacion del vector lentitud aparente lineal para el calculo del grid search
    # ratio_max :: umbral de tasa de tamaños de los contornos que definen la mejor estimacion en el mapa de ajuste del grid search

    # Cálculo de nwin
    npts, nsta = size(data)

    step = round(Int, lwin * nadv)
    nwin = div(npts - lwin, Int(step)) + 1

    data = _filter(data, fsem, fqband)

    pairs = _cciter(nsta)
    trios = _triangles(pairs)

    num_pairs = length(pairs)
    dx, dy = _pair_dist(xStaUTM, yStaUTM, pairs)

    dout = _empty_dict(nwin, nite, save_maps)

    n_threads = Threads.nthreads()
    buffers_dt = [zeros(num_pairs) for _ in 1:n_threads]
    buffers_cc = [zeros(num_pairs) for _ in 1:n_threads]

    buffers_resmap     = [zeros(T, nite, nite) for _ in 1:n_threads]
    buffers_sx      = [zeros(T, nite) for _ in 1:n_threads]
    buffers_sy      = [zeros(T, nite) for _ in 1:n_threads]

    @views @inbounds Threads.@threads for nk in 1:nwin

        tid = Threads.threadid()
        local_dt = buffers_dt[tid]
        local_cc = buffers_cc[tid]
        resmap = buffers_resmap[tid]
        sx = buffers_sx[tid]
        sy = buffers_sy[tid]

        n0 = round(Int, 1 + lwin * nadv * (nk - 1))
        window_data = data[n0:n0+lwin-1, :]

        @inbounds for k in 1:num_pairs
            i, j = pairs[k]
            lag, max_cc = _delay_fftw(window_data[:, j], window_data[:, i], fsem)
            local_dt[k] = lag
            local_cc[k] = max_cc
        end

        mask = local_cc .> cc_th
        npairs = count(mask)

        cc_filt = local_cc[mask]
        pairs_filt = pairs[mask]
        dout["cc"][nk, 1] = mean(cc_filt)
        dout["cc"][nk, 2] = std(cc_filt)

        # error de triangulacion
        # TCE = time delay closure error
        local_dt[.!mask] .= NaN
        ntrios, avg_dt_closure, active_trios, active_errors = _closure_error(local_dt, pairs, trios, nsta)
        dout["tce"][nk,1] = ntrios
        dout["tce"][nk,2] = avg_dt_closure
        dout["tce_detail"][nk] = (active_trios, active_errors)

        if ntrios > n_min
            # apparent slowness
            dx_f = dx[mask]
            dy_f = dy[mask]
            dt_f = local_dt[mask]

            # primer calculo lineal
            s_lin, rms_lin = _slow_linear(dx_f, dy_f, dt_f)
            dout["rms_lin"][nk] = rms_lin

            # vector lentitud aparente (lineal) --> master usado para el grid search
            slow_lin, baz_lin = r2p(-s_lin[1], -s_lin[2])
            dout["slow_lin"][nk,1] = slow_lin
            dout["slow_lin"][nk,2] = baz_lin

            if rms_lin < rms_th
                # vector lentitud aparente preciso
                (px, py), misfit = _grid_search(dx_f, dy_f, dt_f, slomax, sloint, s_lin)

                if all(isfinite, (px, py, misfit))
                    # compute MAE (Mean Absolute Error), grid search uses L1 norm
                    mae = misfit/npairs
                    dout["mae"][nk] = mae

                    # get resmap
                    _resmap!(resmap, sx, sy, px, py, mae, dx_f, dy_f, dt_f, nite)

                    # compute contour
                    res_min = minimum(resmap)
                    is_good, ratio, slobnd, bazbnd = uncertainty_contour(sx, sy, resmap, res_min*1.16, ratio_max)
                    dout["ratio"][nk] = ratio

                    if save_maps
                        dout["bounds"][nk, 1] = minimum(sx)
                        dout["bounds"][nk, 2] = maximum(sx)
                        dout["bounds"][nk, 3] = minimum(sy)
                        dout["bounds"][nk, 4] = maximum(sy)
                        dout["rmsmap"][nk,:,:] .= resmap
                    end

                    if is_good
                        # best estimation of apparent slowness
                        slow_gs, baz_gs = r2p(-px, -py)
                        dout["baz"][nk, 1] = bazbnd[1]
                        dout["baz"][nk, 2] = baz_gs
                        dout["baz"][nk, 3] = bazbnd[2]
                        dout["slow"][nk, 1] = slobnd[1]
                        dout["slow"][nk, 2] = slow_gs
                        dout["slow"][nk, 3] = slobnd[2]
                    end

                end
            end
        end
    end

    return dout
end


function relse_relative(data::AbstractMatrix{T}, xStaUTM::AbstractVector{T}, yStaUTM::AbstractVector{T}, master_data::AbstractArray{T, 3}, master_sx::AbstractVector{T}, master_sy::AbstractVector{T}, master_resmap::AbstractArray{T, 3}, master_bounds::AbstractArray{T, 2}, mae_master::AbstractVector{T}, master_trios_all::AbstractVector, master_errors_all::AbstractVector, active_chan::Union{AbstractVector{J}, Nothing}, fsem::J, nadv::T, fqband::Vector{T}; upsample_factor=20, cc_th::T=0.5, ntrio_min::J=1, nsta_min::J=3, tce_max_tol::T=0.1, slomax::T=2.0, sloint::T=0.02, ratio_max::T=0.05, save_maps::Bool=false) where {T<:Real, J<:Integer}

    npts, nsta_target = size(data)
    lwin, nsta_master, n_master = size(master_data)
    
    nite  = size(master_resmap, 2)
    n_fft = nextpow(2, 2*lwin - 1)
    n_up = n_fft * upsample_factor

    # pre-compute master prior probabilites
    prob_master = [exp.(-(master_resmap[j,:,:]).^2 ./ (2 * mae_master[j]^2)) for j in 1:n_master]

    # pre-compute fft of master data
    master_fft_catalog = _fft_buffer(master_data)

    idxs = isnothing(active_chan) ? (1:nsta_target) : active_chan
    nsta = length(idxs)

    # configura las ventanas
    step = round(Int, lwin * nadv)
    nwin = div(npts - lwin, Int(step)) + 1
    _filter!(data, fsem, fqband)

    # define la geometria
    pairs  = _cciter(nsta)
    trios  = _triangles(pairs)
    dx, dy = _pair_dist(xStaUTM, yStaUTM, pairs)
    num_pairs = length(pairs)

    dout   = _empty_dict(nwin, nite, save_maps)
    dout["best_master"] = fill(0, nwin)
    dout["prob_max"] = fill(NaN, nwin)
    # remove unnecesary keys
    delete!(dout, "slow_lin")
    delete!(dout, "mae")

    # create memory buffers
    n_threads = Threads.nthreads()
    buffers_dt = [zeros(T, num_pairs) for _ in 1:n_threads]
    buffers_cc = [zeros(T, num_pairs) for _ in 1:n_threads]
    buffers_sta_shifts = [zeros(T, nsta) for _ in 1:n_threads]
    buffers_sta_ccs    = [zeros(T, nsta) for _ in 1:n_threads]
    buffers_best_shifts = [zeros(T, nsta) for _ in 1:n_threads]
    buffers_best_ccs    = [zeros(T, nsta) for _ in 1:n_threads]
    buffers_resmap     = [zeros(T, nite, nite) for _ in 1:n_threads]
    buffers_sx      = [zeros(T, nite) for _ in 1:n_threads]
    buffers_sy      = [zeros(T, nite) for _ in 1:n_threads]

    buffers_C_up_freq = [zeros(ComplexF64, n_up) for _ in 1:n_threads]
    buffers_plans = [plan_ifft!(zeros(ComplexF64, n_up), flags=FFTW.MEASURE) for _ in 1:n_threads]

    @views @inbounds Threads.@threads for nk in 1:nwin
        tid = Threads.threadid()

        # Acceso a buffers pre-asignados
        l_dt = buffers_dt[tid]
        l_cc = buffers_cc[tid]
        l_shifts = buffers_sta_shifts[tid]
        l_ccs    = buffers_sta_ccs[tid]
        b_shifts = buffers_best_shifts[tid]
        b_ccs    = buffers_best_ccs[tid]
        resmap   = buffers_resmap[tid]
        s_x = buffers_sx[tid]
        s_y = buffers_sy[tid]
        b_C_up_freq = buffers_C_up_freq[tid]
        b_plan      = buffers_plans[tid]

        n0 = round(Int, 1 + lwin * nadv * (nk - 1))
        window_data = data[n0:n0+lwin-1, :]

        # calcula los fft de los windows
        window_ffts = _fft_buffer(window_data)

        # busca master ideal (el de mayor correlacion)
        max_avg_cc = -1.0
        best_n = 0

        for n in 1:n_master
            current_sum_cc = 0.0
            full_match = true
            for (i, j) in enumerate(idxs)
                delay, cc = _correlate_from_ffts(b_C_up_freq, b_plan, window_ffts[:, i], master_fft_catalog[:, j, n], n_fft, n_up, fsem, upsample_factor)
                if cc < 0.3
                    full_match = false
                    break 
                end
                current_sum_cc += cc
                l_shifts[i] = delay
                l_ccs[i] = cc
            end

            if full_match
                avg_cc = current_sum_cc / nsta
                if avg_cc > max_avg_cc
                    max_avg_cc = avg_cc
                    best_n = n
                    b_shifts .= l_shifts
                    b_ccs .= l_ccs
                end
            end
        end

        dout["cc"][nk, 2] = max_avg_cc

        # save best master
        if best_n > 0 && max_avg_cc > cc_th
            dout["best_master"][nk] = best_n

            # creas el master event
            master = MasterEvent(master_data[:,:,best_n], master_sx[best_n], master_sy[best_n], prob_master[best_n], master_bounds[best_n,:],master_trios_all[best_n], master_errors_all[best_n])

            # llamas al kernel
            _relse_kernel!(nk, dout, master, idxs, pairs, trios, dx, dy, num_pairs, b_shifts, b_ccs, l_dt, l_cc, resmap, s_x, s_y, cc_th, nsta_min, ntrio_min, tce_max_tol, slomax, sloint, ratio_max, save_maps, nite)

        end
    end

    return dout
end


function _relse_kernel!(nk::Int, dout::Dict, master::MasterEvent, idxs, pairs, trios, dx, dy, num_pairs, l_shifts, l_ccs, l_dt, l_cc, resmap, sx, sy, cc_th, nsta_min, ntrio_min, tce_max_tol, slomax, sloint, ratio_max, save_maps, nite)

    # filtro de calidad correlation
    mask_sta = l_ccs .> cc_th

    if count(mask_sta) >= max(nsta_min, 3) # el minimo es 3
        # Reconstrucción de tiempos  con el Maestro como ancla
        @inbounds for k in 1:num_pairs
            i, j = pairs[k]
            if mask_sta[i] && mask_sta[j]
                t_rel = l_shifts[j] - l_shifts[i] # relative delay
                t_abs = master.sx * dx[k] + master.sy * dy[k]
                l_dt[k] = t_rel + t_abs
                l_cc[k] = (l_ccs[i] + l_ccs[j]) / 2
            else
                l_dt[k] = NaN
                l_cc[k] = 0.0
            end
        end

        # filtro de clausura diferencial
        mask  = l_cc .> cc_th
        dout["cc"][nk, 1] = mean(l_cc[mask])

        l_dt[.!mask] .= NaN
        _, _, act_trios, act_errs = _closure_error(l_dt, pairs, trios, idxs)

        # calculo de residuos
        ntrios, avg_delta_tce, com_trios, residuals = _residual_closure_error(act_trios, act_errs, master.trios, master.errors)
        dout["tce"][nk, 1] = ntrios
        dout["tce"][nk, 2] = avg_delta_tce
        dout["tce_detail"][nk] = (com_trios, residuals)

        if length(act_trios) >= ntrio_min && avg_delta_tce < tce_max_tol
            idx_f = findall(mask)
            #la desviación real de tus retardos observados respecto a un plano de onda plana
            slow_lin, sigma_rms = _slow_linear(dx[idx_f], dy[idx_f], l_dt[idx_f])
            sl_x = slow_lin[1]
            sl_y = slow_lin[2]
            in_bounds = (master.bounds[1] <= sl_x <= master.bounds[2]) && 
            (master.bounds[3] <= sl_y <= master.bounds[4])

            if in_bounds
                dout["rms_lin"][nk] = sigma_rms

                println(sigma_rms)

                # mapa de residuos (ahora likelihood)
                _resmap!(resmap, sx, sy, master, dx[idx_f], dy[idx_f], l_dt[idx_f], nite)
                res_min = minimum(resmap)

                # probability map
                prob_rel = exp.(-(resmap).^2 ./ (2 * sigma_rms^2))
                prob_rel ./= maximum(prob_rel)
                
                # posterior
                prob_joint = prob_rel .* (master.prob ./ maximum(master.prob))

                # Esto signfica que p_max es la maxima probabilidad de que el vector de lentitud aparente sea px, py
                p_max, midx = findmax(prob_joint)
                ii, jj = midx.I
                px = sx[ii]
                py = sy[jj]

                if save_maps
                    dout["bounds"][nk, 1] = minimum(sx)
                    dout["bounds"][nk, 2] = maximum(sx)
                    dout["bounds"][nk, 3] = minimum(sy)
                    dout["bounds"][nk, 4] = maximum(sy)
                    dout["rmsmap"][nk,:,:] .= prob_joint
                end

                # compute contour
                # ahora el contorno es fijo
                prob_joint ./= maximum(prob_joint)
                is_good, ratio, slobnd, bazbnd = uncertainty_contour(sx, sy, prob_joint, 0.9, ratio_max)
                dout["ratio"][nk] = ratio
                
                if is_good
                    dout["prob_max"][nk] = p_max # la (maxima) probabilidad de que el vector de lentitud aparente sea px, py
                    # best estimation of apparent slowness
                    slow_gs, baz_gs = r2p(-px, -py)
                    dout["baz"][nk, 1] = bazbnd[1]
                    dout["baz"][nk, 2] = baz_gs
                    dout["baz"][nk, 3] = bazbnd[2]
                    dout["slow"][nk, 1] = slobnd[1]
                    dout["slow"][nk, 2] = slow_gs
                    dout["slow"][nk, 3] = slobnd[2]
                end
            end
        end
    end

    return nothing
end


function _triangles(pairs)
    # lista de estaciones de todos los pares posibles
    stations = unique(vcat([p[1] for p in pairs], [p[2] for p in pairs]))
    N = length(stations)
    
    # 2. Crear un Set de pares para búsqueda rápida
    pair_set = Set(pairs)
    
    triangles = Tuple{Int, Int, Int}[]
    
    # 3. Buscar combinaciones de 3 donde existan los 3 lados
    for i in 1:N, j in i+1:N, k in j+1:N
        s1, s2, s3 = stations[i], stations[j], stations[k]
        # Verificamos si los tres pares existen (en cualquier orden)
        cond1 = (s1, s2) in pair_set || (s2, s1) in pair_set
        cond2 = (s2, s3) in pair_set || (s3, s2) in pair_set
        cond3 = (s3, s1) in pair_set || (s1, s3) in pair_set
        
        if cond1 && cond2 && cond3
            push!(triangles, (s1, s2, s3))
        end
    end
    return triangles
end


function _closure_error(dt::AbstractVector{T}, pairs, triangles, active_chan::Vector{Int}) where T <: AbstractFloat
    # Matriz de busqueda
    # nsta es el número total de estaciones
    nsta = length(active_chan)
    dt_matrix = fill(T(NaN), nsta, nsta)
    @inbounds for (idx, (i, j)) in enumerate(pairs)
        dt_matrix[i, j] =  dt[idx]
        dt_matrix[j, i] = -dt[idx]
    end

    # guardar SOLO los tríos que se pudieron formar
    active_trios = Tuple{Int, Int, Int}[]
    active_errors = T[]
    sum_errors = zero(T)
    
    @inbounds for trio in triangles
        s1, s2, s3 = trio
        
        d12 = dt_matrix[s1, s2]
        d23 = dt_matrix[s2, s3]
        d31 = dt_matrix[s3, s1]

        # Si los tres enlaces existen (no son NaN)
        if !isnan(d12) && !isnan(d23) && !isnan(d31)
            err = d12 + d23 + d31
            abs_err = abs(err)

            real_trio = (active_chan[s1], active_chan[s2], active_chan[s3])
            
            push!(active_trios, real_trio)
            push!(active_errors, abs_err)
            sum_errors += abs_err
        end
    end

    n_found = length(active_errors)
    mean_err = n_found > 0 ? sum_errors / n_found : T(NaN)

    return n_found, mean_err, active_trios, active_errors
end


function _residual_closure_error(active_trios, active_errors, master_trios, master_errors)

    master_map = Dict(zip(master_trios, master_errors))

    common_trios = Tuple{Int, Int, Int}[]
    residuals = eltype(active_errors)[]

    @inbounds for (idx, trio) in enumerate(active_trios)
        if haskey(master_map, trio)
            diff = abs(active_errors[idx] - master_map[trio])
            push!(common_trios, trio)
            push!(residuals, diff)
        end
    end

    avg_delta_tce = isempty(residuals) ? NaN : mean(residuals)
    n_common = length(residuals)

    return n_common, avg_delta_tce, common_trios, residuals
end


function _delay_fftw(s1, s2, fs; upsample_factor=20)
    # compute the delay between traces using the cross-correlation function in the frequency domain
    # the delay is computed by applying a sub-sample interpolation following the Whittaker-Shannon theorem (zero-padding).
    n = length(s1)
    n_fft = nextpow(2, 2n - 1)

    # estandariza y demean
    s1_c = _prepare_signal(s1)
    s2_c = _prepare_signal(s2)
    
    # correlation via FFT (frequency domain)
    s1_p = zeros(eltype(s1_c), n_fft); s1_p[1:n] .= s1_c
    s2_p = zeros(eltype(s2_c), n_fft); s2_p[1:n] .= s2_c

    # compute fft
    s1_f = fft(s1_p)
    s2_f = fft(s2_p)

    return _correlate_from_ffts(s1_f, s2_f, n_fft, fs, upsample_factor)
end


function _fft_buffer(data::AbstractMatrix{T}) where T<:Real

    lwin, nsta = size(data)
    n_fft = nextpow(2, 2*lwin - 1)
    fft_buffer = Matrix{ComplexF64}(undef, n_fft, nsta)
    
    @inbounds for j in 1:nsta
        s_prep = _prepare_signal(data[:,j])
        s_pad = zeros(T, n_fft)
        s_pad[1:lwin] .= s_prep
        fft_buffer[:, j] = fft(s_pad)
    end
    
    return fft_buffer
end


function _fft_buffer(data::AbstractArray{T, 3}) where T<:Real

    lwin, nsta, N = size(data)
    n_fft = nextpow(2, 2*lwin - 1)
    fft_buffer = Array{ComplexF64}(undef, n_fft, nsta, N)
    s_pad = zeros(T, n_fft)

    @inbounds for m in 1:N 
        for j in 1:nsta
            s_prep = _prepare_signal(data[:,j,m])
            fill!(s_pad, 0.0)
            s_pad[1:lwin] .= s_prep
            fft_buffer[:, j, m] = fft(s_pad)
        end
    end
    
    return fft_buffer
end


function _correlate_from_ffts(S1_freq, S2_freq, n_fft, fs, upsample_factor=20)
    C_freq = S1_freq .* conj(S2_freq)

    # interpolation using Whittaker-Shannon theory
    # insertamos ceros en el centro del espectro para aumentar la resolución temporal
    n_up = n_fft * upsample_factor
    C_up_freq = zeros(ComplexF64, n_up)
    
    mid = n_fft ÷ 2
    
    # frecuencias positivas
    C_up_freq[1:mid] = C_freq[1:mid]

    # freq. Nyquists (eviat ringing)
    nyquist_val = C_freq[mid + 1]
    C_up_freq[mid+1] = nyquist_val / 2
    C_up_freq[n_up-mid+1] = nyquist_val / 2

    # frecuencias negativas
    C_up_freq[n_up-mid+2:end] = C_freq[mid+2:n_fft]
    
    # volver al tiempo
    C_up = real(ifft(C_up_freq)) * upsample_factor
    
    # y encontrar el maximum
    max_val, idx = findmax(C_up)
    
    # La FFT interpreta la segunda mitad del vector de correlación como desplazamientos hacia la izquierda (tiempo negativo)
    lag0 = (idx <= n_up ÷ 2) ? (idx - 1) : (idx - 1 - n_up)
    delay = (lag0 / upsample_factor) / fs
    
    return delay, max_val
end

function _correlate_from_ffts(C_up_freq, plan, S1_f, S2_f, n_fft, n_up, fs, upsample_factor)
    # interpolation using Whittaker-Shannon theory
    # insertamos ceros en el centro del espectro para aumentar la resolución temporal
    # n_up = n_fft * upsample_factor
    # C_up_freq = zeros(ComplexF64, n_up)
    
    mid = n_fft ÷ 2
    
    fill!(C_up_freq, 0.0im)

    # frecuencias positivas
    @inbounds @simd for k in 1:mid
        C_up_freq[k] = S1_f[k] * conj(S2_f[k])
    end

    # freq. Nyquists (eviat ringing)
    nyquist_val = S1_f[mid + 1] * conj(S2_f[mid + 1])
    C_up_freq[mid+1] = nyquist_val / 2
    C_up_freq[n_up-mid+1] = nyquist_val / 2

    # frecuencias negativas
    @inbounds @simd for k in 2:mid
        C_up_freq[n_up - mid + k] = S1_f[mid + k] * conj(S2_f[mid + k])
    end

    # IFFT IN-PLACE
    mul!(C_up_freq, plan, C_up_freq)

    # y encontrar el maximum
    max_val = -Inf
    idx = 1
    @inbounds @simd for i in 1:n_up
        v = real(C_up_freq[i])
        if v > max_val
            max_val = v
            idx = i
        end
    end

    max_val *= upsample_factor
    lag0 = (idx <= n_up ÷ 2) ? (idx - 1) : (idx - 1 - n_up)
    delay = (lag0 / upsample_factor) / fs
    
    return delay, max_val
end


function _slow_linear(dx, dy, dt)
    # matriz de diseño
    G = hcat(dx, dy)

    # resuelve el problema de minimos cuadrados
    # s = (G^T G)^-1 G^T dt
    # devuelve el s que minimiza (dt_obs - dt_teo)^2
    s = G \ dt

    # error residual
    rms = norm(G*s - dt) / sqrt(length(dt))

    return s, rms
end


function _grid_search(dx, dy, dt, slow_max, slow_int, master_s, max_iter=3, step_div=5)

    best_sx, best_sy = master_s
    step = slow_int
    n_pairs = length(dt)

    min_error = Inf
    @inbounds for iter in 1:max_iter
        # rango de búsqueda alrededor del maestro
        sx_range = (best_sx - step*step_div) : (step) : (best_sx + step*step_div)
        sy_range = (best_sy - step*step_div) : (step) : (best_sy + step*step_div)

        for sx in sx_range, sy in sy_range
            # Limitamos la búsqueda al círculo de lentitud física posible
            if sx^2 + sy^2 > slow_max^2 continue end

            err = 0.0
            @simd for p in 1:n_pairs
                err += abs(sx * dx[p] + sy * dy[p] - dt[p])
            end

            if err < min_error
                min_error = err
                best_sx, best_sy = sx, sy
            end
        end

        step /= step_div
    end

    return (best_sx, best_sy), min_error
end


function _resmap!(resmap, sx, sy, sx_best, sy_best, sig_t, dx, dy, dt, nite)

    # Matriz de diseño (representa la geometría del array)
    g11 = sum(dx .^ 2)
    g12 = sum(dx .* dy)
    g22 = sum(dy .^ 2)
    
    # el valor 1e-9 evita singularidad (ridge regression)
    det = g11 * g22 - g12 * g12 + 1e-9

    # Inversa de G'G para obtener sigmas
    sig_sx = sig_t * sqrt(abs(g22 / det))
    sig_sy = sig_t * sqrt(abs(g11 / det))

    # Definimos que el rango de búsqueda sea 2 veces la sigma teórica
    slomax_map = 4 * max(sig_sx, sig_sy) + 1e-4

    sx .= range(sx_best - slomax_map, sx_best + slomax_map, length=nite)
    sy .= range(sy_best - slomax_map, sy_best + slomax_map, length=nite)

    n_pairs = length(dt)
    @inbounds for j in 1:nite
        y = sy[j]
        for i in 1:nite
            x = sx[i]
            r = 0.0
            @simd for p in 1:n_pairs
                r += (x * dx[p] + y * dy[p] - dt[p])^2
            end
            resmap[i, j] = r
        end
    end
end


function _resmap!(resmap, sx, sy, master::MasterEvent, dx, dy, dt, nite)

    sx_min = master.bounds[1]
    sx_max = master.bounds[2]
    sy_min = master.bounds[3]
    sy_max = master.bounds[4]

    sx .= range(sx_min, sx_max, length=nite)
    sy .= range(sy_min, sy_max, length=nite)

    n_pairs = length(dt)
    @inbounds for j in 1:nite
        y = sy[j]
        for i in 1:nite
            x = sx[i]
            r = 0.0
            @simd for p in 1:n_pairs
                r += (x * dx[p] + y * dy[p] - dt[p])^2
            end
            resmap[i, j] = r
        end
    end
end