#!/usr/local/bin julia
# coding=utf-8

# GNU GPL v2 licenced to I. Melchor and J. Almendros 08/2022
# Zero LAG CrossCorrelacion

function zlcc(data::AbstractArray, x::AbstractVector, y::AbstractVector, fs::Real, args...; kwargs...)
    
    SA = SeisArray2D(x, y, data, fs)
    
    return zlcc(SA, args...; kwargs...)
end


function zlcc(S::SeisArray2D, lwin::Int, nadv::Real, fmin::Real, fmax::Real, slowmax::Real, toff::Real; slowint_c::Real=0.1, slowint_f::Real=0.01, ccerr::Real=0.95, maac_th::Real=0.3, slowfw::Real=0.5, ratio_max::Real=0.05, return_cmap::Bool=true, stack::Bool=false, baz_th::Real=30.0, baz_lim::Union{Vector{<:Real}, Nothing}=nothing)
    
    # filtramos los datos
    filter!(S, fmin, fmax)

    # definición de ventanas
    npts, nsta = size(S.data)
    toff_samp = round(Int, toff * S.fs)
    step = round(Int, lwin * nadv)
    nwin = floor(Int, (npts - 2 * toff_samp - lwin) / step) + 1

    # inica buffers pre-allocados
    n_threads = Threads.nthreads()
    thread_buffers = [
        init_zlcc_workspace(S, lwin, slowmax, slowint_c, slowfw, slowint_f)
    for _ in 1:n_threads
    ]

    # diccionario de guardado
    dout = Dict{String, Any}()
    dout["time_s"] = fill(NaN, nwin)
    dout["maac"]   = fill(NaN, nwin)
    dout["sx"]     = fill(NaN, nwin)
    dout["sy"]     = fill(NaN, nwin)
    dout["beam"]   = fill(NaN, nwin)
    dout["baz"]    = fill(NaN, (nwin,3))
    dout["slow"]   = fill(NaN, (nwin,3))
    dout["s_ratio"]  = fill(NaN, nwin)
    dout["s_circ"]   = fill(NaN, nwin)
    dout["s_radii"]  = fill(NaN, nwin)
    dout["baz_width"]  = fill(NaN, nwin)
    dout["slow_width"] = fill(NaN, nwin)

    if return_cmap || stack
        return_cmap = true
        dout["slowmap"] = Vector{Union{Matrix{Float32}, Nothing}}(undef, nwin)
    end
    
    @inbounds @views Threads.@threads for nk in 1:nwin
        n0 = 1 + toff_samp + step * (nk - 1)

        tid = Threads.threadid()
        buf = thread_buffers[tid]

        # calcula el mapa de lentitud aparente (coarser)
        _compute_ccmap!(buf, n0)
        maac, idxf = findmax(buf.ccmap_c)

        if maac > maac_th
            # busca el maximo
            sx0 = buf.s_grid_c[idxf[1]]
            sy0 = buf.s_grid_c[idxf[2]]

            # calcula el mapa de lentitud aparente (finer)
            _compute_ccmap!(buf, n0, sx0, sy0)

            # interpola
            rfx = buf.s_grid_f .+ sx0
            rfy = buf.s_grid_f .+ sy0
            interpolate_grids!(buf.ccmap, buf.ccmap_c, buf.ccmap_f, buf.s_grid, buf.s_grid_c, rfx, rfy)

            # calcula el maac
            maac, midx = findmax(buf.ccmap)
            ii, jj = midx.I
            best_sx = buf.s_grid[ii]
            best_sy = buf.s_grid[jj]

            dout["time_s"][nk] = (n0 - 1 - toff_samp) / S.fs
            dout["maac"][nk] = maac
            dout["sx"][nk] = best_sx
            dout["sy"][nk] = best_sy

            if return_cmap
                dout["slowmap"][nk]  = copy(buf.ccmap)
            end

            # calcula el contorno
            level = maac*ccerr
            is_good, uncert = uncertainty_contour(buf.s_grid, buf.s_grid, buf.ccmap, level, ratio_max)
            dout["s_ratio"][nk] = uncert[1]
            dout["s_circ"][nk]  = uncert[2] # circularidad del contorno de íncertidumbre

            if is_good
                radii, slobnd, bazbnd = uncert[3], uncert[4], uncert[5]
                dout["s_radii"][nk] = radii

                # guarda el power beam de la deteccion
                dout["beam"][nk] = _power_beam(buf, n0, best_sx, best_sy)

                # guarda vector de lentitud aparente
                dout["baz"][nk,1] = bazbnd[1]
                dout["baz"][nk,2] = bazbnd[2]
                dout["baz"][nk,3] = bazbnd[3]
                dout["baz_width"][nk] = bazbnd[4]

                dout["slow"][nk,1] = slobnd[1]
                dout["slow"][nk,2] = slobnd[2]
                dout["slow"][nk,3] = slobnd[3]
                dout["slow_width"][nk] = slobnd[4]
            end
        end
    end

    mask = findall(!isnan, dout["maac"])

    if isempty(mask)
        thread_buffers = nothing
        GC.gc()
        return nothing
    end

    if stack
        stack_dout = zlcc_stack(dout, mask, ws, maac_th, baz_th, baz_lim, ccerr, ratio_max)
    end
    
    final_dout = Dict{String, Any}()
    for key in keys(dout)
        val = dout[key]
        if ndims(val) == 1
            final_dout[key] = val[mask]
        else
            final_dout[key] = val[mask, :, :]
        end
    end

    thread_buffers = nothing
    GC.gc()

    if stack
        return final_dout, stack_dout
    else
        return final_dout
    end
end


"""
    functions for ZLCC
"""
function init_zlcc_workspace(S::SeisArray2D, lwin, slowmax, slowint_c, slowfw, slowint_f)
    
    npts, nsta = size(S.data)
    
    # Geometría relativa en muestras (fsem incluido)
    xref, yref = mean(S.xcoord), mean(S.ycoord)
    dx = (S.xcoord .- xref) .* S.fs 
    dy = (S.ycoord .- yref) .* S.fs
    
    # Iterador de pares (triángulo superior)
    citer = cciter(nsta)

    # slowness maximo
    slomax2 = slowmax*slowmax

    # definición de los mapas de lentitud:
    # coarser
    s_grid_c  = -slowmax:slowint_c:slowmax
    nite_c    = size(s_grid_c, 1)
    
    # finer
    s_grid_f  = -slowfw:slowint_f:slowfw
    nite_f  = size(s_grid_f, 1)
    
    # interpolated
    s_grid = -slowmax:slowint_f:slowmax
    nite   = size(s_grid, 1)
    
    # Buffers
    T = Float64
    ccmap = zeros(T, nite, nite)
    ccmap_c = zeros(T, nite_c, nite_c)
    ccmap_f = zeros(T, nite_f, nite_f)

    # Buffer de energía independiente para cada hilo
    benergy = zeros(T, nsta)
    beam = zeros(T, lwin)

    ws = ZLCC_WS_CPU(S.data, dx, dy, citer, lwin, nsta, slomax2, s_grid, s_grid_c, s_grid_f, nite, nite_c, nite_f, ccmap, ccmap_c, ccmap_f, benergy, beam)

    return ws
end



function _compute_ccmap!(ws::ZLCC_WS_CPU, n0::Int)
    
    # desempaqueta (coarser)
    data   = ws.data
    dx, dy = ws.dx, ws.dy
    citer  = ws.citer
    lwin   = ws.lwin
    ccmap  = ws.ccmap_c
    limit_sq = ws.slomax2
    sgrid = ws.s_grid_c
    nite = ws.nite_c
    ebuf = ws.benergy

    @inbounds for j in 1:nite
        py  = sgrid[j]
        py2 = py^2
        
        for i in 1:nite
            px = sgrid[i]
            
            if (px^2 + py2) > limit_sq
                ccmap[i, j] = 0.0
                continue
            end
            
            # Calcular Energías por Estación para este (px, py)
            for s in 1:ws.nsta
                shift = round(Int, px * dx[s] + py * dy[s])

                start_idx = n0 + shift
                
                sq_sum = 0.0
                @simd for k in 0:(lwin-1)
                    val = data[start_idx + k, s]
                    sq_sum = muladd(val, val, sq_sum)
                end
                ebuf[s] = 1.0 / sqrt(sq_sum)
            end
            
            # Correlación Cruzada Sumada
            cc_sum = 0.0
            for (sta_i, sta_j) in citer
                shift_i = round(Int, px * dx[sta_i] + py * dy[sta_i])
                shift_j = round(Int, px * dx[sta_j] + py * dy[sta_j])
                
                idx_i = n0 + shift_i
                idx_j = n0 + shift_j
                
                dot_val = 0.0
                @simd for k in 0:(lwin-1)
                    val_i = data[idx_i + k, sta_i]
                    val_j = data[idx_j + k, sta_j]
                    dot_val = muladd(val_i, val_j, dot_val)
                end
                cc_sum += dot_val * ebuf[sta_i] * ebuf[sta_j]
            end

            ccmap[i, j] = (2 * cc_sum + ws.nsta) / (ws.nsta * ws.nsta)
        end
    end
end



function _compute_ccmap!(ws::ZLCC_WS_CPU, n0::Int, sx0, sy0)
    
    # desempaqueta (finer)
    data   = ws.data
    dx, dy = ws.dx, ws.dy
    citer  = ws.citer
    lwin   = ws.lwin
    limit_sq = ws.slomax2
    ccmap  = ws.ccmap_f
    sgrid = ws.s_grid_f
    nite = ws.nite_f
    ebuf = ws.benergy
    
    @inbounds for j in 1:nite
        py  = sgrid[j] + sy0
        py2 = py^2
        
        for i in 1:nite
            px = sgrid[i] + sx0
            
            if (px^2 + py2) > limit_sq
                ccmap[i, j] = 0.0
                continue
            end

            # Calcular Energías por Estación para este (px, py)
            for s in 1:ws.nsta
                shift = round(Int, px * dx[s] + py * dy[s])
                
                start_idx = n0 + shift
                
                sq_sum = 0.0
                @simd for k in 0:(lwin-1)
                    val = data[start_idx + k, s]
                    sq_sum = muladd(val, val, sq_sum)
                end
                ebuf[s] = 1.0 / sqrt(sq_sum)
            end
            
            # Correlación Cruzada Sumada
            cc_sum = 0.0
            for (sta_i, sta_j) in citer
                shift_i = round(Int, px * dx[sta_i] + py * dy[sta_i])
                shift_j = round(Int, px * dx[sta_j] + py * dy[sta_j])
                
                idx_i = n0 + shift_i
                idx_j = n0 + shift_j
                
                dot_val = 0.0
                @simd for k in 0:(lwin-1)
                    val_i = data[idx_i + k, sta_i]
                    val_j = data[idx_j + k, sta_j]
                    dot_val = muladd(val_i, val_j, dot_val)
                end
                cc_sum += dot_val * ebuf[sta_i] * ebuf[sta_j]
            end

            ccmap[i, j] = (2 * cc_sum + ws.nsta) / (ws.nsta * ws.nsta)
        end
    end
end



function _power_beam(ws::ZLCC_WS_CPU, n0::Int, sx0, sy0)

    # zero-allocation reset
    fill!(ws.beam, 0.0)
    
    # Desempaquetar
    data = ws.data
    dx, dy = ws.dx, ws.dy
    beam = ws.beam
    lwin = ws.lwin
    nsta = ws.nsta

    # Stacking (Beamforming)
    @inbounds for ii in 1:nsta
        delay = sx0 * dx[ii] + sy0 * dy[ii]
        idx_start = n0 + round(Int, delay) - 1

        @simd for k in 1:lwin
            beam[k] += data[idx_start + k, ii]
        end
    end

    # Mean Square
    sum_sq = 0.0
    @inbounds @simd for k in 1:lwin
        val = beam[k]
        # Fused Multiply-Add: val*val + sum_sq
        sum_sq = muladd(val, val, sum_sq)
    end

    return sqrt(sum_sq / lwin) / nsta
end



function zlcc_stack(dout::Dict, mask::Vector{<:Integer}, ws::ZLCC_WS_CPU, maac_th::Real, baz_th::Real, baz_lim::Union{Vector{<:Real}, Nothing}=nothing, ccerr::Real=0.95, ratio_max::Real=0.05)

    raw_maac = dout["maac"][mask,1]
    raw_time = dout["time_s"][mask]
    raw_rms  = dout["rms"][mask]
    raw_baz  = dout["baz"][mask,2]
    raw_slow = dout["slow"][mask,2]
    raw_baz_width = dout["baz_width"][mask]
    raw_smap  = dout["slowmap"][mask,:,:]

    N_total = length(dout["time_s"])

    if length(raw_time) == 0
        return nothing
    end
    
    # Calcular máscara de Stacking
    cond_maac  = raw_maac .>= maac_th
    cond_baz_w = raw_baz_width .<= baz_th

    if baz_lim !== nothing
        bmin, bmax = baz_lim
        if bmin <= bmax
            cond_baz_lim = (raw_baz .>= bmin) .& (raw_baz .<= bmax)
        else
            cond_baz_lim = (raw_baz .>= bmin) .| (raw_baz .<= bmax)
        end
        mask_stack = cond_maac .& cond_baz_w .& cond_baz_lim
    else
        mask_stack = cond_maac .& cond_baz_w
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

    w = raw_maac[nidx]
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
    raw_maps = raw_smap[nidx, :, :]
    w_stack = dropdims(sum(raw_maps .* reshape(w, :, 1, 1), dims=1), dims=1) ./ sum_w
    stack_out["slowmap_stack"] = w_stack

    maac_peak = maximum(w_stack)
    stack_out["maac"] = maac_peak

    is_good, ratio, s_c, slobnd, bazbnd = uncertainty_contour(ws.sx, ws.sy, w_stack, maac_peak * ccerr, ratio_max)
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


