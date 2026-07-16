#!/usr/local/bin julia
# coding=utf-8

# GNU GPL v2 licenced to I. Melchor and J. Almendros 08/2022
# Zero LAG CrossCorrelacion

struct ZLCC_WS_CPU{T<:AbstractFloat, R<:AbstractRange{T}}
    data::Matrix{T}
    dx::Vector{T}
    dy::Vector{T}
    citer::Vector{Tuple{Int, Int}}
    
    lwin::Int
    nsta::Int
    slomax2::T

    # Grillas
    s_grid::R
    s_grid_c::R
    s_grid_f::R

    # tamaños
    nite :: Int
    nite_c :: Int
    nite_f :: Int
    
    # Mapas
    ccmap::Matrix{T}
    ccmap_c::Matrix{T}
    ccmap_f::Matrix{T}
    
    # Buffers de trabajo
    benergy::Vector{T}
    beam::Vector{T}
    taper::Vector{Float64}
    fft_buf::Vector{ComplexF64}
end


function zlcc(data::AbstractArray, x::AbstractVector, y::AbstractVector, fs::Real, args...; kwargs...)
    
    SA = SeisArray2D(x, y, data, fs)
    
    return zlcc(SA, args...; kwargs...)
end


mutable struct ZLCCOutput{T<:AbstractFloat}
    time_s    :: Vector{T}
    maac      :: Vector{T}
    sx        :: Vector{T}
    sy        :: Vector{T}
    slow      :: Matrix{T}   # (nwin, 3)
    baz       :: Matrix{T}   # (nwin, 3)
    fpeak     :: Vector{T}
    beam      :: Vector{T}
    beam_max  :: Vector{T}
    s_ratio   :: Vector{T}
    slow_width:: Vector{T}
    baz_width :: Vector{T}
    smap      :: Vector{Union{Matrix{T}, Nothing}}
end


function zlcc(S::SeisArray2D, lwin::Int, nadv::T, fmin::T, fmax::T, slowmax::T, toff::T, slowint_c::T, slowint_f::T, ccerr::T, maac_th::T, slowfw::T, return_cmap::Bool) where {T<:AbstractFloat}
    
    # filtramos los datos
    filter!(S, fmin, fmax)

    # definición de ventanas
    npts, nsta = size(S.data)
    toff_samp = round(Int, toff * S.fs)
    step = round(Int, lwin * nadv)
    nwin = floor(Int, (npts - 2 * toff_samp - lwin) / step) + 1

    # inica buffers pre-allocados
    # if slowint_f == 0 (solo mapa grueso)
    n_threads = Threads.nthreads()
    thread_buffers = [
        init_zlcc_workspace(S, lwin, slowmax, slowint_c, slowfw, slowint_f)
    for _ in 1:n_threads
    ]

    # Crea el guardado
    nan1 = fill(T(NaN), nwin)       # vector 1D de NaN
    nan2 = fill(T(NaN), nwin, 3)    # matriz (nwin,3) de NaN
    out = ZLCCOutput{T}(
        copy(nan1),   # time_s
        copy(nan1),   # maac
        copy(nan1),   # sx
        copy(nan1),   # sy
        copy(nan2),   # slow
        copy(nan2),   # baz
        copy(nan1),   # fpeak
        copy(nan1),   # beam
        copy(nan1),   # beam_max
        copy(nan1),   # s_ratio
        copy(nan1),   # slow_width
        copy(nan1),   # baz_width
        fill!(Vector{Union{Matrix{T}, Nothing}}(undef, nwin), nothing)
        )

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

            # calcula metricas del beam
            beam = _power_beam(buf, n0, S.fs, best_sx, best_sy)

            # calcula la incertidumbre
            level = maac*ccerr
            uncert = uncertainty_contour(buf.s_grid, buf.s_grid, buf.ccmap, level)

            # guarda datos
            t0 = T(n0 - 1 - toff_samp) / T(S.fs)
            _save_window!(out, nk, t0, maac, best_sx, best_sy, beam, uncert, buf.ccmap, return_cmap)
        end
    end

    _mask_output(out)
end


function _save_window!(out::ZLCCOutput{T}, nk::Int, t0::T, maac::T, sx::T, sy::T, beam, uncert, smap, return_cmap::Bool) where {T<:AbstractFloat}

    out.time_s[nk]     = t0
    out.maac[nk]       = maac
    out.sx[nk]         = sx
    out.sy[nk]         = sy
    out.fpeak[nk]      = beam.fpeak
    out.beam[nk]       = beam.beam_rms
    out.beam_max[nk]   = beam.beam_max

    if !isnothing(uncert)
        out.slow[nk, 1] = uncert.slowmin
        out.slow[nk, 2] = uncert.slow
        out.slow[nk, 3] = uncert.slowmax
        out.baz[nk, 1]  = uncert.bazmin
        out.baz[nk, 2]  = uncert.baz
        out.baz[nk, 3]  = uncert.bazmax
        out.slow_width[nk] = uncert.sloww
        out.baz_width[nk]  = uncert.bazw
        out.s_ratio[nk]    = uncert.ratio
    end

    if return_cmap
        out.smap[nk]  = copy(smap)
    end
end


function _mask_output(out::ZLCCOutput{T}) where {T}
    mask = findall(!isnan, out.time_s)
    isempty(mask) && return nothing
    return ZLCCOutput{T}(
        out.time_s[mask],
        out.maac[mask],
        out.sx[mask],
        out.sy[mask],
        out.slow[mask, :],
        out.baz[mask, :],
        out.fpeak[mask],
        out.beam[mask],
        out.beam_max[mask],
        out.s_ratio[mask],
        out.slow_width[mask],
        out.baz_width[mask],
        out.smap[mask]
    )
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
    all_windows = haning_windows(lwin)
    taper = all_windows[lwin]

    N_pad = nextpow(2, lwin * 4)
    fft_buf = Vector{ComplexF64}(undef, N_pad)

    ws = ZLCC_WS_CPU(S.data, dx, dy, citer, lwin, nsta, slomax2, s_grid, s_grid_c, s_grid_f, nite, nite_c, nite_f, ccmap, ccmap_c, ccmap_f, benergy, beam, taper, fft_buf)

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


function _power_beam(ws::ZLCC_WS_CPU, n0::Int, fs, sx0, sy0)

    # Desempaquetar
    data = ws.data
    dx, dy = ws.dx, ws.dy
    beam = ws.beam
    lwin = ws.lwin
    nsta = ws.nsta
    fft_buf = ws.fft_buf
    taper = ws.taper

    # zero-allocation reset
    fill!(beam, 0.0)
    fill!(fft_buf, 0.0im)
    
    # Stacking (Beamforming)
    @inbounds for ii in 1:nsta
        delay = sx0 * dx[ii] + sy0 * dy[ii]
        idx_start = n0 + round(Int, delay) - 1

        @simd for k in 1:lwin
            beam[k] += data[idx_start + k, ii]
        end
    end

    # Root mean Square
    sum_sq = 0.0
    @inbounds @simd for k in 1:lwin
        val = beam[k]
        sum_sq = muladd(val, val, sum_sq)
    end
    beam_rms = sqrt(sum_sq / lwin) / nsta

    # Peak value and prepare fft_buf
    max_abs = 0.0
    @inbounds for i in 1:lwin
        val = beam[i]
        max_abs = max(max_abs, abs(val))
        fft_buf[i] = val * taper[i]
    end
    beam_max = max_abs / nsta

    # perform fft
    fft!(fft_buf)

    # pico de potencia espectral
    max_power = 0.0
    idx_max = 1
    half_n  = div(length(fft_buf), 2) + 1

    @inbounds for i in 1:half_n
        power = abs2(fft_buf[i])
        if power > max_power
            max_power = power
            idx_max = i
        end
    end

    fpeak = (idx_max - 1) * fs / length(fft_buf)

    return (; beam_rms, beam_max, fpeak)
end

# revisar el stack!
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


