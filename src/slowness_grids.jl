#!/usr/local/bin julia
# coding=utf-8

function interpolate_grids!(buffer_grid::AbstractMatrix{T}, coarse_grid::AbstractMatrix{T}, fine_grid::AbstractMatrix{T}, buffer_range::AbstractVector{T}, coarse_range::AbstractVector{T}, fine_range_x::AbstractVector{T}, fine_range_y::AbstractVector{T}) where T<:Real

    N_buffer = size(buffer_grid, 1)

    # Interpolador Coarse
    itp_coarse = scale(interpolate(coarse_grid, BSpline(Linear())), coarse_range, coarse_range)
    etp_coarse = extrapolate(itp_coarse, Flat())

    @inbounds for j in 1:N_buffer
        sy = buffer_range[j]
        @simd for i in 1:N_buffer
            sx = buffer_range[i]
            buffer_grid[i, j] = etp_coarse(sx, sy)
        end
    end

    # Interpolador Fine
    itp_fine = scale(interpolate(fine_grid, BSpline(Linear())), fine_range_x, fine_range_y)
    etp_fine = extrapolate(itp_fine, Flat())

    # calcular índices del parche fino y los extremos
    buf_min  = first(buffer_range)
    inv_step = one(T) / step(buffer_range)
    fine_min_x, fine_max_x = first(fine_range_x), last(fine_range_x)
    fine_min_y, fine_max_y = first(fine_range_y), last(fine_range_y)

    # Eje X
    raw_start_x = (fine_min_x - buf_min) * inv_step + one(T)
    raw_end_x   = (fine_max_x - buf_min) * inv_step + one(T)
    
    idx_start_x = clamp(ceil(Int, raw_start_x), 1, N_buffer)
    idx_end_x   = clamp(floor(Int, raw_end_x),  1, N_buffer)
    
    # Eje Y
    raw_start_y = (fine_min_y - buf_min) * inv_step + one(T)
    raw_end_y   = (fine_max_y - buf_min) * inv_step + one(T)

    idx_start_y = clamp(ceil(Int, raw_start_y), 1, N_buffer)
    idx_end_y   = clamp(floor(Int, raw_end_y),  1, N_buffer)

    @inbounds for j in idx_start_y:idx_end_y
        sy = buffer_range[j]
        @simd for i in idx_start_x:idx_end_x
            sx = buffer_range[i]
            buffer_grid[i, j] = etp_fine(sx, sy)
        end
    end
end


function get_slowness_coord(slow::T, bazm::T, slow0::Vector{T}, slowmax::T, slowint::T) where T<:Real

    rad = deg2rad(bazm)
    px_theo = -slow * sin(rad)
    py_theo = -slow * cos(rad)

    r  = -slowmax:slowint:slowmax
    sx = collect(r .+ slow0[1])
    sy = collect(r .+ slow0[2])
    ii = argmin(abs.(sx .- px_theo))
    jj = argmin(abs.(sy .- py_theo))

    px = sx[ii]
    py = sy[jj]

    return px, py
end


function get_delays(slow::T, bazm::T, slow0::Vector{T}, slowmax::T, slowint::T, xcoord::Array{T}, ycoord::Array{T}) where T<:Real

    px, py  = get_slowness_coord(slow, bazm, slow0, slowmax, slowint)

    nsta = length(xcoord)
    xref = mean(xcoord)
    yref = mean(ycoord)
    dx = (xcoord .- xref)
    dy = (ycoord .- yref)

    # Calcular Delta Times en segundos
    dt = [(px * dx[i] + py * dy[i]) for i in 1:nsta]

    return dt, [px, py]
end


function contour_size(x::AbstractVector{T}, y::AbstractVector{T}) where T<:Real

    area = zero(T)
    n = length(x)
    @inbounds for i in 1:n-1
        area += x[i] * y[i+1] - x[i+1] * y[i]
    end
    return abs(area + x[n]*y[1] - x[1]*y[n]) / T(2)
end


function polygon_centroid(x::AbstractVector{T}, y::AbstractVector{T}) where T<:Real
    
    n = length(x)
    if n < 3
        return sum(x)/n, sum(y)/n
    end

    A  = zero(T)
    Cx = zero(T)
    Cy = zero(T)

    # Shoelace
    @inbounds for i in 1:n
        j = (i == n) ? 1 : i + 1
        cross = x[i] * y[j] - x[j] * y[i]
        A  += cross
        Cx += (x[i] + x[j]) * cross
        Cy += (y[i] + y[j]) * cross
    end

    area = A * T(0.5)

    # área cercana a 0
    if abs(area) < sqrt(eps(T))
        return sum(x)/n, sum(y)/n
    end

    factor = T(1.0) / (T(6.0) * area)
    
    return Cx * factor, Cy * factor
end


function uncertainty_contour(sx, sy, zmap, level)

    # Calcular contornos
    c = contours(sx, sy, zmap, [level])

    if isempty(levels(c)) || isempty(lines(levels(c)[1]))
        return nothing
    end

    cl = lines(levels(c)[1])
    ncontour = length(cl)

    if ncontour == 0
        return nothing
    end
    
    ratio = NaN
    idx = 1

    if ncontour == 1
        Xs, Ys    = coordinates(cl[1])
    else
        # si hay mas de un contorno, 
        # toma por bueno el contorno dominante
        sizes = map(1:ncontour) do nc
            Xs, Ys = coordinates(cl[nc])
            contour_size(Xs, Ys)
        end

        sort_idx = sortperm(sizes, rev=true)
        max_s = sizes[sort_idx[1]] # dominante
        sec_s = sizes[sort_idx[2]] # secundario

        # calcula la relacion entre dominante y secundario
        ratio = sec_s / max_s
        idx = sort_idx[1]
    end

    # coordenadas del contorno dominante
    Xs, Ys = coordinates(cl[idx])

    # calcula el centroide
    sx_mean, sy_mean = polygon_centroid(Xs, Ys)
    s_mean   = hypot(sx_mean, sy_mean)
    baz_mean = mod(atand(-sx_mean,-sy_mean), 360.0)

    # limites de lentitud
    slo_vec = hypot.(Xs, Ys)
    s_min, s_max = extrema(slo_vec)
    s_width = s_max-s_min

    # limites de back-azimuth
    ang = mod.(atand.(-Xs, -Ys), 360.0)
    sort!(ang)

    diffs = diff(ang)
    push!(diffs, 360.0 - (ang[end] - ang[1]))

    max_gap, idx_gap = findmax(diffs)
    baz_width = 360.0 - max_gap
    
    if max_gap > 180.0
        if idx_gap == length(diffs)
            baz_min = ang[1]
            baz_max = ang[end]
        else
            baz_min = ang[idx_gap + 1]
            baz_max = ang[idx_gap]
        end
    else
        baz_min, baz_max = minimum(ang), maximum(ang)
    end

    return (
        ratio = ratio,
        slowmin = s_min,
        slow = s_mean,
        slowmax = s_max,
        sloww = s_width,
        bazmin = baz_min,
        baz = baz_mean,
        bazmax = baz_max,
        bazw = baz_width,
    )
end
