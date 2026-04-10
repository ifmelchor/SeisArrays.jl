#!/usr/local/bin julia
# coding=utf-8



function contour_size(x::AbstractVector{T}, y::AbstractVector{T}) where T<:Real
    area = 0.0
    n = length(x)
    @inbounds for i in 1:n-1
        area += x[i] * y[i+1] - x[i+1] * y[i]
    end
    return abs(area + x[n]*y[1] - x[1]*y[n]) / 2.0
end



function polygon_centroid(x::AbstractVector{T}, y::AbstractVector{T}) where T<:Real
    
    n = length(x)
    if n < 3
        return sum(x)/n, sum(y)/n
    end

    A  = zero(float(T))
    Cx = zero(float(T))
    Cy = zero(float(T))

    # Shoelace
    @inbounds for i in 1:n
        j = (i == n) ? 1 : i + 1
        cross = x[i] * y[j] - x[j] * y[i]
        A  += cross
        Cx += (x[i] + x[j]) * cross
        Cy += (y[i] + y[j]) * cross
    end

    area = A * 0.5

    # área cercana a 0
    if abs(area) < sqrt(eps(float(T)))
        return sum(x)/n, sum(y)/n
    end

    factor = 1.0 / (6.0 * area)
    
    return Cx * factor, Cy * factor
end



function uncertainty_contour(sx, sy, zmap, level)

    ratio  = NaN
    circty = NaN
    radii  = NaN
    slobnd = [NaN, NaN, NaN, NaN]
    bazbnd = [NaN, NaN, NaN, NaN]

    c = contours(sx, sy, zmap, [level])

    if isempty(levels(c)) || isempty(lines(levels(c)[1]))
        return ratio, circty, radii, slobnd, bazbnd
    end

    cl = lines(levels(c)[1])
    ncontour = length(cl)

    if ncontour == 0
        return ratio, circty, radii, slobnd, bazbnd
    end
    
    area_blob = 0.0
    idx = 0
    if ncontour == 1
        Xs, Ys    = coordinates(cl[1])
        area_blob = contour_size(Xs, Ys)
        idx = 1
        ratio = NaN
    else
        # si hay mas de un contorno, 
        # toma por bueno el contorno dominante
        sizes = zeros(ncontour)
        for nc in 1:ncontour
            Xs, Ys = coordinates(cl[nc])
            sizes[nc] = contour_size(Xs, Ys)
        end

        sort_idx = sortperm(sizes, rev=true)
        max_s = sizes[sort_idx[1]] # dominante
        sec_s = sizes[sort_idx[2]] # secundario

        # calcula la relacion entre dominante y secundario
        ratio = sec_s / max_s

        area_blob = max_s
        idx = sort_idx[1]
    end

    Xs, Ys = coordinates(cl[idx])

    # calcula la circularidad
    perim = 0.0
    n_pts = length(Xs)
    @inbounds for i in 1:n_pts-1
        perim += hypot(Xs[i+1]-Xs[i], Ys[i+1]-Ys[i])
    end
    perim += hypot(Xs[1]-Xs[n_pts], Ys[1]-Ys[n_pts])
    circty = (perim > 0) ? (4 * π * area_blob) / (perim^2) : 0.0

    # calcula el centroide
    sx_mean, sy_mean = polygon_centroid(Xs, Ys)
    s_mean   = hypot(sx_mean, sy_mean)
    baz_mean = mod(atand(-sx_mean,-sy_mean), 360.0)

    # limites de lentitud
    slo_vec = hypot.(Xs, Ys)
    s_max = maximum(slo_vec)
    s_min = minimum(slo_vec)
    s_width = s_max-s_min
    slobnd = [s_min, s_mean, s_max, s_width]

    # limites de back-azimuth
    ang = mod.(atand.(-Xs, -Ys), 360.0)
    sort!(ang)
    diffs = diff(ang)
    push!(diffs, 360.0 - (ang[end] - ang[1]))
    max_gap, idx_gap = findmax(diffs)
    width = 360.0 - max_gap
    
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
    bazbnd = [baz_min, baz_mean, baz_max, width]

    if width < 180
        # calcula el radio del lóbulo de incertidumbre
        radii = sqrt(0.5*s_width*s_mean * sin(0.5*width*π/180))
    end

    return ratio, circty, radii, slobnd, bazbnd
end


function interpolate_grids!(buffer_grid::AbstractMatrix, coarse_grid::AbstractMatrix, fine_grid::AbstractMatrix, buffer_range::AbstractVector, coarse_range::AbstractVector, fine_range_x::AbstractVector, fine_range_y::AbstractVector)

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
    inv_step = 1.0 / step(buffer_range)
    fine_min_x, fine_max_x = first(fine_range_x), last(fine_range_x)
    fine_min_y, fine_max_y = first(fine_range_y), last(fine_range_y)

    # Eje X
    raw_start_x = (fine_min_x - buf_min) * inv_step + 1
    raw_end_x   = (fine_max_x - buf_min) * inv_step + 1
    
    idx_start_x = clamp(ceil(Int, raw_start_x), 1, N_buffer)
    idx_end_x   = clamp(floor(Int, raw_end_x),  1, N_buffer)
    
    # Eje Y
    raw_start_y = (fine_min_y - buf_min) * inv_step + 1
    raw_end_y   = (fine_max_y - buf_min) * inv_step + 1

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
