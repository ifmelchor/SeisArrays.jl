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



function uncertainty_contour(sx, sy, zmap, level, ratio_max=0.05, C_th=0.3)

    c = contours(sx, sy, zmap, [level])

    if isempty(levels(c)) || isempty(lines(levels(c)[1]))
        return false, NaN, [NaN, NaN], [NaN, NaN], [NaN, NaN]
    end

    cl = lines(levels(c)[1])
    ncontour = length(cl)
    is_good   = false
    area_blob = 0.0

    if ncontour == 0
        is_good = false
        ratio = NaN

    elseif ncontour == 1
        Xs, Ys    = coordinates(cl[1])
        area_blob = contour_size(Xs, Ys)
        is_good = true
        idx = 1
        ratio = -1.
        
    else
        # si hay mas de un contorno, 
        # debemos analizar si el contorno es significativo
        sizes = zeros(ncontour)
        for nc in 1:ncontour
            Xs, Ys = coordinates(cl[nc])
            sizes[nc] = contour_size(Xs, Ys)
        end

        sort_idx = sortperm(sizes, rev=true)
        max_s = sizes[sort_idx[1]] # dominante
        sec_s = sizes[sort_idx[2]] # secundario
        ratio = sec_s / max_s

        if ratio > ratio_max
            # Ambigüedad real
            is_good = false
            # no calcula nada
        else
            # toma por bueno el contorno dominante
            is_good   = true
            area_blob = max_s
            idx = sort_idx[1]
        end
    end

    if is_good
        # las coordenadas
        Xs, Ys = coordinates(cl[idx])

        # calcula la circularidad
        perimeter = 0.0
        n_pts = length(Xs)
        @simd for i in 1:n_pts-1
            perimeter += hypot(Xs[i+1]-Xs[i], Ys[i+1]-Ys[i])
        end
        perimeter += hypot(Xs[1]-Xs[n_pts], Ys[1]-Ys[n_pts])
        circularity = (perimeter > 0) ? (4 * π * area_blob) / (perimeter^2) : 0.0

        if circularity < C_th
            sx_mean = sy_mean = NaN
            slobnd = bazbnd = [NaN, NaN]
            return false, ratio, [sx_mean, sy_mean], slobnd, bazbnd
        end

        # calcula el centro geometrico
        sx_mean, sy_mean = polygon_centroid(Xs, Ys)

        # calcula el vector de lentitud medio
        s_mean = hypot(sx_mean, sy_mean)
        baz_mean = mod(atand(-sx_mean,-sy_mean), 360.0)

        # calcula la incertidumbre dada por el contorno
        slo_vec = hypot.(Xs, Ys)
        s_max = maximum(slo_vec)
        s_min = minimum(slo_vec)
        slobnd = [s_min, s_mean, s_max, s_max-s_min]

        ang = mod.(atand.(-Xs, -Ys), 360.0)
        sort!(ang)
        diffs = diff(ang)
        push!(diffs, 360.0 - (ang[end] - ang[1]))
        max_gap, idx_gap = findmax(diffs)
        width = 360.0 - max_gap
        if max_gap > 180.0
            if idx_gap == length(diffs)
                # El hueco es el cierre normal (no cruza el norte)
                baz_min = ang[1]
                baz_max = ang[end]
            else
                # El hueco está en medio -> Los datos cruzan el norte
                baz_min = ang[idx_gap + 1]
                baz_max = ang[idx_gap]
            end
        else
            baz_min, baz_max = minimum(ang), maximum(ang)
        end
        
        bazbnd = [baz_min, baz_mean, baz_max, width]

    else
        sx_mean = sy_mean = NaN
        slobnd = bazbnd = [NaN, NaN]
    end

    return is_good, ratio, [sx_mean, sy_mean], slobnd, bazbnd
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

    replace!(buffer_grid, NaN => Inf)
end
