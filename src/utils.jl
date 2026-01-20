#!/usr/local/bin julia
# coding=utf-8


function array_transfunc(S::SeisArray2D, slomax::Real, sloinc::Real, fmin::Real, fmax::Real, finc::Real)
    return array_transfunc(S.xcoord, S.ycoord, slomax, sloinc, fmin, fmax, finc)
end


function array_transfunc(x::Array{T}, y::Array{T}, slomax::T, sloinc::T, fmin::T, fmax::T, finc::T) where T<:Real

    freqs  = fmin:finc:fmax
    omegas = T(2) * π .* freqs
    s_vals = -slomax:sloinc:slomax

    n_s = length(s_vals)
    n_freq = length(freqs)
    n_sta = length(x)

    inv_nsta  = one(T) / nsta
    inv_nfreq = one(T) / n_freq

    power = zeros(T, n_s, n_s)

    Threads.@threads for j in 1:n_s
        sy = s_vals[j]

        for i in 1:n_s
            sx = s_vals[i]

            sum_power_freq = zero(T)

            for ω in omegas
                beam_sum = zero(Complex{T})

                @simd for k in 1:n_sta
                    delay = sx * x[k] + sy * y[k]
                    beam_sum += cis(ω * delay)
                end

                sum_power_freq += abs2(beam_sum * inv_nsta)
            end

            power[i, j] = sum_power_freq * inv_nfreq
        end
    end

    return s_vals, power
end



function circular_stats(angles::AbstractVector{T}, w::AbstractVector{T}) where T<:Real
    # Convertir a radianes
    rads = deg2rad.(angles)
    
    # Promediar las componentes vectoriales
    sum_s = sum(w .* sin.(rads))
    sum_c = sum(w .* cos.(rads))
    sum_w = sum(w)

    # Calcular el ángulo resultante
    avg_rad = atan(sum_s, sum_c)
    baz_avg = mod(rad2deg(avg_rad), 360)

    R = hypot(sum_s, sum_c) / sum_w
    baz_std = rad2deg(sqrt(-2 * log(clamp(R, 1e-10, 1.0))))

    return baz_avg, baz_std
end



function cciter(nsta::J) where J<:Integer
    cciter = Vector{Tuple{J,J}}()
    for ii in 1:nsta-1
        for jj in ii+1:nsta
            push!(cciter, (ii, jj))
        end
    end

    return cciter
end



function cross_pair_dist(S::SeisArray2D, pairs)
    cross_pair_dist(S.xcoord, S.ycoord, pairs)
end



function cross_pair_dist(x_coords, y_coords, pairs)
    num_pairs = length(pairs)
    dx_full = zeros(num_pairs)
    dy_full = zeros(num_pairs)

    @inbounds for k in 1:num_pairs
        i, j = pairs[k]
        dx_full[k] = x_coords[j] - x_coords[i]
        dy_full[k] = y_coords[j] - y_coords[i]
    end

    return dx_full, dy_full
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



function slowess_linear(dx, dy, dt)
    
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



function init_triads(nsta)
    pairs = cciter(nsta)
    trios = init_triads(nsta, pairs)
    return pairs, trios
end


function init_triads(nsta, pairs)
    triangles = Vector{TriangleDef}()
    pair_map = zeros(Int, nsta, nsta)

    @inbounds for (idx, p) in enumerate(pairs)
        u, v = p
        pair_map[u, v] = idx
        pair_map[v, u] = idx
    end

    sizehint!(triangles, div(nsta^3, 6))

    # Busca triangulos i < j < k
    @inbounds for i in 1:nsta
        for j in (i+1):nsta
            for k in (j+1):nsta

                idx_ij = pair_map[i, j]
                idx_jk = pair_map[j, k]
                idx_ki = pair_map[k, i]

                if idx_ij > 0 && idx_jk > 0 && idx_ki > 0
                    s1 = (pairs[idx_ij] == (i,j)) ? 1.0 : -1.0
                    s2 = (pairs[idx_jk] == (j,k)) ? 1.0 : -1.0
                    s3 = (pairs[idx_ki] == (k,i)) ? 1.0 : -1.0

                    trio = (i, j, k)
                    push!(triangles, TriangleDef(idx_ij, idx_jk, idx_ki, s1, s2, s3, trio))
                end
            end
        end
    end

    return triangles
end





