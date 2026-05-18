#!/usr/local/bin julia
# coding=utf-8


function cross_pair_dist(S::SeisArray2D, pairs)
    cross_pair_dist(S.xcoord, S.ycoord, pairs)
end



function cross_pair_dist(x_coords, y_coords, pairs)
    num_pairs = length(pairs)
    dx_full = zeros(num_pairs)
    dy_full = zeros(num_pairs)
    dist_full = zeros(num_pairs)
    @inbounds for k in 1:num_pairs
        i, j = pairs[k]
        dx = x_coords[j] - x_coords[i]
        dy = y_coords[j] - y_coords[i]
        dx_full[k] = dx
        dy_full[k] = dy
        dist_full[k] = sqrt(dx*dx + dy*dy)
    end

    return dx_full, dy_full, dist_full
end



struct TriangleDef
    # indicadores de la triada
    sta_triad::Tuple{Int, Int, Int}

    # indicides de los pares
    p1_idx::Int
    p2_idx::Int
    p3_idx::Int

    # signo del delay
    s1::Float64
    s2::Float64
    s3::Float64

    # distancia minima/maxima/media del triangulo
    dmin::Float64
    dmax::Float64
    dmean::Float64
end



function init_triads(S::SeisArray2D)
    init_triads(S.xcoord, S.ycoord)
end



function init_triads(x_coords, y_coords)
    nsta = length(x_coords)
    
    # calcula el centroide
    # centroid_x = sum(x_coords) / nsta
    # centroid_y = sum(y_coords) / nsta
    # centroid = (centroid_x, centroid_y)

    # genera los pares
    pairs = cciter(nsta)

    # Calcula las métricas de los pares
    dx, dy, dd = cross_pair_dist(x_coords, y_coords, pairs)

    # Construimos las tríadas
    trios = init_triads(nsta, pairs, dd)

    return pairs, dx, dy, dd, trios
end



function init_triads(nsta::Int, pairs, dd)
    triangles = Vector{TriangleDef}()
    pair_map = zeros(Int, nsta, nsta)

    # Mapeo de pares
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
                trio = (i, j, k)
                idx_ij = pair_map[i, j]
                idx_jk = pair_map[j, k]
                idx_ki = pair_map[k, i]

                if idx_ij > 0 && idx_jk > 0 && idx_ki > 0
                    # Lee distancia máxima/minima/media de esta tríada
                    a = dd[idx_ij]
                    b = dd[idx_jk]
                    c = dd[idx_ki]
                    dmin = min(a, b, c)
                    dmax = max(a, b, c)
                    dmean = (a+b+c)/3

                    # Determinamos los signos
                    s1 = (pairs[idx_ij] == (i,j)) ? 1.0 : -1.0
                    s2 = (pairs[idx_jk] == (j,k)) ? 1.0 : -1.0
                    s3 = (pairs[idx_ki] == (k,i)) ? 1.0 : -1.0

                    push!(triangles, TriangleDef(trio, idx_ij, idx_jk, idx_ki, s1, s2, s3, dmin, dmax, dmean))
                end
            end
        end
    end

    return triangles
end




