#!/usr/local/bin julia
# coding=utf-8


function cciter(nsta::J) where J<:Integer
    cciter = Vector{Tuple{J,J}}()
    for ii in 1:nsta-1
        for jj in ii+1:nsta
            push!(cciter, (ii, jj))
        end
    end

    return cciter
end


function distance_matrix(S::SeisArray2D)
    distance_matrix(S.xcoord, S.ycoord)
end


function distance_matrix(x_coords::AbstractVector{T}, y_coords::AbstractVector{T}) where {T<:AbstractFloat}

    N     = length(x_coords)
    mdx   = zeros(T, N, N)
    mdy   = zeros(T, N, N)
    mdist = zeros(T, N, N)

    @inbounds for i in 1:N, j in i+1:N
        dx = x_coords[j] - x_coords[i]
        dy = y_coords[j] - y_coords[i]
        d  = sqrt(dx*dx + dy*dy)
        mdx[i,j]   =  dx
        mdx[j,i]   = -dx
        mdy[i,j]   =  dy
        mdy[j,i]   = -dy
        mdist[i,j] =   d
        mdist[j,i] =   d
    end

    return mdx, mdy, mdist
end


function init_triads(x_coords::AbstractVector{T}, y_coords::AbstractVector{T}) where {T}

    @assert length(x_coords) == length(y_coords) "x_coords y y_coords deben tener la misma longitud"

    nsta   = length(x_coords)
    triads = Vector{TriangleDef}()
    sizehint!(triads, div(nsta * (nsta-1) * (nsta-2), 6))

    # matriz de distancia (simétrica de diagonal cero)
    mdx, mdy, mdist = distance_matrix(x_coords, y_coords)

    @inbounds for i in 1:nsta, j in i+1:nsta, k in j+1:nsta
        a = mdist[i,j]
        b = mdist[j,k]
        c = mdist[k,i]
        dx_ij = mdx[i,j]
        dx_jk = mdx[j,k]
        dx_ki = mdx[k,i]
        dy_ij = mdy[i,j]
        dy_jk = mdy[j,k]
        dy_ki = mdy[k,i]
        dmin = min(a,b,c)
        dmax = max(a,b,c)
        push!(triads, TriangleDef(i, j, k, dmin, dmax, dx_ij, dx_jk, dx_ki, dy_ij, dy_jk, dy_ki))
    end

    return mdist, triads
end


function init_triads(S::SeisArray2D)
    return init_triads(S.xcoord, S.ycoord)
end

