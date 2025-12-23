

module GPUzlcc

    using CUDA, SAP

    struct BaseCuda{T<:AbstractFloat, J<:Integer} <: AbstractBase
        nsta    :: J
        lwin    :: J
        citer_i :: CuArray{Int32, 1}
        citer_j :: CuArray{Int32, 1}
        sx      :: CuArray{T, 1}
        sy      :: CuArray{T, 1}
        dx      :: CuArray{T, 1}
        dy      :: CuArray{T, 1}
    end

    function _zlcc_kernel!(cc_map, data, n0, sx, sy, dx, dy, citer_i, citer_j, nsta, lwin)
        # indices mapeados a la grilla de lentitud
        ii = (blockIdx().x - 1) * blockDim().x + threadIdx().x
        jj = (blockIdx().y - 1) * blockDim().y + threadIdx().y

        if ii <= size(cc_map, 1) && jj <= size(cc_map, 2)
            px = sx[ii]
            py = sy[jj]

            cc_sum = 0.0f0 # Usamos Float32
            n_pairs = length(citer_i)

            for p in 1:n_pairs
                sta_i = citer_i[p]
                sta_j = citer_j[p]

                idx_i = round(Int32, n0 + px * dx[sta_i] + py * dy[sta_i])
                idx_j = round(Int32, n0 + px * dx[sta_j] + py * dy[sta_j])

                # Producto punto manual
                dot_ij = 0.0f0
                norm_i = 0.0f0
                norm_j = 0.0f0

                for t in 0:lwin
                    val_i = data[idx_i + t, sta_i]
                    val_j = data[idx_j + t, sta_j]
                
                    dot_ij += val_i * val_j
                    norm_i += val_i * val_i
                    norm_j += val_j * val_j
                end

                # Sumar al total con normalización
                if norm_i > 0 && norm_j > 0
                    cc_sum += dot_ij / (sqrt(norm_i) * sqrt(norm_j))
                end
            end

            cc_map[ii, jj] = (2.0f0 * cc_sum + nsta) / (Float32(nsta) * Float32(nsta))

        end
        
        return nothing
    end

    function _ccmap_gpu!(cc_map::CuArray, data::CuArray, n0, sx, sy, dx, dy, citer_i, citer_j, nsta, lwin)
        nite_x = length(sx)
        nite_y = length(sy)
        threads = (16, 16)
        blocks = (ceil(Int, nite_x / threads[1]), ceil(Int, nite_y / threads[2]))

        @cuda threads=threads blocks=blocks _zlcc_kernel!(
            cc_map, data, n0, sx, sy, dx, dy, citer_i, citer_j, nsta, lwin
        )

        return cc_map
    end

    function SAP._ccmap!(cc_map::CuArray, data::CuArray, n0, base::BaseCuda)
        _ccmap_gpu!(cc_map, data, Float32(n0), base.sx, base.sy, base.dx, base.dy, base.citer_i, base.citer_j, base.nsta, base.lwin)
    end

end