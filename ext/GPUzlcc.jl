

module GPUzlcc

    using CUDA, SeisArrays
    using StaticArrays

    const MAX_GPU_STA = 64

    struct ZLCC_WS_GPU{T, A<:CuArray, V<:CuArray}
        data::A             # (npts x nsta) en GPU
        dx::V; dy::V        # Vectores geometría
        sx::V; sy::V        # Grilla Coarse
        ccmap::A            # Mapa de salida
        
        lwin::Int32
        nsta::Int32
        slomax_sq::Float32
    end

    function _zlcc_kernel_opt!(cc_map, data, n0, sx, sy, dx, dy, nsta, lwin, limit_sq)
        # Índices de la grilla de lentitud
        i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
        j = (blockIdx().y - 1) * blockDim().y + threadIdx().y

        if i <= length(sx) && j <= length(sy)
            px = sx[i]
            py = sy[j]

            # Chequeo circular
            if (px^2 + py^2) > limit_sq
                cc_map[i, j] = 0.0f0
                return
            end

            inv_norms = MVector{MAX_GPU_STA, Float32}(undef)
            shifts    = MVector{MAX_GPU_STA, Int32}(undef)

             # Iterar sobre estaciones para calcular su energía en este shift específico
            for s in 1:nsta
                sh = round(Int32, px * dx[s] + py * dy[s])
                shifts[s] = sh
                start_idx = n0 + sh
            
                # Calcular suma de cuadrados (Energía) para esta estación/ventana
                sq_sum = 0.0f0
                for k in 0:(lwin-1)
                    val = data[start_idx + k, s]
                    sq_sum += val * val
                end
                # Guardar el inverso de la raíz (1/RMS) para multiplicar luego
                # Agregamos un epsilon pequeño para evitar división por cero
                inv_norms[s] = 1.0f0 / sqrt(sq_sum + 1.0f-12)
            end

            beam_energy = 0.0f0

            # Loop sobre el tiempo (ventana)
            for t in 0:(lwin-1)
                beam_sample = 0.0f0
                
                # Loop sobre estaciones (O(N))
                for s in 1:nsta
                    sh = shifts[s]
                    val = data[n0 + sh + t, s]
                    beam_sample += val * inv_norms[s]
                end

                beam_energy += beam_sample * beam_sample
            end

            # Fórmula ZLCC basada en Beam Power
            nsta_f = Float32(nsta)
            cc_map[i, j] = (beam_energy - nsta_f) / (nsta_f * (nsta_f - 1.0f0))
        end
        return nothing
    end

    function _compute_ccmap!(ws::ZLCC_ws, n0::Int)

        threads = (16, 16)
        blocks = (cld(length(ws.sx), 16), cld(length(ws.sy), 16))

        if ws.nsta > MAX_GPU_STA
            error("El número de estaciones excede MAX_GPU_STA. Aumenta la constante en el kernel.")
        end
        
        @cuda threads=threads blocks=blocks _zlcc_kernel_opt!(
            ws.ccmap, 
            ws.data, 
            Int32(n0), 
            ws.sx, ws.sy, 
            ws.dx, ws.dy, 
            Int32(ws.nsta), 
            Int32(ws.lwin), 
            Float32(ws.slomax_sq) # Asegurar que sea Float32
        )
        return nothing
    end
end