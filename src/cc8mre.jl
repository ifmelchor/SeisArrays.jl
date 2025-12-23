#!/usr/local/bin julia
# coding=utf-8

# GNU GPL v2 licenced to I. Melchor and J. Almendros 08/2022
# Main ZLCC codes

function zlcc(data::Array{T}, xStaUTM::Array{T}, yStaUTM::Array{T}, slomax::T, sloint::T, fqband::Vector{T}, fsem::J, lwin::J, nwin::J, nadv::T, toff::J, slow0::Vector{T}=[0., 0.], ccerr::T=0.95, slow2::Bool=true, maac_thr::T=0.5, slomax2::T=0.3, sloint2::T=0.02, ratio_max::T=0.05, save_maps::Bool=true, use_gpu::Bool=false) where {T<:Real, J<:Integer}
    
    # define algos parametros params
    nsta   = length(xStaUTM) # nro of stations
    cciter = _cciter(nsta)   # define stations iterator
    toff   = toff*fsem       # convert from seconds to samples

    # definimos matrices precalculadas
    nite   = 1 + 2*round(Int64, slomax/sloint)

    # preparamos los datos : filter and traspose data
    data = collect(data')
    data = _filter(data, fsem, fqband)

    # si el cuda esta instalado
    if use_gpu
        if isdefined(Main, :CUDA)
            CU = Main.CUDA
            cudata = CU.CuArray{Float32}(data)
            cuda_ccmap = CU.zeros(Float32, nite, nite) 
            ccmap = zeros(Float32, nite, nite)
        else
            @warn "CUDA.jl no está cargado. Usando CPU..."
            ccmap = zeros(Float64, nite, nite)
            energy_ws = zeros(Float64, nsta)
            use_gpu = false
        end
    else
        ccmap = zeros(Float64, nite, nite)
        energy_ws = zeros(Float64, nsta)
    end

    if slow2
        nite2  = 1 + 2*round(Int64, slomax2/sloint2)
        ccmap2 = zeros(Float64, nite2, nite2)
        energy_ws2 = zeros(Float64, nsta)
    end

    # create slowness vectorial grid
    r  = -slomax:sloint:slomax
    sx = collect(r .+ slow0[1])
    sy = collect(r .+ slow0[2])
    r2 = -slomax2:sloint2:slomax2

    # Array center and relative displacements
    xref = mean(xStaUTM)
    yref = mean(yStaUTM)

    # compute with fsem to avoid future multiplication
    dx = (xStaUTM .- xref) .* fsem
    dy = (yStaUTM .- yref) .* fsem

    # base object for zlcc
    base = BaseZLCC(nite, nwin, nsta, lwin, cciter, sx, sy, dx, dy)

    # base for cuda
    if use_gpu
        GPU_Ext = Base.get_extension(SAP, :GPUzlcc)
        base_gpu = GPU_Ext.BaseCuda(Int32(nsta), Int32(lwin),
            CU.CuArray{Int32}([c[1] for c in cciter]),
            CU.CuArray{Int32}([c[2] for c in cciter]),
            CU.CuArray{Float32}(sx),
            CU.CuArray{Float32}(sy),
            CU.CuArray{Float32}(dx),
            CU.CuArray{Float32}(dy))
    end

    # init empty dictionary
    dict = _empty_dict(base, save_maps)

    # iterate over time
    @inbounds for nk in 1:nwin
        # n0  = 1 + toff + lwin*nadv*(nk-1)
        n0 = 1 + toff + lwin * nadv * (nk - 1)

        # get ccmap
        if use_gpu
            _ccmap!(cuda_ccmap, cudata, n0, base_gpu)
            copyto!(ccmap, cuda_ccmap)
        else
            _ccmap!(ccmap, data, n0, base, energy_ws)
        end

        # find max value MAAC and position
        maac, midx = findmax(ccmap)
        ci = CartesianIndices(ccmap)[midx]
        ii, jj = ci.I
        best_sx = base.sx[ii]
        best_sy = base.sy[jj]

        # save data of the iteration
        dict["maac"][nk]   = maac

        if save_maps
            dict["slowmap"][nk,:,:] = ccmap
        end

        # compute contour 
        c = contours(base.sx, base.sy, ccmap, [maac*ccerr])
        cl = lines(levels(c)[1])
        ncontour = length(cl)
        is_good = false

        if ncontour == 0
            is_good = false
            ratio = NaN
            # no calcula nada

        elseif ncontour == 1
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
                is_good = true
                idx = sort_idx[1]
            end
        end

        # save the ratio
        dict["slow_ratio"][nk] = ratio

        if is_good
            # calcula la incertidumbre
            Xs, Ys = coordinates(cl[idx])
            ans = r2p.(.-Xs, .-Ys)
            slo = first.(ans)
            s_max = maximum(slo)
            s_min = minimum(slo)
            slobnd = [s_min, s_max]

            ang = last.(ans)
            ang_max = maximum(ang)
            ang_min = minimum(ang)
            unc_lin = ang_max - ang_min
            unc_com = 360.0 - unc_lin
            if unc_lin <= unc_com
                baz_max = ang_max
                baz_min = ang_min
            else
                baz_max = ang_min
                baz_min = ang_max
            end
            bazbnd = [baz_min, baz_max]

            if slow2 && maac > maac_thr
                # compute slownes map with nite2
                sx2 = r2 .+ best_sx
                sy2 = r2 .+ best_sy
                _ccmap!(ccmap2, data, n0, sx2, sy2, base, energy_ws2)

                maac2, midx2 = findmax(ccmap2)
                ii, jj = midx2.I
                best_sx = sx2[ii]
                best_sy = sy2[jj]
            end

            slow, bazm  = r2p(-best_sx, -best_sy)
            rms  = _power_beam(data, n0, best_sx, best_sy, base)
        
        else
            slow = bazm = rms = NaN
            bazbnd = slobnd = [NaN,NaN]
        end

        # save more data
        dict["rms"][nk]    = rms
        dict["slow"][nk]   = slow
        dict["baz"][nk]    = bazm
        dict["slowbnd"][nk,:] = slobnd
        dict["bazbnd"][nk,:]  = bazbnd
    end

    return dict
end

function zlcc_stack(zlcc_out::Dict, slomax::T, sloint::T, maac_th::T, baz_th::T, baz_lim::Union{Vector{T}, Nothing}=nothing, ccerr::T=0.95, min_nidx::J=10, ratio_max::T=0.1, slow0::Vector{T}=[0., 0.]) where {T<:Real, J<:Integer}

    r  = -slomax:sloint:slomax
    sx = collect(r .+ slow0[1])
    sy = collect(r .+ slow0[2])

    N = length(zlcc_out["maac"])
    condMAAC = zlcc_out["maac"] .> maac_th

    bazbnd = zlcc_out["bazbnd"]
    az1 = @. ifelse(bazbnd[:, 1] > 360, NaN, bazbnd[:, 1])
    az2 = @. ifelse(bazbnd[:, 2] > 360, NaN, bazbnd[:, 2])
    azbnd = @. abs(mod(az2 - az1 + 180, 360) - 180)
    condBAZ = azbnd .< baz_th

    if baz_lim !== nothing
        baz_min = baz_lim[1]
        baz_max = baz_lim[2]
        condBAZ_LIM = (zlcc_out["baz"] .>= baz_min) .& (zlcc_out["baz"] .<= baz_max)
        mask = condMAAC .& condBAZ .& condBAZ_LIM
    else
        mask = condMAAC .& condBAZ
    end

    nidx = findall(mask)
    prct_nidx = 100*length(nidx)/N

    if prct_nidx > min_nidx

        # get rms average
        rms = mean(zlcc_out["rms"][nidx])

        # get baz and slow averages
        slow_avg = mean(zlcc_out["slow"][nidx])
        slow_std = std(zlcc_out["slow"][nidx])

        baz_avg  = mean(zlcc_out["baz"][nidx])
        baz_std  = std(zlcc_out["baz"][nidx])

        # get stacked slowmap
        slowmap = dropdims(mean(zlcc_out["slowmap"][nidx,:,:], dims=1), dims=1)

        # get apparent slowness and uncertainty
        maac, midx = findmax(ccmap)
        ii, jj = midx.I
        best_sx = sx[ii]
        best_sy = sy[jj]

        c = contours(sx, sy, slowmap, [maac*ccerr])
        cl = lines(levels(c)[1])
        ncontour = length(cl)
        is_good = false

        if ncontour == 0
            is_good = false
            ratio = NaN

        elseif ncontour == 1
            is_good = true
            idx = 1
            ratio = -1
        
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
            else
                # toma por bueno el contorno dominante
                is_good = true
                idx = sort_idx[1]
            end
        end
        
        if is_good
            Xs, Ys = coordinates(cl[idx])
            ans = r2p.(.-Xs, .-Ys)
            slo = first.(ans)
            ang = last.(ans)
            ang_max = maximum(ang)
            ang_min = minimum(ang)

            unc_lin = ang_max - ang_min
            unc_com = 360.0 - unc_lin

            if unc_lin <= unc_com
                baz_max = ang_max
                baz_min = ang_min
            else
                baz_max = ang_min
                baz_min = ang_max
            end

            bazbnd = [baz_min, baz_max]

            # slo = hypot.(Xs, Ys)
            s_max = maximum(slo)
            s_min = minimum(slo)
            slobnd = [s_min, s_max]

            slow, bazm = r2p(-best_sx, -best_sy)
        
        else
            slow = bazm = NaN
            bazbnd = slobnd = [NaN,NaN]
        end

    else
        ratio = rms = slow = bazm = slow_avg = baz_avg = slow_std = baz_std = NaN
        bazbnd = slobnd = [NaN,NaN] 
        slowmap = [NaN]
    end
        
    return prct_nidx, rms, slowmap, slow, bazm, slobnd, bazbnd, slow_avg, baz_avg, slow_std, baz_std, ratio
end


function _pccorr(data::Array{T}, nkk, px, py, base::BaseZLCC, energy_buf) where T<:Real

    # calculamos los delays
    idx_sta = [round(Int, nkk + px*base.dx[i] + py*base.dy[i]) for i in 1:base.nsta]

    # calculamos energias (diagonal)
    @views @inbounds for ii in 1:base.nsta
        idx = idx_sta[ii]
        dii = data[idx : base.lwin + idx, ii]
        energy_buf[ii] = 1.0 / sqrt(dot(dii, dii))
    end
    
    # calculamos correlaciones cruzadas
    cc_sum = zero(T)
    @views @inbounds for (ii, jj) in base.citer
        dii = data[idx_sta[ii] : base.lwin + idx_sta[ii], ii]
        djj = data[idx_sta[jj] : base.lwin + idx_sta[jj], jj]
        cc_sum += dot(dii, djj) * energy_buf[ii] * energy_buf[jj]
    end

    return (2 * cc_sum + base.nsta) / (base.nsta*base.nsta)
end


function _ccmap!(cc_map, data, n0, base::BaseZLCC, energy_buf)
    @inbounds for (jj, py) in enumerate(base.sy)
        for (ii, px) in enumerate(base.sx)
            cc_map[ii,jj] = _pccorr(data, n0, px, py, base, energy_buf)
        end
    end
    
    return cc_map
end


function _ccmap!(cc_map, data, n0, sx, sy, base::BaseZLCC, energy_buf)
    @inbounds for (jj, py) in enumerate(sy)
        for (ii, px) in enumerate(sx)
            cc_map[ii, jj] = _pccorr(data, n0, px, py, base, energy_buf)
        end
    end
    return cc_map
end


function _power_beam(data::Array{T}, nkk, px, py, base::BaseZLCC) where T<:Real

    beam = zeros(T, base.lwin + 1)

    @inbounds for ii in 1:base.nsta
        delay = px * base.dx[ii] + py * base.dy[ii]
        idx = round(Int, nkk + delay)
        beam .+= @view data[idx : idx + base.lwin, ii]
    end

    return sqrt(mean(abs2, beam)) / base.nsta
end