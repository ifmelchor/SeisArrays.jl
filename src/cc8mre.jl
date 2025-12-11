#!/usr/local/bin julia
# coding=utf-8

# GNU GPL v2 licenced to I. Melchor and J. Almendros 08/2022

# Main ZLCC codes
using Contour

function zlcc(data::Array{T}, xStaUTM::Array{T}, yStaUTM::Array{T}, slomax::T, sloint::T, fqband::Vector{T}, fsem::J, lwin::J, nwin::J, nadv::T, toff::J, slow0::Vector{T}=[0., 0.], ccerr::T=0.95, slow2::Bool=True, maac_thr::T=0.5, slomax2::T=0.3, sloint2::T=0.02) where {T<:Real, J<:Integer}
    
    # define base params
    nsta   = length(xStaUTM) # nro of stations
    cciter = _cciter(nsta)   # stations iterator
    toff   = toff*fsem       # off seconds in sp
    nite   = 1 + 2*round(Int64, slomax/sloint)   # slowness grid

    if slow2
        nite2  = 1 + 2*round(Int64, slomax2/sloint2) # slowness grid2
    else
        nite2 = 0
    end

    # base object for crosscorrelation
    base   = Base(nite, nite2, nwin, nsta, lwin, cciter, slow2)

    # init empty dictionary
    dict = _empty_dict(base)
    
    # create slowness main grid
    slow_grid  = _xygrid(slow0, sloint, slomax)
    sx = slow_grid[1, :, 2]
    sy = slow_grid[:, 1, 1]

    # create deltatimes grid
    dtime     = _dtimefunc(xStaUTM, yStaUTM, fsem) # define delta time function
    time_grid = _dtimemap(dtime, slow_grid, nsta)

    # filter data
    _filter!(data, fsem, fqband)

    # define la funcion:
    multi_r2p(Y, X) = r2p([-1.0 * Y, -1.0 * X])
    
    # iterate over time
    @inbounds for nk in 1:nwin
        n0  = 1 + toff + lwin*nadv*(nk-1)

        # get ccmap
        ccmap = _ccmap(data, n0, time_grid, nite, base)

        # find max value MAAC and position
        ccmax      = findmax(ccmap)
        maac       = ccmax[1]
        (ii, jj)   = ccmax[2].I
        best_slow  = slow_grid[ii, jj, :]

        # save data pf the iteration
        dict["maac"][nk]   = maac
        dict["slowmap"][nk,:,:] = ccmap

        # compute contour
        level = [maac * ccerr]
        c = contours(sx, sy, ccmap, level)
        cl = lines(levels(c)[1])
        nro_contornos = length(cl)

        if nro_contornos > 1
            # si hay mas de un contorno, descarta el analisis
            slow = bazm = rms = NaN
            bazbnd = slobnd = [NaN,NaN]
        else
            # calcula la incertidumbre
            Xs, Ys = coordinates(cl[1])
            ans = map(multi_r2p, Xs, Ys)
            slo = [r[1] for r in ans]
            ang = [r[2] for r in ans]
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

            if slow2 && maac > maac_thr
                # compute slownes map with higher precision
                slow_grid2 = _xygrid(best_slow, sloint2, slomax2)
                time_grid2 = _dtimemap(dtime, slow_grid2, nsta)
                ccmap2 = _ccmap(data, n0, time_grid2, nite2, base)
                ccmax2 = findmax(ccmap2)
                (ii, jj)    = ccmax2[2].I
                best_slow2  = slow_grid2[ii, jj, :]
                slow, bazm  = r2p(-1 .* best_slow2)
                rms  = _rms(data, n0, time_grid2[ii, jj, :], base)
            else
                # get slow, baz and rms
                slow, bazm = r2p(-1 .* best_slow)
                rms = _rms(data, n0, time_grid[ii, jj, :], base)
            end
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


function _pccorr(data::Array{T}, nkk::T, pxytime::Vector{T}, base::Base) where T<:Real
    cc = zeros(T, base.nsta, base.nsta)
    
    for ii in 1:base.nsta
        mii = round(Int32, nkk + pxytime[ii])
        dii = @view data[ii, mii:base.lwin+mii]
        for jj in ii:base.nsta
            mjj = round(Int32, nkk + pxytime[jj])
            djj = @view data[jj, mjj:base.lwin+mjj]
            cc[ii,jj] += dot(dii,djj)
        end
    end

    # computes crosscorr coefficient
    cc_sum = 2*sum([cc[ii,jj]/sqrt(cc[ii,ii]*cc[jj,jj]) for (ii, jj) in base.citer])
    
    return (cc_sum+base.nsta) / (base.nsta*base.nsta)
end


function _ccmap(data::Array{T}, n0::T, time_map::Array{T}, nite::J, base::Base) where {T<:Real, J<:Integer}

    cc_map = zeros(T, nite, nite)
    
    @inbounds for ii in 1:nite, jj in 1:nite
        cc_map[ii,jj] = _pccorr(data, n0, time_map[ii,jj,:], base)
    end
    
    return cc_map
end


function _rms(data::Array{T}, nkk::T, pxytime::Vector{T}, base::Base) where T<:Real

    erg = 0.
    for ii in 1:base.nsta
        mii = round(Int32, nkk + pxytime[ii])
        dii = @view data[ii, 1+mii:base.lwin+mii]
        erg += sqrt(mean(dii.^2))
    end
    
    return erg /= base.nsta
end