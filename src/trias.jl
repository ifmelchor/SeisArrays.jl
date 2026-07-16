#!/usr/local/bin julia
# coding=utf-8

# GNU GPL v2 licenced to I. Melchor and J. Almendros 08/2022
 # TRIAS (TRIad-based Adaptive Slowness)

struct ValidTriad{T<:AbstractFloat}
    tri      :: TriangleDef{T}
    rho      :: T
    dfreq    :: T
    sx       :: T
    sy       :: T
    sigma    :: T
    closure2 :: T
end


struct TriadBuffer{T<:AbstractFloat}
    ws          :: WSGCC{T}
    mdelay      :: Matrix{T}
    msigma      :: Matrix{T}
    mcorr       :: Matrix{T}
    mfreq       :: Matrix{T}
    slowmax     :: T
    slowmax2    :: T
    ValidTriads :: Vector{ValidTriad{T}}
    delta_s2    :: Vector{T}
    ipol_map    :: Matrix{T} # mapa interpolado
    finer_map   :: Matrix{T} # mapa fino
    coarser_map :: Matrix{T} # mapa grueso
    # SoA buffers para _misfitmap!
    _dx1   :: Vector{T}
    _dx2   :: Vector{T}
    _dx3   :: Vector{T}
    _dy1   :: Vector{T}
    _dy2   :: Vector{T}
    _dy3   :: Vector{T}
    _obs1  :: Vector{T}
    _obs2  :: Vector{T}
    _obs3  :: Vector{T}
    _sigma :: Vector{T}
    _rho   :: Vector{T}
    _clos2 :: Vector{T}
    _dfreq :: Vector{T}
end


mutable struct TriasOutput{T<:AbstractFloat}
    time_s    :: Vector{T}
    n_trios   :: Vector{Int}
    rho       :: Vector{T}
    dfreq     :: Vector{T}
    sigma     :: Vector{T}
    sx        :: Vector{T}
    sy        :: Vector{T}
    slow      :: Matrix{T}   # (nwin, 3)
    baz       :: Matrix{T}   # (nwin, 3)
    eta2      :: Vector{T}
    kappa     :: Vector{T}
    ECN       :: Vector{T}   # Error de Clausura Normalizado medio
    CNE       :: Vector{T}   # Criterio de Nyquists Espacial medio
    s_ratio   :: Vector{T}
    slow_width:: Vector{T}
    baz_width :: Vector{T}
    mmap      :: Vector{Union{Matrix{T}, Nothing}}
    trios     :: Vector{Matrix{T}}
end


function trias(data::AbstractArray, x::AbstractVector, y::AbstractVector, fs::Real, args...; kwargs...)
    
    SA = SeisArray2D(x, y, data, fs)
    
    return trias(SA, args...; kwargs...)
end


function trias(S::SeisArray2D, lwin::Int, nadv::T, fmin::T, fmax::T; slowmax::T=2.5, slowint_c::T=0.1, slowint_f::T=0.01, slowfw::T=0.5, min_vtriads::J=1, min_cc::T=0.5, min_psr::T=5.0, max_ECN::T=0.5, B::J=5, upsample::J=20) where {T<:Real, J<:Integer}

    npts, nsta = size(S.data)

    # Define las triadas y la matriz de distancias relativas
    mdist, triads = init_triads(S)

    # Filtra los datos
    filter!(S, fmin, fmax)

    # Define ventanas de analisis
    step = round(Int, lwin * nadv)
    nwin = div(npts - lwin, Int(step)) + 1

    # Define los mapas de lentitud (Grueso, Fino, Completo)
    s_grid_c  = -slowmax:slowint_c:slowmax
    nite_c    = size(s_grid_c, 1)
    rfine     = -slowfw:slowint_f:slowfw
    nite_f    = size(rfine, 1)
    s_grid    = -slowmax:slowint_f:slowmax
    nite      = size(s_grid, 1)

    # Inicia Buffers de procesado
    n_threads = Threads.nthreads()
    buffers = [_init_buffer(nsta, lwin, S.fs, fmin, fmax, B, upsample,slowmax, nite, nite_c, nite_f) for _ in 1:n_threads]

    # Crea el guardado
    nan1 = fill(T(NaN), nwin)       # vector 1D de NaN
    nan2 = fill(T(NaN), nwin, 3)    # matriz (nwin,3) de NaN
    out = TriasOutput{T}(
        copy(nan1),   # time_s
        zeros(Int, nwin),# n_trios
        copy(nan1),   # rho
        copy(nan1),   # dfreq
        copy(nan1),   # sigma
        copy(nan1),   # sx
        copy(nan1),   # sy
        copy(nan2),   # slow
        copy(nan2),   # baz
        copy(nan1),   # eta2
        copy(nan1),   # kappa
        copy(nan1),   # Cavg
        copy(nan1),   # Davg
        copy(nan1),   # s_ratio
        copy(nan1),   # slow_width
        copy(nan1),   # baz_width
        fill!(Vector{Union{Matrix{T}, Nothing}}(undef, nwin), nothing),
        [Matrix{T}(undef, 0, 9) for _ in 1:nwin]
    )

    @views Threads.@threads for nk in 1:nwin
        # inicia el buffer
        tid = Threads.threadid()
        buf = buffers[tid]

        # define la ventana
        n0 = round(Int, 1 + lwin * nadv * (nk - 1))
        window_data = S.data[n0:n0+lwin-1, :]

        # calcula delays
        _delay_matrix_gcc!(buf, window_data, mdist, min_psr, min_cc)

        # get valid triads
        _fill_valid_triads!(buf, triads, min_cc, max_ECN)

        # Filtro de triadas validas minimo
        nvalid = length(buf.ValidTriads)
        nvalid >= min_vtriads || continue

        # Calcula el mapa de misfits GRUESO
        _misfitmap!(buf.coarser_map, buf, s_grid_c, s_grid_c)

        # Lentitud grueso
        idx = argmin(buf.coarser_map)
        sx0 = s_grid_c[idx[1]]
        sy0 = s_grid_c[idx[2]]

        # Calcula el mapa de misfits FINO
        rfx = rfine .+ sx0 
        rfy = rfine .+ sy0
        _misfitmap!(buf.finer_map, buf, rfx, rfy)

        # Interpola
        interpolate_grids!(buf.ipol_map, buf.coarser_map, buf.finer_map, s_grid, s_grid_c, rfx, rfy)

        # Calcula el mínimo
        idxf = firstindex(buf.ipol_map)
        vmin = typemax(T)
        @inbounds for i in eachindex(buf.ipol_map)
            v = buf.ipol_map[i]
            if !isnan(v) && v < vmin
                vmin = v; idxf = i
            end
        end
        
        cidx = CartesianIndices(buf.ipol_map)[idxf]
        sxf  = s_grid[cidx[1]]
        syf  = s_grid[cidx[2]]

        # CALCULA METRICAS DE CALIDAD
        qmetric = _quality_metrics(sxf, syf, max_ECN, buf)

        # Calcula la Pseudo-verosimilitud
        @. buf.ipol_map = exp(-sqrt(nvalid * qmetric.eta2) * buf.ipol_map / qmetric.sigma_med)
        
        # Calcula la Icertidumbre
        # nivel de una desviacion estandar
        level = maximum(buf.ipol_map) / ℯ
        uncert = uncertainty_contour(s_grid, s_grid, buf.ipol_map, level)

        # Guarda datos
        t0 = T(n0 - 1) / T(S.fs)
        _save_window!(out, nk, buf, sxf, syf, qmetric, uncert, t0)
    end

    _mask_output(out)
end


function _flatten_triads(triads::Vector{ValidTriad{T}}) where {T}
    n = length(triads)
    # columnas: i, j, k, rho, dfreq, sx, sy, sigma, closure2
    mat = Matrix{T}(undef, n, 9)
    @inbounds for (row, vt) in enumerate(triads)
        mat[row, 1] = T(vt.tri.i)
        mat[row, 2] = T(vt.tri.j)
        mat[row, 3] = T(vt.tri.k)
        mat[row, 4] = vt.rho
        mat[row, 5] = vt.dfreq
        mat[row, 6] = vt.sx
        mat[row, 7] = vt.sy
        mat[row, 8] = vt.sigma
        mat[row, 9] = vt.closure2
    end
    return mat
end


function _save_window!(out::TriasOutput{T}, nk::Int, buf::TriadBuffer{T}, sxf::T, syf::T, qmetric, uncert, t0::T) where {T}

    out.time_s[nk]     = t0
    out.n_trios[nk]    = length(buf.ValidTriads)
    out.sx[nk]         = sxf
    out.sy[nk]         = syf
    out.rho[nk]        = qmetric.rho_med
    out.dfreq[nk]      = qmetric.fqdom_med
    out.eta2[nk]       = qmetric.eta2
    out.kappa[nk]      = qmetric.kappa
    out.ECN[nk]        = qmetric.ECN_avg
    out.CNE[nk]        = qmetric.CNE_avg
    out.sigma[nk]      = qmetric.sigma_med

    if !isnothing(uncert)
        out.slow[nk, :]   .= uncert.slow
        out.baz[nk, :]    .= uncert.baz
        out.slow_width[nk] = uncert.slow_width
        out.baz_width[nk]  = uncert.baz_width
    end

    out.mmap[nk]       = copy(buf.ipol_map)
    out.trios[nk]      = _flatten_triads(buf.ValidTriads)
end


function _mask_output(out::TriasOutput{T}) where {T}
    mask = findall(!isnan, out.time_s)
    isempty(mask) && return nothing
    return TriasOutput{T}(
        out.time_s[mask],
        out.n_trios[mask],
        out.rho[mask],
        out.dfreq[mask],
        out.sigma[mask],
        out.sx[mask],
        out.sy[mask],
        out.slow[mask, :],
        out.baz[mask, :],
        out.eta2[mask],
        out.kappa[mask],
        out.ECN[mask],
        out.CNE[mask],
        out.s_ratio[mask],
        out.slow_width[mask],
        out.baz_width[mask],
        out.mmap[mask],
        out.trios[mask]
    )
end


function _init_buffer(nsta::J, lwin::J, fs::T, fmin::T, fmax::T, B::J, upsample::J, slowmax::T, nite::J, nite_c::J, nite_f::J) where {T<:AbstractFloat, J<:Integer}

    # Numero de triadas
    ntriads = binomial(nsta, 3)

    # Inicializa WSGCC con taper de 0.2 y metodo ML
    ws = init_wsgcc(lwin, fs, fmin, fmax, B, upsample, T(0.2), :ml)

    # slowness maximo
    slowmax2 = slowmax*slowmax

    # Mapas de lentitud
    ipol_map    = zeros(T, nite, nite)
    finer_map   = zeros(T, nite_f, nite_f)
    coarser_map = zeros(T, nite_c, nite_c)
    mdelay      = zeros(T, nsta, nsta)
    msigma      = zeros(T, nsta, nsta)
    mcorr       = zeros(T, nsta, nsta)
    mfreq       = zeros(T, nsta, nsta)

    triad  = Vector{ValidTriad{T}}()
    sizehint!(triad, ntriads) # Reserva espacio de memoria para las triadas validas
    deltas = Vector{T}()
    sizehint!(deltas, ntriads) # Reserva espacio de memoria para las triadas validas
    _soa(n) = Vector{T}(undef, n)
    
    return TriadBuffer(ws, mdelay, msigma, mcorr, mfreq, slowmax, slowmax2, triad, deltas, ipol_map, finer_map, coarser_map, _soa(ntriads), _soa(ntriads), _soa(ntriads), _soa(ntriads), _soa(ntriads), _soa(ntriads), _soa(ntriads), _soa(ntriads), _soa(ntriads), _soa(ntriads), _soa(ntriads), _soa(ntriads), _soa(ntriads))
end


function _delay_matrix_gcc!(tb::TriadBuffer{T}, data::AbstractMatrix{T}, mdist::AbstractMatrix{T}, psr_th::T, min_cc::T) where {T<:AbstractFloat}

    N     = size(data, 2)
    mdelay  = tb.mdelay
    msigma  = tb.msigma
    mcorr   = tb.mcorr
    mfreq   = tb.mfreq

    CC_UNSET   = T(-2.0)
    CC_INVALID = T(-1.0)
    fill!(mcorr, CC_UNSET)
    
    @views @inbounds for i in 1:N, j in i+1:N
        if mcorr[i,j] == CC_UNSET
            res = _gcc_delay_core!(tb.ws, data[:,j], data[:,i])
            
            valid_psr   = res.psr >= psr_th
            valid_delay = abs(res.delay) < (mdist[i,j] * tb.slowmax)
            valid_corr  = res.coherence >= min_cc
            
            if valid_psr && valid_delay && valid_corr
                mdelay[j,i] =  res.delay
                mdelay[i,j] = -res.delay

                msigma[i,j] =  res.sigma
                msigma[j,i] =  res.sigma

                mcorr[i,j]  =  res.coherence
                mcorr[j,i]  =  res.coherence

                mfreq[i,j]  = res.dfreq
                mfreq[j,i]  = res.dfreq
            else
                mcorr[i,j] = CC_INVALID
                mcorr[j,i] = CC_INVALID
            end
            
        end
    end
end


function _fill_valid_triads!(tb::TriadBuffer{T}, triads::Vector{TriangleDef{T}}, min_cc::T, max_ECN::T) where {T<:AbstractFloat}

    # Limpia antes de acumular
    empty!(tb.ValidTriads)

    @views @inbounds for tri in triads
        # extrae los indices de la triada
        i, j, k       = tri.i, tri.j, tri.k
        dx1, dx2, dx3 = tri.dx_ij, tri.dx_jk, tri.dx_ki
        dy1, dy2, dy3 = tri.dy_ij, tri.dy_jk, tri.dy_ki

        # Filtro de correlacion
        tb.mcorr[i,j] >= min_cc || continue
        tb.mcorr[j,k] >= min_cc || continue
        tb.mcorr[k,i] >= min_cc || continue

        # Calculo de clausura
        dt1 = tb.mdelay[i,j]
        dt2 = tb.mdelay[j,k]
        dt3 = tb.mdelay[k,i]
        closure = dt1 + dt2 + dt3

        # Calculo de slowness por LS
        # Mínimos cuadrados: M = [dx1 dy1; dx2 dy2; dx3 dy3]
        mtm_11 = dx1*dx1 + dx2*dx2 + dx3*dx3
        mtm_12 = dx1*dy1 + dx2*dy2 + dx3*dy3
        mtm_22 = dy1*dy1 + dy2*dy2 + dy3*dy3
        mtd_1  = dx1*dt1 + dx2*dt2 + dx3*dt3
        mtd_2  = dy1*dt1 + dy2*dt2 + dy3*dt3
        det = mtm_11 * mtm_22 - mtm_12^2
        inv_det = one(T) / det
        sx = inv_det * ( mtm_22 * mtd_1 - mtm_12 * mtd_2)
        sy = inv_det * (-mtm_12 * mtd_1 + mtm_11 * mtd_2)
        
        # Filtro de lentitud
        s_triad = sqrt(sx^2 + sy^2)
        s_triad <= tb.slowmax || continue

        # Calculo de t_max
        t1 = abs(dx1 * sx + dy1 * sy)
        t2 = abs(dx2 * sx + dy2 * sy)
        t3 = abs(dx3 * sx + dy3 * sy)
        t_max = (t1 + t2 + t3) / T(2.0)

        # Calculo de la freq dominante
        fq1   = tb.mfreq[i,j]
        fq2   = tb.mfreq[j,k]
        fq3   = tb.mfreq[k,i]
        fqmed = sort3_median(fq1, fq2, fq3)

        # Criterio de Nyquist espacial
        t_max*fqmed > T(0.5) || continue

        # Calculo de sigma
        s1 = tb.msigma[i,j]
        s2 = tb.msigma[j,k]
        s3 = tb.msigma[k,i]
        sigma = sqrt(s1^2 + s2^2 + s3^2)

        # Filtro de Clausura Normalizada
        ecn = closure*closure/(t_max+sigma)^2
        ecn < max_ECN || continue

        # Calcula la coherencia de la triada
        cc1 = tb.mcorr[i,j]
        cc2 = tb.mcorr[j,k]
        cc3 = tb.mcorr[k,i]
        rho = sort3_median(cc1, cc2, cc3)
        # rho = (cc1 + cc2 + cc3) / 3.0

        # Triada Válida!
        push!(tb.ValidTriads, ValidTriad{T}(tri, rho, fqmed, sx, sy, sigma, closure*closure))
    end

    # Llena el SoA
    @inbounds for t in eachindex(tb.ValidTriads)
        vt = tb.ValidTriads[t]
        i, j, k = vt.tri.i, vt.tri.j, vt.tri.k
        tb._dx1[t]   = vt.tri.dx_ij
        tb._dx2[t]   = vt.tri.dx_jk
        tb._dx3[t]   = vt.tri.dx_ki
        tb._dy1[t]   = vt.tri.dy_ij
        tb._dy2[t]   = vt.tri.dy_jk
        tb._dy3[t]   = vt.tri.dy_ki
        tb._obs1[t]  = tb.mdelay[i,j]
        tb._obs2[t]  = tb.mdelay[j,k]
        tb._obs3[t]  = tb.mdelay[k,i]
        tb._sigma[t] = vt.sigma
        tb._rho[t]   = vt.rho
        tb._clos2[t] = vt.closure2
        tb._dfreq[t] = vt.dfreq
    end

end


function _misfitmap!(like_map::AbstractMatrix{T}, tb::TriadBuffer{T}, s_grid_x::AbstractVector{T}, s_grid_y::AbstractVector{T}) where {T<:AbstractFloat}

    nx = length(s_grid_x)
    ny = length(s_grid_y)

    inv_2 = T(0.5)
    nvtrios = length(tb.ValidTriads)

     @views begin
        dx1   = tb._dx1[1:nvtrios]
        dx2   = tb._dx2[1:nvtrios]
        dx3   = tb._dx3[1:nvtrios]
        dy1   = tb._dy1[1:nvtrios]
        dy2   = tb._dy2[1:nvtrios]
        dy3   = tb._dy3[1:nvtrios]
        obs1  = tb._obs1[1:nvtrios]
        obs2  = tb._obs2[1:nvtrios]
        obs3  = tb._obs3[1:nvtrios]
        sig   = tb._sigma[1:nvtrios]
        rho   = tb._rho[1:nvtrios]
        clos2 = tb._clos2[1:nvtrios]
    end

    fill!(like_map, typemax(T))

    @inbounds for jj in 1:ny
        sy  = s_grid_y[jj]
        sy2 = sy * sy

        for ii in 1:nx
            sx = s_grid_x[ii]

            (sx*sx + sy2) <= tb.slowmax2 || continue

            acum_d = zero(T)
            acum_n = zero(T)

            for t in 1:nvtrios
                dt1  = sx * dx1[t] + sy * dy1[t]
                dt2  = sx * dx2[t] + sy * dy2[t]
                dt3  = sx * dx3[t] + sy * dy3[t]

                # tiempo maximo de la triada evaluada en sx, sy
                tmax = sig[t] + inv_2 * (abs(dt1) + abs(dt2) + abs(dt3))
                
                # residuos norma L1
                r = abs(dt1 - obs1[t]) + abs(dt2 - obs2[t]) + abs(dt3 - obs3[t])
                
                # peso
                w    = rho[t] * exp(-clos2[t] / (tmax * tmax))
                
                acum_d += w
                acum_n += w * r
            end

            like_map[ii, jj] = acum_n / acum_d
        end
    end
end


function _quality_metrics(sx::T, sy::T, max_ecn::T, tb::TriadBuffer{T}) where {T<:AbstractFloat}

    inv_2 = T(0.5)
    nvalid    = length(tb.ValidTriads)
    rho_med   = median!(@view tb._rho[1:nvalid])
    sigma_med = median!(@view tb._sigma[1:nvalid])
    fqdom_med = median!(@view tb._dfreq[1:nvalid])
    
    acum_ECN = T(0.0) # acumulador de error de clausura normalizado
    acum_CNE = T(0.0) # acumulador de criterio de Nyquits espacial
    resize!(tb.delta_s2, nvalid)

    @inbounds for t in 1:nvalid
        vt = tb.ValidTriads[t]
        # Calcula delay teoricos
        dt_t1 = abs(sx * vt.tri.dx_ij + sy * vt.tri.dy_ij)
        dt_t2 = abs(sx * vt.tri.dx_jk + sy * vt.tri.dy_jk)
        dt_t3 = abs(sx * vt.tri.dx_ki + sy * vt.tri.dy_ki)
        t_max = inv_2 * (dt_t1 + dt_t2 + dt_t3)

        acum_ECN += vt.closure2/(t_max + vt.sigma)^2
        acum_CNE += t_max * vt.dfreq

        dsx = vt.sx - sx
        dsy = vt.sy - sy
        tb.delta_s2[t] = dsx*dsx + dsy*dsy
    end

    ECN_avg      = acum_ECN/nvalid
    CNE_avg      = acum_CNE/nvalid
    delta_s2_med = median!(tb.delta_s2)

    slow2        = sx*sx + sy*sy
    kappa = slow2 > T(1e-12) ? exp(-delta_s2_med / slow2) : zero(T)
    
    sigmoid_C = sigmoid(-ECN_avg; x0=-max_ecn)
    sigmoid_D = sigmoid( CNE_avg; x0=T(0.5))
    eta2 = kappa * sigmoid_C * sigmoid_D

    return (; rho_med, sigma_med, fqdom_med, ECN_avg, CNE_avg, kappa, eta2)
end



