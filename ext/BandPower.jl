
module BandPower

    using Multitaper
    using Base.Threads
    using SAP

    function __init__()
        if Threads.nthreads() == 1
            @warn "Julia está corriendo con un solo nucleo. Verifica JULIA_NUM_THREADS."
        end
    end

    function SAP.rolling_bandpower(data::AbstractMatrix{T}, fs::Integer, fq_band::AbstractVector, lwin::Integer, nwin::Integer, NW=3.5) where {T<:Real}

        npts, nsta = size(data)
        dt   = 1/fs
        K    = convert(Int32, 2*NW - 1)
        nadv = floor(Int, (npts - lwin) / (nwin - 1))
        BP   = zeros(Float64, nwin) # band power

        # Pre-calculo
        s_temp = multispec(view(data, 1:lwin, 1), ctr=true, dt=dt, NW=NW, K=K, pad=1.0)
        freq = s_temp.f
        nfreq = length(freq)
        df = freq[2] - freq[1]

        idx1 = findfirst(f -> f >= fq_band[1], freq)
        idx2 = findlast(f -> f <= fq_band[2], freq)

        @threads for n in 1:nwin
            n0  = 1 + nadv * (n-1)
            nf  = n0 + lwin - 1

            psd_avg_window = zeros(Float64, nfreq)
            for s in 1:nsta
                s_spec = multispec(view(data, n0:nf, s), ctr=true, dt=dt, NW=NW, K=K, pad=1.0)
                psd_avg_window .+= s_spec.S
            end
            psd_avg_window ./= nsta

            band_power = sum(view(psd_avg_window, idx1:idx2)) * df
            BP[n] = sqrt(band_power)
        end

        return BP
    end


end # module

