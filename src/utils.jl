#!/usr/local/bin julia
# coding=utf-8

# Utility functions for cc8mre.jl

# GNU GPL v2 licenced to I. Melchor and J. Almendros 08/2022

"""
   _empty_dict(*args)

Genera un dict vacio para llenar durante el procesado.
"""
function _empty_dict(base::BaseZLCC, save_maps::Bool)
    dict = Dict()
    
    for attr in ("maac", "rms", "slow", "baz", "slow_ratio")
        dict[attr] = Array{Float64}(undef, base.nwin)
    end
    
    if save_maps
      dict["slowmap"] = Array{Float64}(undef, base.nwin, base.nite, base.nite)
    end

    dict["slowbnd"] = Array{Float64}(undef, base.nwin, 2)
    dict["bazbnd"] = Array{Float64}(undef, base.nwin, 2)

    return dict
end


function _cciter(nsta::J) where J<:Integer
  
  cciter = Vector{Tuple{J,J}}()
  for ii in 1:nsta-1
      for jj in ii+1:nsta
          push!(cciter, (ii, jj))
      end
  end

  return cciter
end


function get_delays(slow::T, bazm::T, slow0::Vector{T}, slomax::T, sloint::T, xStaUTM::Array{T}, yStaUTM::Array{T}) where T<:Real

    rad = deg2rad(bazm)
    px = -slow * sin(rad)
    py = -slow * cos(rad)

    r  = -slomax:sloint:slomax
    sx = collect(r .+ slow0[1])
    sy = collect(r .+ slow0[2])
    ii = argmin(abs.(sx .- px_theo))
    jj = argmin(abs.(sy .- py_theo))

    px = sx[ii]
    py = sy[jj]

    nsta = length(xStaUTM)
    xref = mean(xStaUTM)
    yref = mean(yStaUTM)
    dx = (xStaUTM .- xref)
    dy = (yStaUTM .- yref)

    # Calcular Delta Times en segundos
    dt = [(px * dx[i] + py * dy[i]) for i in 1:nsta]

    return dt, [px, py]
end


"""
  p2r(x, y)
    
    Get slowness vector
    
"""
function p2r(slow::T, bazm::T, slow0::Vector{T}, slomax::T, sloint::T) where T<:Real
  if 0 <= bazm <= 90
     px = -slow*sin(deg2rad(bazm))
     py = -slow*cos(deg2rad(bazm))
  end

  if 90 < bazm <= 180
     px = -1*slow*cos(deg2rad(bazm-90))
     py = slow*sin(deg2rad(bazm-90))
  end

  if 180 < bazm <= 270
     px = slow*sin(deg2rad(bazm-180))
     py = slow*cos(deg2rad(bazm-180))
  end

  if 270 < bazm <= 360
     px = slow*cos(deg2rad(bazm-270))
     py = -1*slow*sin(deg2rad(bazm-270))
  end

  # seach position on grid
  pxi = 1 + ((px - slow0[1] + slomax) / sloint)
  pyj = 1 + ((py - slow0[2] + slomax) / sloint)

  return [px, py], round.(Int64, [pxi, pyj])
end


"""
  r2p(x, y)
    
    Get slowness and back-azimuth angles
    
"""

@inline function r2p(x::T, y::T) where T<:Real
    if x == 0 && y == 0
        return (0.0, 666.0)
    end

    slowness = hypot(x, y)
    azimuth = mod(atand(x, y), 360)

    if azimuth == 0
        azimuth = 360.0
    end

    return (slowness, azimuth)
end


"""
  count_size(x, y)
    
    Get contour size from Lazada Equation
    
"""

function contour_size(x, y)
    area = 0.0
    n = length(x)
    for i in 1:n-1
        area += x[i] * y[i+1] - x[i+1] * y[i]
    end
    return abs(area + x[n]*y[1] - x[1]*y[n]) / 2.0
end


"""
  bm2(*args)
    
    Get slowness and back-azimuth bounds
    Javi's way
    
"""
function bm2(msum::AbstractArray{T}, pmax::T, pinc::T, ccmax::T, ccerr::T) where T<:Real
  nite = size(msum, 1)
  bnd = Bounds(666., -1., 666., -1.)
  q = Array{Bool}(undef, nite, nite)

  ccmin = ccmax - ccerr

  for i in 1:nite, j in 1:nite
    px = pinc * (i-1) - pmax  
    py = pinc * (j-1) - pmax 
      
    if (px == 0) && (py == 0)
      continue
    end

    if msum[i,j] >= ccmin
      q[i,j] = 1
      
      for x in (-px+pinc, -px-pinc)
        for y in (-py+pinc, -py-pinc)
          s, a = r2p([x, y])
          
          if s > bnd.slomax
            bnd.slomax = s 
          end

          if s < bnd.slomin
            bnd.slomin = s 
          end

          if a > bnd.azimax
            bnd.azimax = a 
          end

          if a < bnd.azimin
            bnd.azimin = a 
          end

        end
      end
    else
      q[i,j] = 0
    end

  end

  if (bnd.azimax > 355) && (bnd.azimin < 5)
    bnd.azimin = 666.
    bnd.azimax = 1.
    
    for i in 1:nite, j in 1:nite
      px = pinc * (i-1) - pmax 
      py = pinc * (j-1) - pmax

      if (px == 0) && (py == 0)
        continue
      end
        
      if q[i,j]
        for x in (-px+pinc, -px-pinc)
          for y in (-py+pinc, -py-pinc)

            s, a = r2p([x, y])

            if x > 0 && a > bnd.azimax
              bnd.azimax = a
            end

            if x < 0 && a < bnd.azimin
              bnd.azimin = a
            end

          end
        end
      end

    end

  end

  return bnd
end


"""
    Filter signal
"""
function _fb2_inplace!(input, output, coef)
    ndata = length(input)
    output[1], output[2] = input[1], input[2]
    @inbounds for j in 3:ndata
        output[j] = coef[1]*input[j] + coef[2]*input[j-1] + coef[3]*input[j-2] + coef[4]*output[j-1] + coef[5]*output[j-2]
    end
end

function _get_fb2_coefs(fc, fs, lowpass, T)
    a = tan(pi * fc / fs)
    amort = 0.47
    d = 1 + 2 * amort * a + a * a
    
    a0 = lowpass ? (a * a / d) : (1 / d)
    a1 = lowpass ? (2 * a0) : (-2 * a0)
    a2 = a0
    b1 = -(2 * a * a - 2) / d
    b2 = -(1 - 2 * amort * a + a * a) / d

    return (T(a0), T(a1), T(a2), T(b1), T(b2))
end


function _filter!(data::Array{T}, fs::J, fq_band::Vector{T}) where {T<:Real, J<:Real}
  
  fl, fh = fq_band
  ntime, nsta = size(data)

  coef_h = _get_fb2_coefs(fh, fs, true, T)
  coef_l = _get_fb2_coefs(fl, fs, false, T)

  temp_buf = zeros(T, ntime)
  
  @views for i in 1:nsta
    # Forward
    _fb2_inplace!(data[:, i], temp_buf, coef_h)
    _fb2_inplace!(temp_buf, data[:, i], coef_l)
    # reverse
    reverse!(data[:, i])
    # backward
    _fb2_inplace!(data[:, i], temp_buf, coef_h)
    _fb2_inplace!(temp_buf, data[:, i], coef_l)
    # reverse
    reverse!(data[:, i])
  end
end


function _filter(data::Array{T}, fs::J, fq_band::Vector{T}) where {T<:Real, J<:Real}
    
    U = deepcopy(data)
    _filter!(U, fs, fq_band)

    return U
end
