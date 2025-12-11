#!/usr/local/bin julia
# coding=utf-8

# Utility functions for cc8mre.jl

# GNU GPL v2 licenced to I. Melchor and J. Almendros 08/2022

# using PointwiseKDEs

"""
   _empty_dict(*args)

Genera un dict vacio para llenar durante el procesado.
"""
function _empty_dict(base::Base)
    dict = Dict()
    
    # for attr in ("slow", "baz", "maac", "rms", "error")
    for attr in ("maac", "rms", "slow", "baz")
        dict[attr] = Array{Float64}(undef, base.nwin)
    end
    
    dict["slowbnd"] = Array{Float64}(undef, base.nwin, 2)
    dict["bazbnd"] = Array{Float64}(undef, base.nwin, 2)
    dict["slowmap"] = Array{Float64}(undef, base.nwin, base.nite, base.nite)

    return dict
end

    #
    # This function cretes the slownes grid
    #
function _xygrid(slow0::Vector{T}, sloint::T, slomax::T) where T<:Real

    # define the size of the grid
    nite    = 1 + 2*round(Int64, slomax/sloint)
    
    # init the grid in memeory
    xy_grid = Array{T}(undef, nite, nite, 2)
    
    # fill the grid
    for ii in 1:nite, jj in 1:nite
        px = slow0[1] - slomax + sloint*(ii-1)
        # pxi = pinc * px/pinc
        py = slow0[2] - slomax + sloint*(ii-1)
        # pyj = pinc * py/pinc
        xy_grid[ii,jj,:] = [px,py]
    end
    
    xy_grid[:,:,2] = adjoint(xy_grid[:,:,2])
    
    return xy_grid
end

    #
    # This function cretes the deltatime grid
    #
function _dtimemap(dtime_func::Function, pxy_map::Array{T}, nsta::J) where {T<:Real, J<:Integer}
    
    nite = size(pxy_map, 1)

    time_map = Array{T}(undef, nite, nite, nsta)
    
    for ii in 1:nite, jj in 1:nite 
        time_map[ii,jj,:] = dtime_func(pxy_map[ii,jj,:])
    end

    return time_map
end


"""
   _dtimefunc(*args)

Genera la función que devuelve los delta times para un vector de lentidud aparente
"""
function _dtimefunc(stax::Array{T}, stay::Array{T}, fsem::J) where {T<:Real, J<:Integer}
    xref = mean(stax)
    yref = mean(stay)
    dtime(pxy) = [pxy[1]*(stx-xref) + pxy[2]*(sty-yref) for (stx, sty) in zip(stax,stay)] .* fsem
    return dtime
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

"""
  get_dtimes(x, y)
    
    Devuelve delta times correspondientes a un slowness y un azimuth
    
"""
function get_dtimes(slow::T, bazm::T, slow0::Vector{T}, slomax::T, sloint::T, fsem::J, xStaUTM::Array{T}, yStaUTM::Array{T}) where {T<:Real, J<:Integer}

  nsta  = length(xStaUTM)
  pxy, pij   = p2r(slow, bazm, slow0, slomax, sloint)
  dtime      = _dtimefunc(xStaUTM, yStaUTM, fsem)
  slow_grid  = _xygrid(slow0, sloint, slomax)
  time_grid  = _dtimemap(dtime, slow_grid, nsta)

  return pxy, time_grid[pij[1], pij[2], :]
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

function r2p(pxy::Vector{T}) where T<:Real
  x = pxy[1]
  y = pxy[2]

  slowness = hypot(x, y)
  
  if y < 0
    azimuth = 180+atand(x/y)
  end

  if y > 0
    azimuth = atand(x/y)
    
    if x < 0
      azimuth += 360
    end
  end
  
  if y == 0
    if x > 0
      azimuth = 90.
    end
    
    if x < 0
      azimuth = 270.
    end

    if x == 0
      azimuth = 666.
    end
  end
  
  if azimuth == 0
      azimuth = 360
  end
  
  return (slowness, azimuth)
end


function _ijbound(arr::AbstractArray{T}, cclim::T) where T<:Real
  
  pos = findmax(arr)[2]

  if pos == 1 || pos == size(arr, 1)
    return false, nothing
  end
  
  i = findmin(abs.( arr[1:pos-1] .- cclim ))[2]
  j = findmin(abs.( arr[pos+1:end] .- cclim ))[2]

  return true, (i,pos+j)
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
  fb2(*args)
    
    Filter signal
"""
function _fb2(x::Array{T}, fc::T, fs::J, lowpass::Bool; amort=0.47) where {T<:Real, J<:Real}

  a = tan(pi*fc/fs)
  b = 2*a*a - 2
  c = 1 - 2*amort*a + a*a
  d = 1 + 2*amort*a + a*a

  if lowpass
    a0 = a*a/d
    a1 = 2*a0
  else
    a0 = 1/d
    a1 = -2*a0
  end
  
  a2 = a0
  b1 = -b/d
  b2 = -c/d   
  
  ndata = size(x, 1)
  y = Array{T}(undef, ndata)
  y[1] = x[1]
  y[2] = x[2]

  for j in 3:ndata
    y[j] = a0*x[j] + a1*x[j-1] + a2*x[j-2] + b1*y[j-1] + b2*y[j-2]
  end

  return y
end


function _filter!(data::Array{T}, fs::J, fq_band::Vector{T}) where {T<:Real, J<:Real}
  
  fl, fh = fq_band
  nsta = size(data,1)
  
  for i in 1:nsta
    temp = _fb2(data[i,:], fh, fs, true)
    data[i,:] = _fb2(temp, fl, fs, false)
    temp = reverse(data[i,:])
    data[i,:] = _fb2(temp, fh, fs, true)
    temp = _fb2(data[i,:], fl, fs, false)
    data[i,:] = reverse(temp)
  end

end


function _filter(data::Array{T}, fs::J, fq_band::Vector{T}) where {T<:Real, J<:Real}
    
    U = deepcopy(data)
    _filter!(U, fs, fq_band)

    return U
end


"""
  spb(args)
    
    Compute the mpaac
    
"""
# function pmmac(cmap::Array{T,3}) where T<:Real
  
#   pdfx = LinRange(0, 1, 100)
#   data = reshape(cmap, 1, :)
#   kde  = PointwiseKDE(data)
#   pdfy = rand(kde, 100)
#   mpaac = pdfx[findmax(pdfy)[2][2]]
  
#   return mpaac
# end

# function mpm(slowmap::Array{T,3})  where T<:Real
  
#   nite = size(slowmap, 2)
#   spbmap = Array{T}(undef, nite, nite)

#   for ii in 1:nite
#     for jj in 1:nite
#       data = reshape(slowmap[:,ii,jj], (1, :))
#       data = convert(Array{Float64}, data)
#       data_min = findmin(data)[1]
#       data_max = findmax(data)[1]
#       x_space = LinRange(data_min, data_max, 100)
#       kde = PointwiseKDE(data)
#       y_space = rand(kde, 100)
#       cc_ij = x_space[findmax(y_space)[2][2]]
#       spbmap[ii,jj] = cc_ij
#     end
#   end
    
#   return spbmap
# end


# """
#   spb(args)
    
#     Compute the slowmness and back_azimuth time map
    
# """

# function slobaztmap(slowmap::Array{T,3}, pinc::J, pmax::J, cc_th::J)  where {T<:Real, J<:Real}

#   nwin = size(slowmap, 1)
#   nite = size(slowmap, 2)

#   pxymap = _pxymap([0.,0.], nite, pinc, pmax)
#   sbtm = Array{Vector{Tuple{T,T}}}(undef, nwin)

#   for t in 1:nwin
#     data = slowmap[t,:,:]

#     sbtm_t = Vector{Tuple{T,T}}()
#     for ii in 1:nite
#       for jj in 1:nite
#         if data[ii,jj] > cc_th
#           push!(sbtm_t, r2p(-1 .* pxymap[ii,jj,:]))
#         end
#       end
#     end

#     sbtm[t] = sbtm_t
#   end

#   return sbtm
# end
