#!/usr/local/bin julia
# coding=utf-8

struct BaseZLCC{T<:Real, J<:Integer}
  nite    :: J              #  --> slowness grid nite x nite
  nwin    :: J              #  --> number of time windows
  nsta    :: J              #  --> number of stations
  lwin    :: J              #  --> time window length
  citer   :: Vector{Tuple{J,J}}
  sx      :: Vector{T}
  sy      :: Vector{T}
  dx      :: Vector{T}
  dy      :: Vector{T}
end


mutable struct ATF{T<:Real}
    freqs :: AbstractArray{T}
    x     :: AbstractArray{T}
    y     :: AbstractArray{T}
    sx    :: AbstractArray{T} # Horizontal slownesses in s/km in x-direction of beamforming grid (first dimension)
    sy    :: AbstractArray{T} # Horizontal slownesses in s/km in y-direction of beamforming grid (second dimension)
    power :: AbstractArray{T,2} # Power of array response at each (sx, sy) point
end

