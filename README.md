# SeisArrays.jl

> **Seismic (and infrasound) Array Processing in Julia**

`SeisArrays.jl` is a Julia package designed for the efficient processing of seismic and infrasound array data. 

<!-- It provides specialized data structures and algorithms for ambient noise analysis, beamforming, and wavefield characterization, leveraging Julia's multiple dispatch and type system for maximum performance. -->

## 🚀 Features

- **Specialized Types:** Robust `SeisArray2D` structure to handle geometry (UTM) and waveform data together.
- **In-place Processing:** Memory-efficient signal processing that minimizes allocations.
- **Beamforming Algorithms:** - **ZLCC** (Zero-Lag Cross Correlation).
  - **TCWALS** (Time-Closure Weighted Adaptive Likelihood Slowness).
- **Flexible Architecture:** Built on `AbstractSeisArray` to support future 3D geometries (boreholes/topography) seamlessy.

## 📦 Installation

```julia
using Pkg
Pkg.add(url="[https://github.com/ifmelchor/SeisArrays.jl.git](https://github.com/ifmelchor/SeisArrays.jl.git)")
