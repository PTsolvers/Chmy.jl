# Examples Overview

This page provides an overview of [Chmy.jl](https://github.com/PTsolvers/Chmy.jl) examples. These selected examples demonstrate how Chmy.jl can be used to solve various numerical problems using architecture-agnostic kernels both on a single-device and in a distributed way.

## Table of Contents

| Example     | Description |
|:------------|:------------|
| [Diffusion 2D](https://github.com/PTsolvers/Chmy.jl/blob/main/examples/diffusion_2d.jl) | Solving the 2D diffusion equation on a uniform grid. |
| [Diffusion 2D with MPI](https://github.com/PTsolvers/Chmy.jl/blob/main/examples/diffusion_2d_mpi.jl) | Solving the 2D diffusion equation on a uniform grid and distributed parallelisation using MPI. |
| [Single-Device Performance Optimisation](https://github.com/PTsolvers/Chmy.jl/blob/main/examples/diffusion_2d_perf.jl) | Revisiting the 2D diffusion problem with focus on performance optimisation techniques on a single-device architecture. |
| [Stokes 2D with MPI](https://github.com/PTsolvers/Chmy.jl/blob/main/examples/stokes_2d_inc_ve_T_mpi.jl) | Solving the 2D Stokes equation with thermal coupling on a uniform grid. |
| [Stokes 3D with MPI](https://github.com/PTsolvers/Chmy.jl/blob/main/examples/stokes_3d_inc_ve_T_mpi.jl) | Solving the 3D Stokes equation with thermal coupling on a uniform grid and distributed parallelisation using MPI. |
| [Diffusion 1D with Metal](https://github.com/PTsolvers/Chmy.jl/blob/main/examples/diffusion_1d_mtl.jl) | Solving the 1D diffusion equation using the Metal backend and single precision (`Float32`) on a uniform grid. |
| [2D Grid Visualization](https://github.com/PTsolvers/Chmy.jl/blob/main/examples/grids_2d.jl) | Visualization of a 2D `StructuredGrid`. |
| [3D Grid Visualization](https://github.com/PTsolvers/Chmy.jl/blob/main/examples/grids_3d.jl) | Visualization of a 3D `StructuredGrid`. |
