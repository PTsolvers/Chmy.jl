# Examples Overview

This page provides an overview of [Chmy.jl](https://github.com/PTsolvers/Chmy.jl) examples. These examples demonstrate how [Chmy.jl](https://github.com/PTsolvers/Chmy.jl) can be used to solve various numerical problems using architecture-agnostic kernels both on a single-device and in a distributed way.

## Table of Contents


| Example    | Description | Keywords |
|:------------|:------------|:---------|
| [Basic Diffusion](diffusion_2d.md) | Solving the 2D diffusion equation on an uniform grid. It covers setting up the problem, managing degrees of freedom, the assembly process, boundary conditions, solving the linear system, and visualizing the results. | scalar-valued solution, Dirichlet boundary conditions |
| [Diffusion with MPI](diffusion_2d_mpi.md) | Solving the 2D diffusion equation on an uniform grid distributedly using MPI. | parallel computing, MPI, scalar-valued solution |
| [Stokes Flow 2D](stokes_2d_inc_ve_T.md) | Solving the 2D Stokes flow equations with temperature-dependent viscosity. It introduces handling coupled vector- and scalar-valued solutions and applying Dirichlet and Neumann boundary conditions. | vector-valued solution, coupled problems, boundary conditions |
| [Stokes Flow 3D](stokes_3d_inc_ve_T.md) |Solving the 3D Stokes flow equations. This tutorial focuses on setting up a 3D problem, managing complex boundary conditions, and visualizing 3D results. | 3D simulations, Stokes flow, complex boundary conditions |
| [Performance Optimization](diffusion_2d_perf.md) | Revisiting the 2D diffusion problem with focus on performance optimization techniques. It includes tips on efficient code practices and tools for profiling and improving code performance. | performance optimization, profiling, scalar-valued solution |
| [Batch Processing](batcher.md) | Introduction of batch processing techniques for running multiple simulations efficiently. It demonstrates how to set up batch jobs, manage outputs, and utilize computational resources effectively. | batch processing, job management, automation |
