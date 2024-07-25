# Chmy.jl

[Chmy.jl](https://github.com/PTsolvers/Chmy.jl) is a backend-agnostic toolkit for finite difference computations on multi-dimensional computational staggered grids. Chmy.jl features task-based distributed memory parallelisation capabilities.

## Installation

To install Chmy.jl, one can simply add it using the Julia package manager:

```julia
julia> using Pkg
julia> Pkg.add("Chmy")
```

After the package is installed, one can load the package by using:

```julia
using Chmy
```

!!! info "Install from a Specific Branch" 
    For developers and advanced users, one might want to use the implementation of Chmy.jl from a specific branch by specifying the url. In the following code snippet, we do this by explicitly specifying to use the current implementation that is available under the `main` branch:

    ```julia
    using Pkg; Pkg.add(url="https://github.com/PTsolvers/Chmy.jl#main")
    ```

## Feature Summary

Chmy.jl provides a comprehensive framework for handling complex computational tasks on structured grids, leveraging both single and multi-device architectures. It seamlessly integrates with Julia's powerful parallel and concurrent programming capabilities, making it suitable for a wide range of scientific and engineering applications.

A general list of the features is:

- Distributed computing support with MPi.jl
- Multi-dimensional, parameterizable discrete and continuous fields on structured grids
- High-level interface for specifying boundary conditions with automatic batching for performance
- Finite difference and interpolation operators on discrete fields
- Extensibility. The whole package is written in pure Julia, so adding new functions, simplification rules, and model transformations has no barrier.

## Funding

The development of this package is supported by the [GPU4GEO PASC project](https://pasc-ch.org/projects/2021-2024/gpu4geo/index.html). More information about the GPU4GEO project can be found on the [GPU4GEO website](https://ptsolvers.github.io/GPU4GEO/).
