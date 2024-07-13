# Chmy.jl

[Chmy.jl](https://github.com/PTsolvers/Chmy.jl) is a backend-agnostic toolkit for finite difference computations on multi-dimensional computational staggered grids. Chmy.jl features task-based distributed memory parallelisation capabilities.

## Installation

To install Chmy.jl, one can simply add it using the Julia package manager:

```julia
using Pkg
Pkg.add("Chmy")
```

After the package is installed, one can load the package by using:

```julia
using Chmy
```


## Feature Summary

Chmy.jl provides a comprehensive framework for handling complex computational tasks on structured grids, leveraging both single and multi-device architectures. It seamlessly integrates with Julia's powerful parallel and concurrent programming capabilities, making it suitable for a wide range of scientific and engineering applications.

A general list of the features is:

- Single device & distributed (MPI) computing support
- Multi-dimensional, parameterizable field definition on structured grids
- Specification of boundary conditions for numerical fields
- Various grid operators (finite difference, grid masking, value interpolation) for fields defined in Cartesian coordinates
- Extensibility. The whole package is written in pure Julia, so adding new functions, simplification rules, and model transformations has no barrier.



## Running Tests


### CPU tests

To run the Chmy test suite on the CPU, simple run `test` from within the package mode or using `Pkg`:
```julia-repl
using Pkg
Pkg.test("Chmy")
```

### GPU tests

To run the Chmy test suite on CUDA or ROC backend (Nvidia or AMD GPUs), respectively, run the tests using `Pkg` adding following `test_args`:

#### For CUDA backend (Nvidia GPUs):

```julia-repl
using Pkg
Pkg.test("Chmy"; test_args=["--backend=CUDA"])
```

#### For ROC backend (AMD GPUs):

```julia-repl
using Pkg
Pkg.test("Chmy"; test_args=["--backend=AMDGPU"])
```



## Funding

The development of this package is supported by the [GPU4GEO PASC project](https://pasc-ch.org/projects/2021-2024/gpu4geo/index.html). More information about the GPU4GEO project can be found on the [GPU4GEO website](https://ptsolvers.github.io/GPU4GEO/).
