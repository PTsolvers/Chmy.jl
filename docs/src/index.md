```@raw html
---
# https://vitepress.dev/reference/default-theme-home-page
layout: home

hero:
  name: Chmy.jl Docs
  text: Finite differences and staggered grids on CPUs and GPUs
  tagline: A backend-agnostic toolkit for finite difference computations with task-based distributed memory parallelisation.
  actions:
    - theme: brand
      text: Getting Started
      link: /getting_started/introduction
    - theme: alt
      text: API Reference 📚
      link: /lib/modules
    - theme: alt
      text: View on GitHub
      link: https://github.com/PTsolvers/Chmy.jl
  image:
    src: /logo.png
    alt: Chmy.jl

features:
  - icon: 🚀
    title: Backend Agnostic
    details: Effortlessly execute your code on CPUs and GPUs with KernelAbstractions.jl.
    link: /concepts/architectures

  - icon: ⚡
    title: Multi-device
    details: Leverages task-based distributed memory parallelisation capabilities with MPI.jl.
    link: /concepts/distributed

  - icon: 🛠️
    title: Framework
    details: Fields, boundary conditions and interpolation operators on structured grids.
    link: /getting_started/introduction

  - icon: 🧩
    title: Extensibility
    details: Written in pure Julia, adding new functions, simplification rules, and model transformations has no barrier.
    link: /concepts/kernels
---
```

## What is Chmy.jl?

[Chmy.jl](https://github.com/PTsolvers/Chmy.jl) (pronounce *tsh-mee*) is a **backend-agnostic toolkit for finite difference computations** on multi-dimensional computational staggered grids. Chmy.jl features **task-based distributed memory parallelisation capabilities** and provides a comprehensive **framework for handling complex computational tasks on structured grids**, leveraging both single and multi-device architectures. It seamlessly integrates with Julia's powerful parallel and concurrent programming capabilities, making it suitable for a wide range of scientific and engineering applications.

## How to Install Chmy.jl?

To install Chmy.jl, one can simply add it using the Julia package manager by running the following command in the Julia REPL:

```julia-repl
julia> using Pkg

julia> Pkg.add("Chmy")
```

After the package is installed, one can load the package by using:

```julia-repl
julia> using Chmy
```

If you want to use the latest unreleased version of Chmy.jl, you can run the following command:

```julia-repl
julia> using Pkg

julia> Pkg.add(url="https://github.com/PTsolvers/Chmy.jl")
```

## Select an Accelerator Backend

:::code-group

```julia [CPUs]
using Chmy
using KernelAbstractions

backend = CPU()
arch = Arch(backend)
```

```julia [Nvidia GPUs]
using Chmy
using KernelAbstractions
using CUDA
backend = CUDABackend()
arch = Arch(backend)
```

```julia [AMD GPUs]
using Chmy
using KernelAbstractions
using AMDGPU
backend = ROCBackend()
arch = Arch(backend)
```

```julia [Apple GPUs]
using Chmy
using KernelAbstractions
using Metal
backend = MetalBackend()
arch = Arch(backend)
```

:::

## Funding

The development of this package is supported by the [GPU4GEO](https://pasc-ch.org/projects/2021-2024/gpu4geo/index.html) and ∂GPU4GEO PASC projects. More information about the GPU4GEO project can be found on the [GPU4GEO website](https://ptsolvers.github.io/GPU4GEO/).
