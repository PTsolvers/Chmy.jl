# Architectures

## Backend Selection & Architecture Initialization

Chmy.jl supports CPUs, as well as CUDA and ROC backends for Nvidia and AMD GPUs through a thin wrapper around the [`KernelAbstractions.jl`](https://github.com/JuliaGPU/KernelAbstractions.jl) for users to select desirable backends.

```julia
# Default with CPU
arch = Arch(CPU())
```

```julia
using CUDA

arch = Arch(CUDABackend())
```

```julia
using AMDGPU

arch = Arch(ROCBackend())
```

At the beginning of program, one may specify the backend and initialize the architecture they desire to use. The initialized `arch` variable will be required explicitly at creation of some objects such as grids and kernel launchers.

## Specifying the device ID and stream priority

On systems with multiple GPUs, passing the keyword argument `device_id` to the `Arch` constructor will select and set the selected device as a current device.

For advanced users, we provide a function `activate!(arch; priority)` for specifying the stream priority owned by the task one is executing. The stream priority will be set to `:normal` by default, where `:low` and `:high` are also possible options given that the target backend has priority control over streams implemented.

## Distributed Architecture

Our distributed architecture builds upon the abstraction of having GPU clusters that build on the same GPU architecture. Note that in general, GPU clusters may be equipped with hardware from different vendors, incorporating different types of GPUs to exploit their unique capabilities for specific tasks.

!!! warning "GPU-Aware MPI Required for Distributed Module on GPU backend"
    The `Distributed` module currently only supports [GPU-aware MPI](https://juliaparallel.org/MPI.jl/stable/usage/#CUDA-aware-MPI-support) when a GPU backend is selected for multi-GPU computations. For the `Distributed` module to function properly, any GPU-aware MPI library installation shall be used. Otherwise, a segmentation fault will occur.


To make the `Architecture` object aware of MPI topology, user can pass an MPI communicator object and dimensions of the Cartesian topology to the `Arch` constructor:

```julia
using MPI

arch = Arch(CPU(), MPI.COMM_WORLD, (0, 0, 0))
```

Passing zeros as the last argument will automatically spread the dimensions to be as close as possible to each other, see [MPI.jl documentation](https://juliaparallel.org/MPI.jl/stable/reference/topology/#MPI.Dims_create) for details.
