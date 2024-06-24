# Architectures

The abstract type `Architecture` can be concretized to be either an architecture of a concrete type `SingleDeviceArchitecture` or a `DistributedArchitecture` for MPI usage.

## Single Device Architecture

An object being the type of `SingleDeviceArchitecture` meaning that it contains the handle to a single CPU or GPU device with a respetive backend, where a device can be specified by an integer ID.

### Backend Selection & Architecture Initialization

We currently support CPU architectures, as well as CUDA and ROC backends for Nvidia and AMD GPUs through a thin wrapper around the [`KernelAbstractions.jl`](https://github.com/JuliaGPU/KernelAbstractions.jl) for users to select desirable backends.

```julia
# Default with CPU
arch = Arch(CPU())
```

```julia
# using CUDA
arch = Arch(CUDABackend())
```

```julia
# using AMDGPU
arch = Arch(ROCBackend())
```

At the beginning of program, one may specify the backend and initialize the architecture they desire to use. The initialized `arch` variable will be required explicitly at creation of some entiries such as grids and kernel launchers.

### Specifying Stream Priority

For advanced users, we provide a convenient wrapper `activate!(arch::SingleDeviceArchitecture; priority=:normal)` for specifying the stream priority owned by the task one is executing. The stream priority will be set to `:normal` by default, where `:low` and `:high` are also possible options given that the target backend has priority control over streams implemented.

This internally uses the backend-agnostic `priority!(::Backend, prio::Symbol)` function exposed by `KernelAbstractions.jl`. The exact implementation depends on the target backend used. As an example, see [`AMDGPU.priority!`](https://amdgpu.juliagpu.org/stable/streams/#AMDGPU.priority!).


## Distributed Architecture

Our distributed architecture builds upon the abstraction of having GPU clusters that build on the same GPU architecture. Note that in general, GPU clusters may be equipped with hardware from different vendors, incorporating different types of GPUs to exploit their unique capabilities for specific tasks.

With this abstraction of having homogeneous clusters, an object of type `DistributedArchitecture{ChildArch,Topo}` can have an uniform `child_arch` property representing all child architectures that comprise the distributed architecture. We also provide conveninent getters `get_backend` and `get_device` for obtaining the underlying backend and device of child architectures.

Additionally, a distributed architecture also has a `topo` property that specifies that underlying MPI topology used. We currently support the creation of MPI communicator with Cartesian topology information attached.