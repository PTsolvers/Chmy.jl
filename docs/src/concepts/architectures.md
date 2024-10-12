# Architectures

## Backend Selection & Architecture Initialization

Chmy.jl supports CPUs, as well as CUDA, ROC and Metal backends for Nvidia, AMD and Apple M-series GPUs through a thin wrapper around the [`KernelAbstractions.jl`](https://github.com/JuliaGPU/KernelAbstractions.jl) for users to select desirable backends. For distributed usage of Chmy.jl see the concept documentation on [Distributed](./distributed.md).

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

```julia
using Metal

arch = Arch(MetalBackend())
```

At the beginning of program, one may specify the backend and initialize the architecture they desire to use. The initialized `arch` variable will be required explicitly at creation of some objects such as grids and kernel launchers.

## Specifying the device ID and stream priority

On systems with multiple GPUs, passing the keyword argument `device_id` to the `Arch` constructor will select and set the selected device as a current device.

For advanced users, we provide a function `activate!(arch; priority)` for specifying the stream priority owned by the task one is executing. The stream priority will be set to `:normal` by default, where `:low` and `:high` are also possible options given that the target backend has priority control over streams implemented.