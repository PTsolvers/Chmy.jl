# Kernels

The [KernelAbstactions.jl](https://github.com/JuliaGPU/KernelAbstractions.jl) package provides a macro-based dialect that hides the intricacies of vendor-specific GPU programming. It allows one to write hardware-agnostic kernels that can be instantiated and launched for different device backends without modifying the high-level code nor sacrificing performance.

Followingly, we show how to write and launch kernels on various backends. We also explain the concept of a `Launcher` in [Chmy.jl](https://github.com/PTsolvers/Chmy.jl), that complements the default kernel launching, allowing us to perform more fine-grained operations on different `Field` objects.

## Writing Kernels

This section highlights some important features of [KernelAbstactions.jl](https://github.com/JuliaGPU/KernelAbstractions.jl) that are essential for understanding the high-level abstraction of the kernel concept that is used throughout our package. As it barely serves for illustrative purposes, for more specific examples, please refer to their [documentation](https://juliagpu.github.io/KernelAbstractions.jl/stable/).


```julia
using KernelAbstactions

# Define a kernel that performs element-wise operations on A
@kernel function mul2(A)
    I = @index(Global, Cartesian)
    A[I] *= 2
end
```

With the kernel `mul2` as defined above, we can launch the kernel to perform the element-wise operations on host.

```julia
# Define array and work group size
workgroup_size = 64
A              = ones(1024, 1024)
backend        = get_backend(A) # CPU

# Launch kernel and explicitly synchronize
mul2(backend, workgroup_size)(A, ndrange=size(A))
synchronize(backend)

# Result assertion
@assert(all(A .== 2.0) == true)
```

To launch the kernel on GPU devices, one could simply define `A` as `CuArray`, `ROCArray` or `oneArray` as detailed in the section ["launching kernel on the backend"](https://juliagpu.github.io/KernelAbstractions.jl/stable/quickstart/#Launching-kernel-on-the-backend). More fine-grained memory access is available using the `@index` macro as described [here](https://juliagpu.github.io/KernelAbstractions.jl/stable/api/#KernelAbstractions.@index).


!!! info "Kernel Synchronization"
    In previous section about [task-based parallelism](workers.md), we have mentioned and showed the advantages of asynchronous execution. When it comes to using both CPU and GPU, it appears natural that by default kernel launches are asynchronous. Therefore, in [KernelAbstactions.jl](https://github.com/JuliaGPU/KernelAbstractions.jl), one also needs to explicitly use `synchronize(backend)` function to wait on a series of kernel launches.


## Kernel Launcher

In [Chmy.jl](https://github.com/PTsolvers/Chmy.jl), the `KernelLaunch` module is designed to provide handy utilities for performing different grid operations on selected data entries of `Field`s that are involved at each kernel launch, in which the grid geometry underneath is also taken into account.

Followingly, we define a kernel launcher associated with an `UniformGrid` object, supporting CUDA backend.

```julia
# Define backend and geometry
arch   = Arch(CUDABackend())
grid   = UniformGrid(arch; origin=(-1, -1), extent=(2, 2), dims=(126, 126))

# Define launcher
launch = Launcher(arch, grid)
```

We also have two kernel functions `compute_q!` and `update_C!` defined, which shall update the fields `q` and `C` using grid operators (see section [Grid Operators](./grid_operators.md)) `∂x`, `∂y`, `divg` that are anchored on some grid `g` accordingly.

```julia
@kernel inbounds = true function compute_q!(q, C, χ, g::StructuredGrid, O)
    I = @index(Global, NTuple)
    I = I + O
    q.x[I...] = -χ * ∂x(C, g, I...)
    q.y[I...] = -χ * ∂y(C, g, I...)
end

@kernel inbounds = true function update_C!(C, q, Δt, g::StructuredGrid, O)
    I = @index(Global, NTuple)
    I = I + O
    C[I...] -= Δt * divg(q, g, I...)
end
```

To spawn the kernel, we invoke the launcher using the `launch` function to perform the field update at each physical timestep, and specify desired boundary conditions for involved fields in the kernel.

```julia
# Define physics, numerics, geometry ...
for it in 1:nt
    # without boundary conditions
    launch(arch, grid, compute_q! => (q, C, χ, grid))

    # with Neumann boundary conditions
    launch(arch, grid, update_C! => (C, q, Δt, grid); bc=batch(grid, C => Neumann(); exchange=C))
end

# Explicit synchronization request
KernelAbstractions.synchronize(backend)
```

The launcher internally handles two different scenarios differently, depending on if there are boundary conditions (see section [Boundary Conditions](./bc.md)) to be imposed on some fields involved in the kernel or not, inline functions `launch_without_bc` or  `launch_with_bc` are called respectively.