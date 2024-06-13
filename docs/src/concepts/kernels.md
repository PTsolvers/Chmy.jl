# Kernels


## `KernelAbstractions.jl`

The [KernelAbstactions.jl](https://github.com/JuliaGPU/KernelAbstractions.jl) package provides a macro-based dialect that hides the intricacies of vendor-specific GPU programming. It allows one to write hardware-agnostic kernels that can be instantiated and launched for different device backends without modifying the high-level code nor sacrificing performance.

```bash
julia> Chmy.KernelLaunch.
Launcher
outer_width
inner_offset
inner_worksize
outer_offset  
outer_worksize     
worksize
```



The launcher internally handles two different scenarios differently, depending with the boundary conditions are considered or not, inline functions `launch_with_bc` or `launch_without_bc` are called.

Launching a kernel function without having to consider about the boundary conditions is relatively straightforward.

```julia
@inline function launch_without_bc(backend, launcher, offset, kernel, args...)
    groupsize = heuristic_groupsize(backend, Val(ndims(launcher)))
    fun = kernel(backend, groupsize, worksize(launcher))
    fun(args..., offset)
    return
end
```



```julia
@inline function launch_with_bc(arch, grid, launcher, offset, kernel, bc, args...)
    backend   = Architectures.get_backend(arch)
    groupsize = StaticSize(heuristic_groupsize(backend, Val(ndims(launcher))))

    if isnothing(outer_width(launcher))
        fun = kernel(backend, groupsize, StaticSize(worksize(launcher)))
        fun(args..., offset)
        bc!(arch, grid, bc)
    else
        inner_fun = kernel(backend, groupsize, StaticSize(inner_worksize(launcher)))
        inner_fun(args..., offset + Offset(inner_offset(launcher)...))

        N = ndims(grid)
        ntuple(Val(N)) do J
            Base.@_inline_meta
            D = N - J + 1
            outer_fun = kernel(backend, groupsize, StaticSize(outer_worksize(launcher, Dim(D))))
            ntuple(Val(2)) do S
                put!(launcher.workers[D][S]) do
                    outer_fun(args..., offset + Offset(outer_offset(launcher, Dim(D), Side(S))...))
                    bc!(Side(S), Dim(D), arch, grid, bc[D][S])
                    KernelAbstractions.synchronize(backend)
                end
            end
            wait(launcher.workers[D][1])
            wait(launcher.workers[D][2])
        end
    end
    return
end
```





## Writing & Launching Kernels

In the following section, we illustrate how to write and launch a kernel  in [Chmy.jl](https://github.com/PTsolvers/Chmy.jl).

```julia
# Define backend and geometry
arch = Arch(CPU())
grid = UniformGrid(arch; origin=(-1, -1), extent=(2, 2), dims=(126, 126))
# Define launcher
launch = Launcher(arch, grid; outer_width=(16, 8))
```



```julia
# nt = ...
for it in 1:nt
    # without boundary conditions
    launch(arch, grid, compute_q! => (q, C, χ, grid))

    # with Neumann boundary conditions
    launch(arch, grid, update_C! => (C, q, Δt, grid); bc=batch(grid, C => Neumann(); exchange=C))
end

# Explicit synchronization request
KernelAbstractions.synchronize(backend)
```