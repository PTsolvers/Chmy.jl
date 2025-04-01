module KernelLaunch

export Launcher
export worksize, outer_width, inner_worksize, inner_offset, outer_worksize, outer_offset

using Chmy
using Chmy.Architectures
using Chmy.Grids
using Chmy.BoundaryConditions
using Chmy.Workers

using KernelAbstractions

"""
    struct Launcher{Worksize,OuterWidth,Workers}

A struct representing a launcher for asynchronous kernel execution.
"""
struct Launcher{Worksize,OuterWidth,Workers}
    workers::Workers
end

"""
    Launcher(arch, grid; outer_width=nothing)

Constructs a `Launcher` object configured based on the input parameters.

## Arguments:
- `arch`: The associated architecture.
- `grid`: The grid defining the computational domain.
- `outer_width`: Optional parameter specifying outer width.

!!! warning

    worksize for the last dimension N takes into account only last outer width
    W[N], N-1 uses W[N] and W[N-1], N-2 uses W[N], W[N-1], and W[N-2].
"""
function Launcher(arch, grid; outer_width=nothing)
    worksize = size(grid, Center()) .+ 2

    if !isnothing(outer_width)
        setup() = activate!(arch; priority=:high)
        workers = ntuple(Val(ndims(grid))) do _
            Base.@_inline_meta
            ntuple(_ -> Worker(; setup), Val(2))
        end
    else
        workers = nothing
    end

    return Launcher{worksize,outer_width,typeof(workers)}(workers)
end

Base.@assume_effects :foldable Base.ndims(::Launcher{WorkSize}) where {WorkSize} = length(WorkSize)
Base.@assume_effects :foldable worksize(::Launcher{WorkSize}) where {WorkSize} = WorkSize
Base.@assume_effects :foldable outer_width(::Launcher{WorkSize,OuterWidth}) where {WorkSize,OuterWidth} = OuterWidth

Base.@assume_effects :foldable inner_worksize(launcher::Launcher) = worksize(launcher) .- 2 .* outer_width(launcher)
Base.@assume_effects :foldable inner_offset(launcher::Launcher) = outer_width(launcher)

Base.@assume_effects :foldable function outer_worksize(launcher::Launcher, ::Dim{D}) where {D}
    ntuple(Val(ndims(launcher))) do I
        Base.@_inline_meta
        if I < D
            worksize(launcher)[I]
        elseif I == D
            outer_width(launcher)[I]
        else
            worksize(launcher)[I] - 2outer_width(launcher)[I]
        end
    end
end

Base.@assume_effects :foldable function outer_offset(launcher::Launcher, ::Dim{D}, ::Side{S}) where {D,S}
    ntuple(Val(ndims(launcher))) do I
        Base.@_inline_meta
        if I < D
            0
        elseif I == D
            S == 1 ? 0 : worksize(launcher)[I] - outer_width(launcher)[I]
        else
            outer_width(launcher)[I]
        end
    end
end

"""
    (launcher::Launcher)(arch::Architecture, grid, kernel_and_args::Pair{F,Args}; bc=nothing) where {F,Args}

Launches a computational kernel using the specified `arch`, `grid`, `kernel_and_args`, and optional boundary conditions (`bc`).

## Arguments:
- `arch::Architecture`: The architecture on which to execute the computation.
- `grid`: The grid defining the computational domain.
- `kernel_and_args::Pair{F,Args}`: A pair consisting of the computational kernel `F` and its arguments `Args`.
- `bc=nothing`: Optional boundary conditions for the computation.

!!! warning
    - `arch` should be compatible with the `Launcher`'s architecture.
    - If `bc` is `nothing`, the kernel is launched without boundary conditions.
    - The function waits for the computation to complete before returning.
"""
function (launcher::Launcher)(arch::Architecture, grid, kernel_and_args::Pair{F,Args}; bc=nothing) where {F,Args}
    kernel, args = kernel_and_args

    backend = Architectures.get_backend(arch)
    offset  = Offset(-1)

    if isnothing(bc)
        launch_without_bc(backend, launcher, offset, kernel, args...)
    else
        launch_with_bc(arch, grid, launcher, offset, kernel, bc, args...)
    end

    KernelAbstractions.synchronize(backend)
    return
end

@inline function launch_without_bc(backend, launcher, offset, kernel, args...)
    groupsize = heuristic_groupsize(backend, Val(ndims(launcher)))
    fun = kernel(backend, groupsize, worksize(launcher))
    fun(args..., offset)
    return
end

import KernelAbstractions.NDIteration.StaticSize

@inline function launch_with_bc(arch, grid, launcher, offset, kernel, bc, args...)
    backend   = Architectures.get_backend(arch)
    groupsize = StaticSize(heuristic_groupsize(backend, Val(ndims(launcher))))

    if isnothing(outer_width(launcher))
        fun = kernel(backend, groupsize, StaticSize(worksize(launcher)))
        fun(args..., offset)
        bc!(arch, grid, bc)
    else
        modify_sync!(args, disable_task_sync!)

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

        modify_sync!(args, enable_task_sync!)
    end
    return
end

end
