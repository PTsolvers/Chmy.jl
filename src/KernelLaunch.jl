module KernelLaunch

export Launcher
export Offset

using Chmy.Architectures
using Chmy.Grids
using Chmy.BoundaryConditions
using Chmy.Workers

using KernelAbstractions

struct Offset{O} end

Offset(o::Vararg{Integer}) = Offset{o}()
Offset() = Offset{0}()

Base.:+(::Offset{O1}, ::Offset{O2}) where {O1,O2} = Offset((O1 .+ O2)...)
Base.:+(::Offset{O}, tp::Tuple{Vararg{Integer}}) where {O} = O .+ tp
Base.:+(tp::Tuple{Vararg{Integer}}, off::Offset) = off + tp

struct Launcher{InnerWorksize,
                InnerOffset,
                OuterWorkSizes,
                OuterOffsets,
                OuterWidth,
                Workers}
    workers::Workers
end

function Launcher(arch, grid; outer_width=nothing)
    worksize = size(grid, Center()) .+ 2

    if isnothing(outer_width)
        inner_worksize  = worksize
        inner_offset    = Offset()
        outer_worksizes = nothing
        outer_offsets   = nothing
        workers         = nothing
    else
        # TODO
    end

    return Launcher{inner_worksize,inner_offset,outer_worksizes,outer_offsets,typeof(workers)}(workers)
end

Base.@assume_effects :foldable Base.ndims(::Launcher{WorkSize}) where {} = length(WorkSize)
Base.@assume_effects :foldable inner_worksize(::Launcher{InnerWorkSize}) where {InnerWorkSize} = InnerWorkSize

Base.@assume_effects :foldable function outer_worksize(::Launcher{IWS,IO,OuterWorkSizes}, ::Dim{D}) where {IWS,IO,OuterWorkSizes,D}
    return InnerWorkSize[D]
end

Base.@assume_effects :foldable function inner_offset(::Launcher{IWS,InnerOffset}) where {IWS,InnerOffset}
    return InnerOffset
end

Base.@assume_effects :foldable function outer_offset(::Launcher{IWS,IO,OWS,OuterOffset}, ::Dim{D}, ::Side{S}) where {IWS,IO,OWS,OuterOffset}
    return OuterOffset[D][S]
end

Base.@assume_effects :foldable outer_width(::Launcher{IWS,IO,OWS,OO,OuterWidth}) where {IWS,IO,OWS,OO,OuterWidth} = OuterWidth

function (l::Launcher)(arch::Architecture, grid::StructuredGrid, kernel::F, args...;
                       bc=nothing, async=false) where {F}
    backend   = Chmy.get_backend(arch)
    groupsize = heuristic_groupsize(backend, Val(ndims(l)))

    offset = Offset(-1, -1)

    if isnothing(bc)
        worksize = size(grid, Center()) .+ 2
        fun = kernel(backend, groupsize, worksize)
        fun(args..., offset)
    else
    end

    async || KernelAbstractions.synchronize(backend)
end

@inline function launch_without_bc(backend, worksize, groupsize, offset, kernel, args...)
    fun = kernel(backend, groupsize, worksize)
    fun(args..., offset)
    return
end

@inline function launch_with_bc(backend, launcher, groupsize, kernel, bc, args...)
    inner_fun = kernel(backend, groupsize, inner_worksize(launcher))
    inner_fun(args..., offset + inner_offset(launcher))

    if isnothing(outer_width(l))
        bc!(arch, grid, bc)
    else
        ntuple(Val(ndims(grid))) do D
            Base.@_inline_meta
            outer_fun = kernel(backend, groupsize, outer_worksize(launcher, Dim(D)))
            ntuple(Val(2)) do S
                put!(launcher.workers[D][S]) do
                    outer_fun(args..., offset + outer_offset(launcher, Dim(D), Side(S)))
                    bc!(Side(S), Dim(D), arch, grid, bc[D][S])
                end
            end
            wait(launcher.workers[D][1])
            wait(launcher.workers[D][2])
        end
    end
end

end
