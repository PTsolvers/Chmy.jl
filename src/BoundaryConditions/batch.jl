"""
    abstract type AbstractBatch

Abstract type representing a batch of boundary conditions.
"""
abstract type AbstractBatch end

const BatchSet{N} = NTuple{N,Tuple{AbstractBatch,AbstractBatch}}

"""
    bc!(arch::Architecture, grid::StructuredGrid, batch::BatchSet)

Apply boundary conditions using a batch set `batch` containing an `AbstractBatch` per dimension per side of `grid`.

# Arguments
- `arch`: The architecture.
- `grid`: The grid.
- `batch:`: The batch set to apply boundary conditions to.
"""
function bc!(arch::Architecture, grid::SG{N}, batch::BatchSet{N}) where {N}
    ntuple(Val(N)) do D
        Base.@_inline_meta
        bc!(Val(1), Val(D), arch, grid, batch[D][1])
        bc!(Val(2), Val(D), arch, grid, batch[D][2])
    end
    return
end

"""
    struct EmptyBatch <: AbstractBatch

EmptyBatch represents no boundary conditions.
"""
struct EmptyBatch <: AbstractBatch end

bc!(::Val, ::Val, ::Architecture, ::SG, ::EmptyBatch) = nothing

struct FieldBatch{K,F,B} <: AbstractBatch
    fields::F
    conditions::B
    function FieldBatch(fields::NTuple{K,Field}, conditions::NTuple{K,FBC}) where {K}
        return new{K,typeof(fields),typeof(conditions)}(fields, conditions)
    end
end

regularise(::StructuredGrid{N}, bc::FieldBoundaryCondition) where {N} = ntuple(_ -> (bc, bc), N)

default_bcs(grid::StructuredGrid) = NamedTuple{axes_names(grid)}(ntuple(_ -> (nothing, nothing), Val(ndims(grid))))

expand(bc::FBCOrNothing)                     = (bc, bc)
expand(bc::Tuple{FBCOrNothing,FBCOrNothing}) = bc

@inline regularise(grid::StructuredGrid, bcs::TupleBC) = merge(default_bcs(grid), map(expand, bcs)) |> Tuple

@inline function reorder(conditions::NTuple{K,NTuple{N,SidesBCs}}) where {K,N}
    ntuple(Val(N)) do D
        Base.@_inline_meta
        ntuple(J -> conditions[J][D][1], Val(K)),
        ntuple(J -> conditions[J][D][2], Val(K))
    end
end

@inline prune(fields, ::NTuple{K,Nothing}) where {K} = (), ()

@inline function prune(fields, bcs)
    f_bc   = (zip(fields, bcs)...,)
    pruned = filter(x -> !isnothing(last(x)), f_bc)
    return (zip(pruned...)...,)
end

function batch(::SDA, grid::StructuredGrid{N}, f_bcs::Vararg{FieldAndBC,K}) where {N,K}
    fs, bcs = zip(f_bcs...)
    bcs_reg = map(x -> regularise(grid, x), bcs) |> reorder
    return _batch(fs, bcs_reg)
end

# ntuple version is type unstable for some reason
@generated function _batch(fs::NTuple{K,Field{<:Any,N}}, bcs::Tuple) where {N,K}
    quote
        @inline
        Base.Cartesian.@ntuple $N D -> begin
            Base.Cartesian.@ntuple 2 S -> begin
                bcs[D][S] isa Tuple{Vararg{Nothing}} ? EmptyBatch() : FieldBatch(prune(fs, bcs[D][S])...)
            end
        end
    end
end

bc!(arch::Architecture, grid::SG, f_bc::Vararg{FieldAndBC}; kwargs...) = bc!(arch, grid, batch(arch, grid, f_bc...; kwargs...))

# batched kernels
@kernel function bc_kernel!(side::Val, dim::Val, grid::SG{N}, batch::FieldBatch{K}) where {N,K}
    J = @index(Global, NTuple)
    I = J .- 1
    ntuple(Val(K)) do ifield
        Base.@_inline_meta
        @inbounds begin
            f   = batch.fields[ifield]
            bc  = batch.conditions[ifield]
            Ibc = insert_dim(dim, I, halo_index(side, dim, f, location(f, dim)))
            bc!(side, dim, grid, f, location(f, dim), bc, Ibc...)
        end
    end
end

function bc!(side::Val, dim::Val, arch::Architecture, grid::SG, batch::FieldBatch)
    worksize = remove_dim(dim, size(grid, Center()) .+ 2)
    bc_kernel!(Architectures.get_backend(arch), 256, worksize)(side, dim, grid, batch)
    return
end
