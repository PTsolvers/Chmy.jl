"""
    AbstractBatch

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
@generated function bc!(arch::Architecture, grid::SG{N}, batch::BatchSet{N}) where {N}
    quote
        @inline
        Base.Cartesian.@nexprs $N D -> begin
            bc!(Side(1), Dim($N - D + 1), arch, grid, batch[$N-D+1][1])
            bc!(Side(2), Dim($N - D + 1), arch, grid, batch[$N-D+1][2])
        end
        return
    end
end

"""
    EmptyBatch <: AbstractBatch

EmptyBatch represents no boundary conditions.
"""
struct EmptyBatch <: AbstractBatch end

bc!(::Side, ::Dim, ::Architecture, ::SG, ::EmptyBatch) = nothing

"""
    FieldBatch <: AbstractBatch

FieldBatch is a batch of boundary conditions, where each field has one boundary condition.
"""
struct FieldBatch{K,F,B} <: AbstractBatch
    fields::F
    conditions::B
    function FieldBatch(fields::NTuple{K,Field},
                        conditions::NTuple{K,FieldBoundaryCondition}) where {K}
        return new{K,typeof(fields),typeof(conditions)}(fields, conditions)
    end
end

Base.show(io::IO, ::FieldBatch{K}) where {K} = print(io, "FieldBatch ($K fields)")

"""
    ExchangeBatch <: AbstractBatch

ExchangeBatch represents a batch used for MPI communication.
"""
struct ExchangeBatch{K,F} <: AbstractBatch
    fields::F
    function ExchangeBatch(fields::NTuple{K,Field}) where {K}
        return new{K,typeof(fields)}(fields)
    end
end

Base.show(io::IO, ::ExchangeBatch{K}) where {K} = print(io, "ExchangeBatch ($K fields)")

# per-field interface

function batch(grid, fields_bcs::Vararg{Pair{<:Field,<:PerFieldBC}}; exchange=nothing)
    fields, bcs = regularise(grid, fields_bcs)
    exchange = regularise_exchange(grid, exchange)
    return batch_set(grid, fields, bcs, exchange)
end

# exchange-only interface

function batch(grid; exchange=nothing)
    exchange = regularise_exchange(grid, exchange)
    return batch_set(grid, (), (), exchange)
end

# custom batcher interface
function batch(grid, fields, bcs; exchange=nothing)
    exchange = regularise_exchange(grid, exchange)
    return batch_set(grid, fields, bcs, exchange)
end

# one-sided batch

batch(side::Side, dim::Dim, grid::SG, fields::Tuple, bcs::Tuple; exchange=nothing) = batch_impl(connectivity(grid, dim, side), fields, bcs, exchange)

batch_impl(::Connected, fields, bcs, exch::Nothing)                = EmptyBatch()
batch_impl(::Connected, fields, bcs, exch::Tuple{})                = EmptyBatch()
batch_impl(::Connected, fields, bcs, exch::Tuple{Vararg{Nothing}}) = EmptyBatch()
batch_impl(::Connected, fields, bcs, exch::Tuple)                  = ExchangeBatch(exch)

batch_impl(::Bounded, fields::Tuple{}, bcs::Tuple{}, exch)              = EmptyBatch()
batch_impl(::Bounded, fields::Tuple, bcs::Tuple{Vararg{Nothing}}, exch) = EmptyBatch()
batch_impl(::Bounded, fields::Tuple, bcs::Tuple, exch)                  = FieldBatch(prune(fields, bcs)...)

# make a set from regularised list of fields, boundary conditions, and exchange overrides
@generated function batch_set(grid::SG{N}, fields, bcs, exch) where {N}
    quote
        @inline
        Base.Cartesian.@ntuple $N D -> begin
            batch(Side(1), Dim(D), grid, fields, bcs[D][1]; exchange=exch[D]),
            batch(Side(2), Dim(D), grid, fields, bcs[D][2]; exchange=exch[D])
        end
    end
end

@inline function regularise(grid, fields_bcs::Tuple{Vararg{Pair{<:Field,<:PerFieldBC}}})
    fields, bcs = zip(fields_bcs...) # split into fields and bcs
    bcs_reg = map(bc -> regularise_impl(grid, bc), bcs) |> reorder
    return fields, bcs_reg
end

default_bcs(grid::StructuredGrid) = NamedTuple{axes_names(grid)}(ntuple(_ -> (nothing, nothing), Val(ndims(grid))))

expand(bc::FBCOrNothing)                     = (bc, bc)
expand(bc::Tuple{FBCOrNothing,FBCOrNothing}) = bc

regularise_impl(::StructuredGrid{N}, bc::FieldBoundaryCondition) where {N} = ntuple(_ -> (bc, bc), N)
regularise_impl(grid::StructuredGrid, bcs::TupleBC) = merge(default_bcs(grid), map(expand, bcs)) |> Tuple

# Exchange
default_exchange(grid) = NamedTuple{axes_names(grid)}(ntuple(_ -> nothing, Val(ndims(grid))))

regularise_exchange(grid, ::Nothing) = ntuple(_ -> nothing, Val(ndims(grid)))
regularise_exchange(grid, field::Field) = ntuple(_ -> (field,), Val(ndims(grid)))
regularise_exchange(grid, fields::Tuple{Vararg{Field}}) = ntuple(_ -> fields, Val(ndims(grid)))

function regularise_exchange(grid, fields::NamedTuple{named_dims,<:Tuple{Vararg{Field}}}) where {named_dims}
    return merge(default_exchange(grid), fields) |> Tuple
end

@inline function reorder(conditions::NTuple{K,NTuple{N,SidesBCs}}) where {K,N}
    ntuple(Val(N)) do D
        Base.@_inline_meta
        ntuple(J -> conditions[J][D][1], Val(K)),
        ntuple(J -> conditions[J][D][2], Val(K))
    end
end

@inline function prune(fields, bcs)
    f_bc   = (zip(fields, bcs)...,)
    pruned = filter(x -> !isnothing(last(x)), f_bc)
    return (zip(pruned...)...,)
end

bc!(arch::Architecture, grid::SG, f_bc::Vararg{FieldAndBC}; kwargs...) = bc!(arch, grid, batch(grid, f_bc...; kwargs...))

# batched kernels
@kernel function bc_kernel!(side::Side, dim::Dim,
                            grid::SG{N},
                            fields::NTuple{K,Field},
                            conditions::NTuple{K,FieldBoundaryCondition}) where {N,K}
    J = @index(Global, NTuple)
    I = J .- 1
    ntuple(Val(K)) do ifield
        Base.@_inline_meta
        @inbounds begin
            f = fields[ifield]
            bc = conditions[ifield]
            loc = location(f, dim)
            bc!(side, dim, grid, f, loc, bc, I...)
        end
    end
end

function bc!(side::Side, dim::Dim, arch::Architecture, grid::SG, batch::FieldBatch)
    worksize = remove_dim(dim, size(grid, Center()) .+ 2)
    bc_kernel!(Architectures.get_backend(arch), 256, worksize)(side, dim, grid, batch.fields, batch.conditions)
    return
end

# merging batches and batch sets
Base.merge(b1::FieldBatch, b2::FieldBatch) = FieldBatch((b1.fields..., b2.fields...), (b1.conditions..., b2.conditions...))
Base.merge(b1::FieldBatch, b2::EmptyBatch) = b1
Base.merge(b1::EmptyBatch, b2::FieldBatch) = b2

Base.merge(b1::ExchangeBatch, b2::ExchangeBatch) = ExchangeBatch((b1.fields..., b2.fields...))
Base.merge(b1::ExchangeBatch, b2::EmptyBatch)    = b1
Base.merge(b1::EmptyBatch, b2::ExchangeBatch)    = b2

@inline function Base.merge(b1::NTuple{N,Tuple{AbstractBatch,AbstractBatch}},
                            b2::NTuple{N,Tuple{AbstractBatch,AbstractBatch}}) where {N}
    ntuple(Val(N)) do D
        Base.@_inline_meta
        merge(b1[D][1], b2[D][1]),
        merge(b1[D][2], b2[D][2])
    end
end
