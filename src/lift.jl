struct Lifted{O <: Operator, K} <: Operator
    op::O
    axes::NTuple{K, Int}

    """
        Lifted(op, axes)
        Lifted(op, axes...)

    Wraps kD operator `op` so that, when lowered with an nD list of coordinates, the k-th coordinate
    of `op` is taken along `axes[k]`. The axes may be in any order, e.g. `Lifted(op, 3, 1)` applies
    the first dimension of `op` along the third axis and the second along the first. A negative axis
    `-i` mirrors the stencil, i.e. the corresponding dimension of `op` is taken along the axis `i` with
    negated offsets, so that a rule written once for the upper wall can be reused at the lower wall.
    """
    function Lifted(op::O, axes::NTuple{K, Integer}) where {O <: Operator, K}
        K > 0 || throw(ArgumentError("at least one lifted axis is required"))
        all(!iszero, axes) || throw(ArgumentError("the lifted axes must be nonzero"))
        allunique(map(abs, axes)) || throw(ArgumentError("the lifted axes must be unique"))
        return new{O, K}(op, map(Int, axes))
    end
end
Lifted(op::Operator, axes::Vararg{Integer, N}) where {N} = Lifted(op, axes)

tensorrank(lf::Lifted, args) = tensorrank(lf.op, args)
checkranks(lf::Lifted, args::NTuple{N, DTerm}) where {N} = checkranks(lf.op, args)

function lower_rule(lf::Lifted{O, K}, args, comps, face::Face{N}, x::NTuple{N, Position}) where {O, K, N}
    maximum(abs, lf.axes) <= N || throw(ArgumentError("the lifted axes exceed the number of dimensions"))
    # tensor components are not mapped to the lifted axes yet, so only scalar calls of scalars can be lifted
    isempty(comps) || expansion_error(lf)
    checkscalar(lf, args)
    # mirrored along negative axes: the side of the face flips, and the position changes sign, so that the outward
    # distance from the wall is kept
    lface = map(a -> mirror(face[abs(a)], a), lf.axes)
    lx = map(a -> mirror(x[abs(a)], a), lf.axes)
    term = lower_rule(lf.op, args, comps, lface, lx)::DTerm
    lf.axes == ntuple(identity, Val(N)) && return term
    return insert_stencil(term, Val(N), lf.axes)
end

# replace kD samples `f[p₁, …, pₖ]` with nD samples at `pⱼ` along `axes[j]`, mirrored for negative axes, and zero otherwise
@inline function insert_stencil(term, dims, axes)
    return @match term begin
        DExpr(Inds(offsets), (arg,)) && if length(offsets) == length(axes) end => makeinds_unchecked(arg, insert_offsets(dims, axes, offsets))
        DExpr(head, args) => something(insert_args(head, args, dims, axes), term)
        _ => term
    end
end

@inline function insert_offsets(dims, axes, offsets)
    return ntuple(dims) do a
        j = findfirst(b -> abs(b) == a, axes)
        isnothing(j) && return (0 // 1)::Position
        return mirror(offsets[j], axes[j])::Position
    end
end

@inline function insert_args(head, args, dims, axes)
    for k in eachindex(args)
        arg = insert_stencil(args[k], dims, axes)
        arg ==ₛ args[k] && continue
        return DExpr(head, insert_args(args, arg, dims, axes, k))
    end
    return nothing
end
@inline function insert_args(args::NTuple{N, DTerm}, arg, dims, axes, k) where {N}
    return ntuple(Val(N)) do j
        j < k ? args[j] : j == k ? arg : insert_stencil(args[j], dims, axes)
    end
end
