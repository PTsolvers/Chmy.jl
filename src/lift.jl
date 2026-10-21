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

flip(::Lower) = Upper()
flip(::Upper) = Lower()

flipside(::Far) = Far()
flipside(n::Near) = Near(flip(n.side), n.ξ)

function lower_rule(lf::Lifted{O, K}, args, comps, offs::NTuple{N, Rational{Int}}, block::Block{N}) where {O, K, N}
    maximum(abs, lf.axes) <= N || throw(ArgumentError("the lifted axes exceed the number of dimensions"))
    # tensor components are not mapped to the lifted axes yet, so only scalar calls of scalars can be lifted
    isempty(comps) || expansion_error(lf)
    checkscalar(lf, args)
    # the offsets and the block restricted to the lifted axes, negating the offsets and flipping the wall sides
    # for negative axes
    loffs = map(a -> flipsign(offs[abs(a)], a), lf.axes)
    lblock = map(a -> a > 0 ? block[a] : flipside(block[-a]), lf.axes)
    term = lower_rule(lf.op, args, comps, loffs, lblock)::DTerm
    lf.axes == ntuple(identity, Val(N)) && return term
    return insert_stencil(term, Val(N), lf.axes)
end

# replace kD samples `f[o₁, …, oₖ]` with nD samples where the offset along `|axes[j]|` is `sign(axes[j]) * oⱼ` and zero otherwise
@inline function insert_stencil(term, dims, axes)
    return @match term begin
        DExpr(Inds(offsets), (arg,)) && if length(offsets) == length(axes) end => makeinds_unchecked(arg, insert_offsets(dims, axes, offsets))
        DExpr(head, args) => something(insert_args(head, args, dims, axes), term)
        _ => term
    end
end

@inline function insert_offsets(dims, axes, offsets)
    result = zero_offsets(dims)
    for j in eachindex(axes)
        axis = @inbounds axes[j]
        result = @inbounds Base.setindex(result, flipsign(offsets[j], axis), abs(axis))
    end
    return result
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
