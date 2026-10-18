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
    negated offsets, so that a rule written once for the lower wall can be reused at the upper wall.

    Only the samples created by the stencil rule of `op` are lifted.
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

function stencil_rule(lf::Lifted{O, K}, args::Tuple{Vararg{DTerm}}, ctx::LowerContext{N}) where {O, K, N}
    maximum(abs, lf.axes) <= N || throw(ArgumentError("the lifted axes exceed the number of dimensions"))
    # construct the context restricted to the lifted axes
    lctx = LowerContext(
        ntuple(i -> ctx.locs[abs(lf.axes[i])], Val(K)),
        # negate global offsets for negative axes
        ntuple(Val(K)) do i
            a = lf.axes[i]
            o = ctx.offs[abs(a)]
            return flipsign(o, a)
        end,
        # flip wall sides for negative axes
        ntuple(Val(K)) do i
            a = lf.axes[i]
            b = ctx.blck[abs(a)]
            return a > 0 ? b : flipside(b)
        end
    )
    term = stencil_rule(lf.op, args, lctx)::DTerm
    lf.axes == ntuple(identity, Val(N)) && return term
    return insert_stencil(term, args, Val(N), lf.axes)
end

# replace kD samples `f[o₁, …, oₖ]` with nD samples where the offset along `|axes[j]|` is `sign(axes[j]) * oⱼ` and zero otherwise
@inline function insert_stencil(term, kept, dims, axes)
    any(x -> x ==ₛ term, kept) && return term
    return @match term begin
        DExpr(Inds(offsets), (arg,)) && if length(offsets) == length(axes) end => makeinds_unchecked(arg, insert_offsets(dims, axes, offsets))
        DExpr(head, args) => something(insert_args(head, args, kept, dims, axes), term)
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

@inline function insert_args(head, args, kept, dims, axes)
    for k in eachindex(args)
        arg = insert_stencil(args[k], kept, dims, axes)
        arg ==ₛ args[k] && continue
        return DExpr(head, insert_args(args, arg, kept, dims, axes, k))
    end
    return nothing
end
@inline function insert_args(args::NTuple{N, DTerm}, arg, kept, dims, axes, k) where {N}
    return ntuple(Val(N)) do j
        j < k ? args[j] : j == k ? arg : insert_stencil(args[j], kept, dims, axes)
    end
end
