struct Lifted{O <: Operator, K} <: Operator
    op::O
    axes::NTuple{K, Int}

    """
        Lifted(op, axes)
        Lifted(op, axis)

    Wraps kD operator `op` so that, when lowered with an nD list of coordinates, the k-th coordinate
    of `op` is taken along `axes[k]`. The axes may be in any order, e.g. `Lifted(op, (3, 1))` applies
    the first dimension of `op` along the third axis and the second along the first.
    A single `axis` is equivalent to `(axis,)`.
    """
    function Lifted(op::O, axes::NTuple{K, Integer}) where {O <: Operator, K}
        K > 0 || throw(ArgumentError("at least one lifted axis is required"))
        all(>(0), axes) || throw(ArgumentError("the lifted axes must be positive"))
        allunique(axes) || throw(ArgumentError("the lifted axes must be unique"))
        return new{O, K}(op, map(Int, axes))
    end
end

Lifted(op::Operator, axes::Vararg{Integer, N}) where {N} = Lifted(op, axes)

tensorrank(lf::Lifted, args) = tensorrank(lf.op, args)
checkranks(lf::Lifted, args::NTuple{N, DTerm}) where {N} = checkranks(lf.op, args)

function stencil_rule(lf::Lifted, args::Tuple{Vararg{DTerm}}, coords::NTuple{N, DTerm}) where {N}
    maximum(lf.axes) <= N || throw(ArgumentError("the lifted axes exceed the number of coordinates"))
    term = stencil_rule(lf.op, args, map(a -> coords[a], lf.axes))::DTerm
    lf.axes == ntuple(identity, Val(N)) && return term
    return insert_stencil(term, coords, lf.axes)
end

# replace kD samples `f[c₁, …, cₖ]` with nD samples `f[coords...]` where the coordinate along `axes[j]` is `cⱼ`
@inline function insert_stencil(term, coords, axes)
    return @match term begin
        DExpr(Inds(), args) && if length(args) == length(axes) + 1 end => makeinds_unchecked(args[1], insert_coords(coords, axes, args))
        DExpr(head, args) => something(insert_args(head, args, coords, axes), term)
        _ => return term
    end
end

# set the coordinates along `axes` to the sample indices `args[2:end]`
@inline function insert_coords(coords, axes, args)
    for j in eachindex(axes)
        coords = @inbounds Base.setindex(coords, args[j + 1], axes[j])
    end
    return coords
end

@inline function insert_args(head, args, coords, axes)
    for k in eachindex(args)
        arg = insert_stencil(args[k], coords, axes)
        arg ==ₛ args[k] && continue
        return DExpr(head, insert_args(args, arg, coords, axes, k))
    end
    return nothing
end
@inline function insert_args(args::NTuple{N, DTerm}, arg, coords, axes, k) where {N}
    return ntuple(Val(N)) do j
        j < k ? args[j] : j == k ? arg : insert_stencil(args[j], coords, axes)
    end
end
