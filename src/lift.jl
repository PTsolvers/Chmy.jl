struct Lifted{O <: Operator} <: Operator
    op::O
    axis::Int

    """
    Lifted(op::Operator, axis::Integer)

    Wraps 1D operator `op` in a way that when lowered with nD list
    of coordinates, the operator is applied along `axis`.
    """
    function Lifted(op::O, axis::Integer) where {O <: Operator}
        axis > 0 || throw(ArgumentError("the lifted axis must be positive"))
        return new{O}(op, Int(axis))
    end
end

tensorrank(lf::Lifted, args) = tensorrank(lf.op, args)
checkranks(lf::Lifted, args::NTuple{N, DTerm}) where {N} = checkranks(lf.op, args)

function stencil_rule(lf::Lifted, args::Tuple{Vararg{DTerm}}, coords::NTuple{N, DTerm}) where {N}
    lf.axis <= N || throw(ArgumentError("the lifted axis exceeds the number of coordinates"))
    term = stencil_rule(lf.op, args, (coords[lf.axis],))::DTerm
    N == 1 && return term
    return insert_stencil(term, coords, lf.axis)
end

# replace 1D samples `f[c]` with nD samples `f[coords...]` where the coordinate along `axis` is `c`
@inline function insert_stencil(term, coords, axis)
    return @match term begin
        DExpr(Inds(), (arg, coord)) => begin
            ncoords = @inbounds Base.setindex(coords, coord, axis)
            return makeinds_unchecked(arg, ncoords)
        end
        DExpr(head, args) => something(insert_args(head, args, coords, axis), term)
        _ => return term
    end
end

@inline function insert_args(head, args, coords, axis)
    for k in eachindex(args)
        arg = insert_stencil(args[k], coords, axis)
        arg ==ₛ args[k] && continue
        return DExpr(head, insert_args(args, arg, coords, axis, k))
    end
    return nothing
end
@inline function insert_args(args::NTuple{N, DTerm}, arg, coords, axis, k) where {N}
    return ntuple(Val(N)) do j
        j < k ? args[j] : j == k ? arg : insert_stencil(args[j], coords, axis)
    end
end
