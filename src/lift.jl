struct Lifted{O<:Operator} <: Operator
    op::O
    axis::Int

    """
    Lifted(op::Operator, axis::Integer)

    Wraps 1D operator `op` in a way that when lowered with nD list
    of locations and/or indices, the operator is applied along `axis`.
    """
    function Lifted(op::O, axis::Integer) where {O<:Operator}
        axis > 0 || throw(ArgumentError("the lifted axis must be positive"))
        return new{O}(op, Int(axis))
    end
end

tensorrank(lf::Lifted, args) = tensorrank(lf.op, args)
checkranks(lf::Lifted, args::NTuple{N,DTerm}) where {N} = checkranks(lf.op, args)

struct Insert{L,N}
    locs::L
    inds::NTuple{N,DTerm}
end

function stencil_rule(lf::Lifted, args::Tuple{Vararg{DTerm}}, inds::NTuple{N,DTerm}) where {N}
    lf.axis <= N || throw(ArgumentError("the lifted axis exceeds the number of grid indices"))
    term = stencil_rule(lf.op, args, (inds[lf.axis],))::DTerm
    N == 1 && return term
    return insert_stencil(term, Insert(nothing, inds), lf.axis)
end
function stencil_rule(lf::Lifted, args::Tuple{Vararg{DTerm}}, locs::NTuple{N,Location}, inds::NTuple{N,DTerm}) where {N}
    lf.axis <= N || throw(ArgumentError("the lifted axis exceeds the number of grid indices"))
    term = stencil_rule(lf.op, args, (locs[lf.axis],), (inds[lf.axis],))::DTerm
    N == 1 && return term
    return insert_stencil(term, Insert(locs, inds), lf.axis)
end

@inline function insert_stencil(term, ctx::Insert{Nothing}, axis)
    @match term begin
        DExpr(Inds(), (arg, ind)) => begin
            ninds = @inbounds Base.setindex(ctx.inds, ind, axis)
            return makeinds_unchecked(arg, ninds)
        end
        DExpr(head, args) => something(insert_args(head, args, ctx, axis), term)
        _ => return term
    end
end
@inline function insert_stencil(term, ctx, axis)
    @match term begin
        DExpr(Inds(), (arg, ind)) => begin
            @match arg begin
                DExpr(Locs((loc,)), (arg2,)) => begin
                    ninds = @inbounds Base.setindex(ctx.inds, ind, axis)
                    nlocs = @inbounds Base.setindex(ctx.locs, loc, axis)
                    return makeinds_unchecked(makelocs_unchecked(arg2, nlocs), ninds)
                end
                _ => throw(ArgumentError("location-aware stencil rules must define locations on fields"))
            end
        end
        DExpr(head, args) => something(insert_args(head, args, ctx, axis), term)
        _ => return term
    end
end

@inline function insert_args(head, args, ctx, axis)
    for k in eachindex(args)
        arg = insert_stencil(args[k], ctx, axis)
        arg==ₛargs[k] && continue
        return DExpr(head, insert_args(args, arg, ctx, axis, k))
    end
    return nothing
end
@inline function insert_args(args::NTuple{N,DTerm}, arg, ctx, axis, k) where {N}
    return ntuple(Val(N)) do j
        j < k ? args[j] : j == k ? arg : insert_stencil(args[j], ctx, axis)
    end
end
