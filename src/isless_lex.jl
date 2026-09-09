# Ranks preserve the ordering of the static symbolic core.
function termrank(term)
    @match term begin
        Index(_) => 0
        Tensor(_, _, _, _) => 1
        ZeroTensor(_) => 2
        IdTensor(_) => 3
        Literal(_) => 4
        DExpr(_, _) => 5
    end
end

function headrank(head)
    @match head begin
        Comp(_) => 0
        Locs(_) => 1
        Inds() => 2
        Call(_) => 3
    end
end

(<ₛ)(x, y) = x < y

# Operator values are not DTerms. User-defined operators retain identity ordering.
(<ₛ)(x::Fun, y::Fun) = isless(nameof(x.f), nameof(y.f))
(<ₛ)(::Fun, ::Operator) = true
(<ₛ)(::Operator, ::Fun) = false
(<ₛ)(x::Operator, y::Operator) = isless(objectid(x), objectid(y))

(<ₛ)(x::Location, y::Location) = x isa Point && y isa Segment

"""
    x <ₛ y

Compare dynamic symbolic terms lexicographically for canonical factor ordering.
"""
function (<ₛ)(x::DTerm, y::DTerm)
    tx, ty = tensorrank(x), tensorrank(y)
    tx == ty || return tx < ty

    rx, ry = termrank(x), termrank(y)
    rx == ry || return rx < ry

    @match (x, y) begin
        (Index(i), Index(j)) => i < j
        (Literal(a), Literal(b)) => isless(a, b)::Bool
        (Tensor(_, nx, _, _), Tensor(_, ny, _, _)) => begin
                                                      nx == ny || return isless(nx, ny)
                                                      x==ₛy || throw(ArgumentError("tensors with the same name must have the same rank, kind, and uniformity"))
                                                      false
                                                      end
        (DExpr(_, _), DExpr(_, _)) => isless_expr(x, y)
        _ => false # Equal-rank zero and identity tensors.
    end
end

function isless_expr(x, y)::Bool
    hx, hy = head(x), head(y)
    ax, ay = args(x), args(y)
    rx, ry = headrank(hx), headrank(hy)
    rx == ry || return rx < ry

    @match (hx, hy) begin
        (Call(opx), Call(opy)) => begin
            opx == opy || return (opx<ₛopy)::Bool
            isless_args(ax, ay)
        end
        (Comp(ix), Comp(iy)) || (Locs(ix), Locs(iy)) => begin
            ax[1]==ₛay[1] || return ax[1]<ₛay[1]
            isless_args(ix, iy)
        end
        _ => isless_args(ax, ay) # Grid indexing stores argument and indices together.
    end
end

function isless_args(xs, ys)
    for i in 1:min(length(xs), length(ys))
        x, y = xs[i], ys[i]
        isequal(x, y) || return x<ₛy
    end
    return length(xs) < length(ys)
end
