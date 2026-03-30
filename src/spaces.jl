abstract type Space <: STerm end

struct Segment <: Space end
struct Point <: Space end

scale(::Segment)  = 1
offset(::Segment) = -1

scale(::Point)  = 1
offset(::Point) = 0

function Base.getindex(t::STerm, loc::Vararg{Space})
    length(loc) == 0 && return t
    SExpr(Loc(), t, loc...)
end

Base.getindex(s::SLiteral, ::Vararg{Space}) = s

# `Node` is transparent to location indexing, but the resulting term stays
# wrapped so later substitutions can still target the protected subtree.
function Base.getindex(expr::SExpr{Node}, loc::Vararg{Space,N}) where {N}
    N == 0 && return expr
    return node(argument(expr)[loc...])
end

function Base.getindex(sub::SExpr{Ind}, loc::Vararg{Space})
    length(loc) == 0 && return sub
    inds = indices(sub)
    arg  = argument(sub)
    return arg[loc...][inds...]
end
