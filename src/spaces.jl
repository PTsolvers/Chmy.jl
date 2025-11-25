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

Base.getindex(s::SUniform, ::Vararg{Space}) = s

function Base.getindex(sub::SExpr{Ind}, loc::Vararg{Space})
    inds = indices(sub)
    arg  = argument(sub)
    return SExpr(Ind(), SExpr(Loc(), arg, loc...), inds...)
end
