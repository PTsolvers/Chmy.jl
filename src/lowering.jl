"""
    lower_rule(op, args, inds)

Expand the scalar call `op(args...)` at the absolute positions `inds`, one [`Position`](@ref) per axis, into a scalar
expression of samples. The arguments are in component form, see [`components`](@ref), and the rule samples them with
`getindex`, which lowers them in turn, e.g. `f[i + 1//2] - f[i - 1//2]`; a term read at the evaluation point is
`f[inds...]`. Rules depending on the boundary read the frames of the positions, see [`Coord`](@ref).

The rule of an operator with one argument can be a method of `lower_rule(op, arg::DTerm, inds)`, a rule in one
dimension a method of `lower_rule(op, args, i::Position)`, and both `lower_rule(op, arg::DTerm, i::Position)`. The
argument is unwrapped before the position, and the unwrapped parameters must be typed as shown.
"""
lower_rule(op::Operator, args, inds) = _lower_rule(op, args, inds)

# the rule of the only argument, then the rule at the only position
_lower_rule(op, (arg,)::Tuple{DTerm}, inds::Tuple) = lower_rule(op, arg, inds)
_lower_rule(op, args::Union{DTerm, Tuple}, (i,)::Tuple{Position}) = lower_rule(op, args, i)
_lower_rule(op, (arg,)::Tuple{DTerm}, inds::Tuple{Position}) = lower_rule(op, arg, inds)
_lower_rule(op, args, inds) = no_rule_error(op, args, inds)

# a pointwise function is the function of its arguments sampled at the evaluation point
lower_rule(op::Fun, args::Tuple, inds::Tuple) = makecall(op, map(a -> a[inds...], args))

"""
    Project(element, axis)

Projects its argument, which may be a tensor, on the trace `element` along `axis`: the lower wall `↓`, the upper wall
`↑`, or the value independent of the axis `:`, see [`Trace`](@ref), e.g. `Project(↑, 2)(S)` is `S` on the top wall in
2D. The other axes are read at the evaluation point, and the projection is constant along `axis`.
"""
struct Project <: Operator
    element::Global
    axis::Int
    function Project(element::Global, axis::Integer)
        axis > 0 || throw(ArgumentError("the projected axis must be positive"))
        return new(element, axis)
    end
end

tensorrank(::Project, args) = tensorrank(only(args))
checkranks(::Project, args::Tuple{Vararg{DTerm}}) = length(args) == 1 || throw(ArgumentError("a projection requires one argument"))

# constants are not projected
evaluate(::Project, (t,)::Tuple{DTerm}) = isconstant(t) ? t : nothing
component(p::Project, args, I) = componentwise(p, args, I)

function lower_rule(p::Project, t::DTerm, inds::NTuple{N, Position}) where {N}
    p.axis <= N || throw(ArgumentError(LazyString("the projected axis ", p.axis, " exceeds the number of dimensions ", N)))
    return t[Base.setindex(inds, Trace(p.element), p.axis)...]
end

# the term in component form sampled at the positions `x`: calls are expanded with their rules, constants are kept,
# and a coordinate is the sum of the coordinate of the origin of the frame and the offset
function sample(term::DTerm, x::NTuple{N, Position})::DTerm where {N}
    return @match term begin
        DExpr(Inds(_), _) => resample_error()
        DExpr(Call(op), args) => sample_call(op, args, x)
        AddMul(_, _, _) => mapentries(t -> sample(t, x), term)
        # components of calls kept in component form, e.g. `𝐧[1]`, have no value at the evaluation point
        DExpr(Comp(_), (arg,)) => iscall(arg) ? unknown_component_error(term) : sample_leaf(term, x)
        Coordinate(_) => coordinate_value(term, x)
        _ => sample_leaf(term, x)
    end
end

# constants are the same at every position
sample_leaf(term, x) = isconstant(term) ? term : makeinds_unchecked(term, x)

# a call is the scalar of samples returned by its rule
function sample_call(op, args, x)::DTerm
    term = lower_rule(op, args, x)::DTerm
    tensorrank(term) == 0 || rule_rank_error(op)
    return term
end

# a coordinate is the distance from the lower wall in grid units, the coordinate of the origin of the frame plus the
# offset, and the coordinate of the wall at a trace
function coordinate_value(ξ, x::NTuple{N, Position}) where {N}
    1 <= value(ξ) <= N || coordinate_dims_error(value(ξ), N)
    return coordinate_value(ξ, x[value(ξ)], Val(N))
end
coordinate_value(ξ, c::Coord, n) = origin(ξ, c.frame, n) + c.offset
coordinate_value(ξ, t::Trace, n) = trace_coordinate(ξ, t.element, n)

trace_coordinate(ξ, side::Side, n) = origin(ξ, side, n)
trace_coordinate(ξ, ::Colon, n) = throw(ArgumentError(LazyString("the coordinate `", ξ, "` varies along its axis")))

# the coordinate of the origin of the frame: zero at the lower wall, and otherwise a sample of the coordinate, which is
# constant along the other axes, built unchecked as it is lowered already
origin(ξ, ::Lower, n) = ZERO
origin(ξ, frame::Union{Colon, Upper}, n) = makeinds_unchecked(ξ, Base.setindex(ntuple(_ -> Trace(:), n), Coord(frame, 0), value(ξ)))

@noinline resample_error() = throw(ArgumentError("samples cannot be sampled again"))
@noinline unknown_component_error(term) = throw(ArgumentError(LazyString("the component `", term, "` has no value at the evaluation point")))
@noinline function coordinate_dims_error(a, n)
    throw(ArgumentError(LazyString("coordinate `", Coordinate(a), "` exceeds the number of dimensions ", n)))
end
@noinline rule_rank_error(op) = throw(ArgumentError(LazyString("the lowering rule of '", op, "' must return a scalar")))
@noinline function no_rule_error(op, args, inds)
    throw(ArgumentError(LazyString("operator '", op, "' has no lowering rule for ", count_of(args), " arguments in ", count_of(inds), " dimensions")))
end

count_of(x::Tuple) = length(x)
count_of(x) = 1
