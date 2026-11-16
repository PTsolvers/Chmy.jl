"""
    lower_rule(op, args, inds)

Expand the scalar call `op(args...)` at the absolute positions `inds`, one [`Position`](@ref) per axis, into an
expression of samples of the arguments at absolute positions, e.g. `f[i + 1//2] - f[i - 1//2]`, which is lowered
further. The arguments are not lowered, and are in component form, see [`components`](@ref). Terms that are not
sampled are read at `inds`, e.g. `f[i + 1//2] - f`, and a rule returning a call of its own operator, e.g.
`op(args...)`, keeps the call and lowers its arguments at `inds`. Rules depending on the boundary read the frames of
the positions, see [`Coord`](@ref).

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

# a pointwise function is the call itself, which is kept, and its arguments are lowered at the evaluation point
lower_rule(op::Fun, args::Tuple, inds::Tuple) = makecall(op, args)

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
    return makeinds(t, Base.setindex(inds, Trace(p.element), p.axis))
end

"""
    lower(term)

Lower the samples in `term`, e.g. `divg(grad(u))[(:, 1//2), (:, 1//2)]`: the sampled scalar expression is expanded in
component form in as many dimensions as there are positions, see [`components`](@ref), and its calls with their
[`lower_rule`](@ref), into samples of fields at absolute positions, e.g. `u[(:, -1//2), (:, 1//2)]`. Uniform terms
are kept as they are. A coordinate is the sum of the coordinate of the origin of the frame, sampled, e.g.
`ξ¹[(:, 0), :]`, and the offset, e.g. `ξ¹[(:, 0), :] + 1//2`; it is the offset itself in the frame of the lower wall.
The result is in canonical form, so samples multiplied by zero are dropped.
"""
function lower(term::DTerm)::DTerm
    return @match term begin
        DExpr(Inds(x), (inner,)) => lower_inds(inner, x)
        AddMul(_, _, _) => mapentries(lower, term)
        _ => walkargs(lower, term)
    end
end

# the expression sampled at the positions `x`
lower_inds(inner::DTerm, x::Tuple{Vararg{Position}}) = lowerat(components(inner, length(x)), x)

# the term in component form evaluated at the positions `x`
function lowerat(term::DTerm, x::NTuple{N, Position})::DTerm where {N}
    tensorrank(term) == 0 || lower_rank_error()
    return @match term begin
        DExpr(Inds(s), (inner,)) => begin
            length(s) == N || lower_dims_error()
            lowerat(inner, s)
        end
        DExpr(Call(op), args) => lower_call(op, args, x)
        AddMul(_, _, _) => mapentries(t -> lowerat(t, x), term)
        # components of calls kept in component form, e.g. `𝐧[1]`, have no value at the evaluation point
        DExpr(Comp(_), (arg,)) => iscall(arg) ? unknown_component_error(term) : makeinds(term, x)
        Coordinate(_) => coordinate_value(term, x)
        _ => makeinds(term, x)
    end
end

# the call in component form is expanded with its rule; the result of the rule is a scalar expression evaluated at `x`,
# which is lowered further, unless the rule returns a call of its own operator, which is kept with its arguments lowered
function lower_call(op, args, x)::DTerm
    term = lower_rule(op, args, x)::DTerm
    tensorrank(term) == 0 || rule_rank_error(op)
    iscall(term) && operation(term) == op && return walkargs(t -> lowerat(t, x), term)
    return lowerat(term, x)
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

@noinline lower_rank_error() = throw(ArgumentError("lower requires a scalar expression; select a scalar entry, e.g. expr[1, 2]"))
@noinline lower_dims_error() = throw(ArgumentError("samples must have as many positions as there are dimensions"))
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
