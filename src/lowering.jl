"""
    lower_rule(op, args, face, x)
    lower_rule(op, args)

Expand the scalar call `op(args...)`, evaluated at the position `x` next to the `face`, into a scalar expression
evaluated at `x`, which is lowered further. Samples in the expression are relative to `x`, e.g. `f[1//2, 0]`, and
terms that are not sampled are read at `x`, e.g. `f[1//2, 0] - f`. A rule returning a call of its own operator, e.g.
`op(args...)`, keeps the call and lowers only its arguments. The position `x` is in the frame of [`lower`](@ref):
relative to the grid point of the kernel along the spanning axes `:` of the face, and to the wall along its sides `↓`
and `↑`, or a global position. Tensor arguments are in component form, see [`components`](@ref). Calls with a
component form in the number of dimensions, see [`evaluate`](@ref), are lowered through it instead. Most rules only
sample relative to `x` and ignore `face` and `x`; the rules that depend on the boundary read them, e.g. overrides.
Rules of 1D operators can omit `face` and `x`.
"""
function lower_rule end

# 1D operators don't need the face and the position
function lower_rule(op::Operator, args, face, x)
    length(x) == 1 || no_rule_error(op, args, length(x))
    return lower_rule(op, args)
end

lower_rule(op::Operator, args) = no_rule_error(op, args, 1)

# a pointwise function is the call itself, which is kept, and its arguments are lowered at the evaluation point
lower_rule(op::Fun, args, face, x) = makecall(op, args)

"""
    At(positions...)

Samples its argument, which may be a tensor, at `positions` relative to the evaluation point, e.g. `At(↓, 0)(W)` is
the trace of `W` on the lower wall of the first axis. See [`getindex`](@ref) for the positions.
"""
struct At <: Operator
    positions::Tuple{Vararg{Position}}
    At(positions::Tuple{Vararg{PositionLike}}) = new(map(asposition, positions))
end
At(positions::Vararg{PositionLike}) = At(positions)

tensorrank(::At, args) = tensorrank(only(args))
checkranks(::At, args::Tuple{Vararg{DTerm}}) = length(args) == 1 || throw(ArgumentError("`At` requires one argument"))

# scalars are sampled right away, and tensors component by component
evaluate(at::At, (t,)::Tuple{DTerm}) = tensorrank(t) == 0 ? makeinds(t, at.positions) : nothing
component(at::At, (t,)::Tuple{DTerm}, I) = makeinds(component(t, I), at.positions)

"""
    lower(expr, face, x)

Lower the scalar expression `expr`, evaluated at the position `x` next to the `face`, e.g. `(:, :)` in the interior
or `(↓, :)` by the left wall, into samples of fields, e.g. `f[1//2, 0]`, evaluating calls in component form in
`length(x)` dimensions, see [`evaluate`](@ref), and expanding the other calls with their [`lower_rule`](@ref). The
positions are relative to the grid point of the kernel along the spanning axes `:` of the
face, and to the wall point along its sides `↓` and `↑`, in the direction of the axis: integer positions are points,
half-integer positions are segments. A global position `↓`, `↑` or `:` evaluates `expr` on the traces of the fields.
Samples nested in `expr` shift the position, so `lower(expr, face, x)` equals `lower(expr[x...], face, 0)`. Uniform
terms are kept as they are. The result is in canonical form, so samples multiplied by zero are dropped.
"""
function lower(expr::DTerm, face::Face{N}, x::NTuple{N, PositionLike}) where {N}
    return lowerat(expr, face, map(asposition, x))
end
lower(::DTerm, ::Tuple{}, ::Tuple{}) = lower_nodims_error()
lower(::DTerm, ::Tuple, ::Tuple) = context_length_error()

# the term evaluated at the position `x`
function lowerat(term::DTerm, face::Face{N}, x::NTuple{N, Position})::DTerm where {N}
    tensorrank(term) == 0 || lower_rank_error()
    return @match term begin
        DExpr(Inds(s), (inner,)) => begin
            length(s) == N || lower_dims_error()
            lowerat(inner, face, map(shift, x, s))
        end
        DExpr(Call(op), args) => lower_call(op, args, face, x)
        AddMul(_, _, _) => mapentries(t -> lowerat(t, face, x), term)
        DExpr(Comp(comps), (arg,)) => begin
            all(c -> 1 <= c <= N, comps) || throw(BoundsError(arg, comps))
            # a component of a call is read from the call in component form
            iscall(arg) ? lowerat(component(components(arg, N), comps), face, x) : makeinds(term, x)
        end
        Index(_) => index_input_error()
        Coordinate(a) => coordinate_value(a, x)
        _ => makeinds(term, x)
    end
end

# the call with its tensor arguments in component form is evaluated in `N` dimensions if possible, and expanded with
# its rule otherwise; the result of the rule is a scalar expression evaluated at `x`, which is lowered further, unless
# the rule returns a call of its own operator, which is kept with its arguments lowered
function lower_call(op, args, face::Face{N}, x) where {N}
    packed = map(t -> components(t, N), args)
    evaluated = evaluate(op, packed, N)::Union{Nothing, DTerm}
    isnothing(evaluated) || return lowerat(evaluated, face, x)
    term = lower_rule(op, packed, face, x)::DTerm
    tensorrank(term) == 0 || rule_rank_error(op)
    iscall(term) && operation(term) == op && return walkargs(t -> lowerat(t, face, x), term)
    return lowerat(term, face, x)
end

# a free coordinate is the distance from the lower wall in grid units: the point `𝑖` lies at `𝑖 - 1`, where the grid
# index `𝑖` is the grid point of the kernel along the spanning axes of the face, and the wall point along its sides
function coordinate_value(a, x::NTuple{N, Position}) where {N}
    1 <= a <= N || coordinate_dims_error(a, N)
    return coordinate_value(a, x[a])
end
coordinate_value(a, p::Rational{Int}) = Index(a) + (p - 1)
coordinate_value(a, ::Lower) = ZERO
coordinate_value(a, p::Global) = throw(ArgumentError(LazyString("coordinates are not defined at the global position ", LocationName(p))))

@noinline lower_nodims_error() = throw(ArgumentError("lowering requires at least one dimension"))
@noinline context_length_error() = throw(ArgumentError("the face and the position must have the same number of dimensions"))
@noinline lower_rank_error() = throw(ArgumentError("lower requires a scalar expression; select a scalar entry, e.g. expr[1, 2]"))
@noinline lower_dims_error() = throw(ArgumentError("samples must have as many positions as there are dimensions"))
@noinline index_input_error() = throw(ArgumentError("grid indices are produced by lowering and cannot appear in its input, use coordinates instead"))
@noinline function coordinate_dims_error(a, n)
    throw(ArgumentError(LazyString("coordinate `", Coordinate(a), "` exceeds the number of dimensions ", n)))
end
@noinline rule_rank_error(op) = throw(ArgumentError(LazyString("the lowering rule of '", op, "' must return a scalar")))
@noinline function no_rule_error(op, args, n)
    throw(ArgumentError(LazyString("operator '", op, "' has no lowering rule for ", length(args), " arguments in ", n, " dimensions")))
end
