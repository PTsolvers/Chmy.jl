"""
    lower_rule(op, args, comps, face, x)
    lower_rule(op, args, face, x)
    lower_rule(op, args)

Expand the component `comps` of the call `op(args...)`, evaluated at the position `x` next to the `face`, into an
expression of terms sampled at offsets relative to `x`, e.g. `f[1//2, 0]`. The calls combining the samples are kept,
while the samples are lowered further. The position `x` is in the frame of [`lower`](@ref): relative to the grid
point of the kernel along the spanning axes `:` of the face, and to the wall along its sides `↓` and `↑`, or a global
position. Most rules only sample relative to `x` and ignore `face` and `x`; the rules that depend on the boundary
read them, e.g. overrides. Rules of scalar operators can omit `comps`, and rules of 1D operators can also omit `face`
and `x`.
"""
function lower_rule end

# scalar operators don't need the component
function lower_rule(op::Operator, args, comps, face, x)
    isempty(comps) || expansion_error(op)
    return lower_rule(op, args, face, x)
end

# 1D operators don't need the face and the position
function lower_rule(op::Operator, args, face, x)
    length(x) == 1 || no_rule_error(op, args, length(x))
    return lower_rule(op, args)
end

lower_rule(op::Operator, args) = no_rule_error(op, args, 1)

# pointwise functions are evaluated with tensor arguments in component form one level deep,
# computing the entire result to read its component `comps`
function lower_rule(op::Union{Fun, BroadcastedFun}, args, comps, face, x)
    result = evaluate(op, map(t -> components(t, length(x)), args))
    isnothing(result) || return here(result[comps...], x)
    # a scalar function of scalars reads its arguments at the evaluation point
    isempty(comps) || expansion_error(op)
    return makecall_unchecked(op, map(t -> here(t, x), args))
end

# the value of `t` at the evaluation point of a rule
here(t::DTerm, ::NTuple{N, Position}) where {N} = makeinds_unchecked(t, ntuple(_ -> 0 // 1, Val(N)))

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

function lower_rule(at::At, (t,)::Tuple{DTerm}, comps, face, x)
    length(at.positions) == length(x) || lower_dims_error()
    return makeinds_unchecked(t[comps...], at.positions)
end

"""
    lower(expr, face, x)

Lower the scalar expression `expr`, evaluated at the position `x` next to the `face`, e.g. `(:, :)` in the interior
or `(↓, :)` by the left wall, into samples of fields, e.g. `f[1//2, 0]`, expanding operator calls with their
[`lower_rule`](@ref). The positions are relative to the grid point of the kernel along the spanning axes `:` of the
face, and to the wall point along its sides `↓` and `↑`, in the direction of the axis: integer positions are points,
half-integer positions are segments. A global position `↓`, `↑` or `:` evaluates `expr` on the traces of the fields.
Samples nested in `expr` shift the position, so `lower(expr, face, x)` equals `lower(expr[x...], face, 0)`. Uniform
terms are kept as they are.
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
        DExpr(Call(op), args) => lower_call(op, args, (), face, x)
        DExpr(Comp(comps), (arg,)) => begin
            all(c -> 1 <= c <= N, comps) || throw(BoundsError(arg, comps))
            iscall(arg) ? lower_call(operation(arg), arguments(arg), comps, face, x) : leaf(term, x)
        end
        ZeroTensor(_) => ZERO
        IdTensor(_) => ONE
        Index(_) => index_input_error()
        Coordinate(a) => coordinate_value(a, x)
        _ => leaf(term, x)
    end
end

# the result of a rule combines uniform terms and samples relative to `x` with calls, which are kept as they are
function lowerrel(term::DTerm, face::Face{N}, x::NTuple{N, Position})::DTerm where {N}
    return @match term begin
        DExpr(Call(_), _) => walkargs(t -> lowerrel(t, face, x), term)
        DExpr(Inds(_), _) => lowerat(term, face, x)
        _ => isuniform(term) ? term : unsampled_error()
    end
end

# the result of the rule is a scalar expression of samples relative to `x`
function lower_call(op, args, comps, face, x)
    term = lower_rule(op, args, comps, face, x)::DTerm
    tensorrank(term) == 0 || rule_rank_error(op)
    return lowerrel(term, face, x)
end

# a field is sampled at its position, uniform terms are not sampled
leaf(t::DTerm, x) = isuniform(t) ? t : makeinds_unchecked(t, x)

# a free coordinate is the distance from the lower wall in grid units: the point `𝑖` lies at `𝑖 - 1`, where the grid
# index `𝑖` is the grid point of the kernel along the spanning axes of the face, and the wall point along its sides
function coordinate_value(a, x::NTuple{N, Position}) where {N}
    1 <= a <= N || coordinate_dims_error(a, N)
    return coordinate_value(a, x[a])
end
coordinate_value(a, p::Rational{Int}) = offset_term(Index(a), p - 1)
coordinate_value(a, ::Lower) = ZERO
coordinate_value(a, p::Global) = throw(ArgumentError(LazyString("coordinates are not defined at the global position ", LocationName(p))))

# canonical form `term`, `term + x`, or `term - x` of `term` shifted by `x`, with `x` positive
function offset_term(term::DTerm, x::Rational{Int})
    iszero(x) && return term
    return x > 0 ? term + stencil_constant(x) : term - stencil_constant(-x)
end

@noinline lower_nodims_error() = throw(ArgumentError("lowering requires at least one dimension"))
@noinline context_length_error() = throw(ArgumentError("the face and the position must have the same number of dimensions"))
@noinline lower_rank_error() = throw(ArgumentError("lower requires a scalar expression; select a scalar entry, e.g. expr[1, 2]"))
@noinline lower_dims_error() = throw(ArgumentError("samples must have as many positions as there are dimensions"))
@noinline unsampled_error() = throw(ArgumentError("lowering rules must read terms through samples, e.g. `f[0, 0]`"))
@noinline index_input_error() = throw(ArgumentError("grid indices are produced by lowering and cannot appear in its input, use coordinates instead"))
@noinline function coordinate_dims_error(a, n)
    throw(ArgumentError(LazyString("coordinate `", Coordinate(a), "` exceeds the number of dimensions ", n)))
end
@noinline rule_rank_error(op) = throw(ArgumentError(LazyString("the lowering rule of '", op, "' must return a scalar")))
@noinline function no_rule_error(op, args, n)
    throw(ArgumentError(LazyString("operator '", op, "' has no lowering rule for ", length(args), " arguments in ", n, " dimensions")))
end
