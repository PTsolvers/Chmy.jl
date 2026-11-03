"""
    lower_rule(op, args, comps, coord, ctx)
    lower_rule(op, args, coord, ctx)
    lower_rule(op, args)

Expand the component `comps` of the call `op(args...)`, sampled at the coordinate `coord` relative to the evaluation
point, into an expression of terms sampled at offsets relative to this sample, e.g. `f[1//2, 0]`. The calls combining
the samples are kept, while the samples are lowered further. Most rules only sample relative to themselves and ignore
`coord` and `ctx`; the lowering context `ctx` holds the face and the origin given to [`lower`](@ref) for the rules that
depend on where the sample is relative to the face. Rules of scalar operators can omit `comps`, and rules of 1D
operators can also omit `coord` and `ctx`.
"""
function lower_rule end

# scalar operators don't need the component
function lower_rule(op::Operator, args, comps, coord, ctx)
    isempty(comps) || expansion_error(op)
    return lower_rule(op, args, coord, ctx)
end

# 1D operators don't need the coordinate and the context
function lower_rule(op::Operator, args, coord, ctx)
    length(coord) == 1 || no_rule_error(op, args, length(coord))
    return lower_rule(op, args)
end

lower_rule(op::Operator, args) = no_rule_error(op, args, 1)

# pointwise functions are evaluated with tensor arguments in component form one level deep,
# computing the entire result to read its component `comps`
function lower_rule(op::Union{Fun, BroadcastedFun}, args, comps, coord, ctx)
    result = evaluate(op, map(x -> components(x, length(coord)), args))
    isnothing(result) || return here(result[comps...], coord)
    # a scalar function of scalars reads its arguments at the evaluation point
    isempty(comps) || expansion_error(op)
    return makecall_unchecked(op, map(x -> here(x, coord), args))
end

# the value of `x` at the evaluation point of a rule
here(x::DTerm, coord) = makeinds(x, map(zero, coord))

"""
    lower(expr, face, origin)

Lower the scalar expression `expr` next to the `face`, e.g. `(:, :)` in the interior or `(↓, :)` by the left wall, into
located samples of fields, e.g. `f[𝓈, 𝓅][0, 1]`, expanding operator calls with their [`lower_rule`](@ref). The `origin`
is the position of the evaluation point: the lattice coordinate along the spanning axes `:` (`0` for points and `1//2`
for segments), and the outward distance from the wall along the sides `↓` and `↑`. Samples at integer coordinates are
located at points, and at half-integer ones at segments; their indices are relative to the grid point of the kernel,
which is the wall point along the sides. Uniform terms are kept as they are.
"""
function lower(expr::DTerm, face::Face{N}, origin::NTuple{N, Union{Integer, Rational}}) where {N}
    all(x -> denominator(x) <= 2, origin) || origin_error()
    haslocs(expr) && located_input_error()
    coord = zero_offsets(Val(N))
    return lower(makeinds_unchecked(expr, coord), coord, LoweringContext(face, map(Rational{Int}, origin)))
end
lower(::DTerm, ::Tuple{}, ::Tuple{}) = lower_nodims_error()
lower(::DTerm, ::Tuple, ::Tuple) = context_length_error()

function haslocs(term::DTerm)::Bool
    return @match term begin
        DExpr(Locs(_), _) => true
        DExpr(_, args) => any(haslocs, args)
        _ => false
    end
end

# the result of a rule combines uniform terms and samples relative to `coord` with calls, which are kept as they are
function lower(term::DTerm, coord::Coord{N}, ctx::LoweringContext{N})::DTerm where {N}
    return @match term begin
        DExpr(Call(_), _) => walkargs(x -> lower(x, coord, ctx), term)
        DExpr(Inds(_), _) => lowerterm(term, coord, ctx)
        # a leaf located only along global axes, e.g. `g[↓, :]`
        DExpr(Locs(_), _) => term
        _ => isuniform(term) ? term : unsampled_error()
    end
end

# lower the sample `term` relative to the sample at `base`, accumulating the offsets until the sampled term is a field,
# or a scalar expression that is expanded with its rule
function lowerterm(term::DTerm, base::Coord{N}, ctx::LoweringContext{N})::DTerm where {N}
    return @match term begin
        # located leaves are produced by rules, e.g. for boundary data, and are kept as they are
        DExpr(Inds(_), (inner,)) && if islocs(inner) end => term
        DExpr(Inds(shift), (inner,)) => lowersample(inner, shift, base, ctx)
        _ => unsampled_error()
    end
end

function lowersample(inner::DTerm, shift, base::Coord{N}, ctx::LoweringContext{N})::DTerm where {N}
    length(shift) == N || lower_dims_error()
    tensorrank(inner) == 0 || lower_rank_error()
    coord = ntuple(a -> base[a] + shift[a], Val(N))
    return @match inner begin
        DExpr(Inds(_), _) => lowerterm(inner, coord, ctx)
        DExpr(Call(op), args) => lower_call(op, args, (), coord, ctx)
        DExpr(Comp(comps), (arg,)) => begin
            all(c -> 1 <= c <= N, comps) || throw(BoundsError(arg, comps))
            iscall(arg) ? lower_call(operation(arg), arguments(arg), comps, coord, ctx) : leaf(inner, coord, ctx)
        end
        ZeroTensor(_) => ZERO
        IdTensor(_) => ONE
        Index(_) => index_input_error()
        Coordinate(a) => coordinate_value(a, coord, ctx)
        _ => leaf(inner, coord, ctx)
    end
end

# the result of the rule is a scalar expression of samples relative to `coord`
function lower_call(op, args, comps, coord, ctx)
    term = lower_rule(op, args, comps, coord, ctx)::DTerm
    tensorrank(term) == 0 || rule_rank_error(op)
    return lower(term, coord, ctx)
end

zero_offsets(::Val{N}) where {N} = ntuple(_ -> 0 // 1, Val(N))

sample_location(x::Rational{Int}) = isinteger(x) ? Point() : Segment()

# the leaf of the scalar `x` sampled at `coord`, located at the point or segment of the sample along the axes where
# `frames` is `nothing`, with indices relative to the grid point of the kernel, and at the global locations `frames`
# without indices along the other axes; uniform terms are not located
leaf(x::DTerm, coord::Coord{N}, ctx::LoweringContext{N}) where {N} = leaf(x, ntuple(_ -> nothing, Val(N)), coord, ctx)
function leaf(x::DTerm, frames, coord::Coord{N}, ctx::LoweringContext{N}) where {N}
    isuniform(x) && return x
    xs = ntuple(a -> position(ctx, coord, a), Val(N))
    located = makelocs_unchecked(x, map(frame_location, frames, xs))
    inds = frame_indices(frames, xs)
    return isempty(inds) ? located : makeinds_unchecked(located, inds)
end

frame_location(::Nothing, x) = sample_location(x)
frame_location(loc::Location, x) = loc

frame_indices(::Tuple{}, ::Tuple{}) = ()
frame_indices(frames, xs) = (frame_index(first(frames), first(xs))..., frame_indices(Base.tail(frames), Base.tail(xs))...)

frame_index(::Nothing, x) = (floor(x),)
frame_index(::Location, x) = ()

# a free coordinate is the distance of the sample from the lower wall in grid units,
# point `1` lies on the lower wall, so point `i` is at `i - 1` and segment `i` at `i - 1//2`
function coordinate_value(a, coord::Coord{N}, ctx::LoweringContext{N}) where {N}
    1 <= a <= N || coordinate_dims_error(a, N)
    return offset_term(Index(a), position(ctx, coord, a) - 1)
end

# canonical form `term`, `term + x`, or `term - x` of `term` shifted by `x`, with `x` positive
function offset_term(term::DTerm, x::Rational{Int})
    iszero(x) && return term
    return x > 0 ? term + stencil_constant(x) : term - stencil_constant(-x)
end

@noinline lower_nodims_error() = throw(ArgumentError("lowering requires at least one dimension"))
@noinline context_length_error() = throw(ArgumentError("the face and the origin must have the same number of dimensions"))
@noinline origin_error() = throw(ArgumentError("the origin must be a multiple of 1//2"))
@noinline located_input_error() = throw(ArgumentError("locations are produced by lowering, use `𝒷` for boundary data"))
@noinline lower_rank_error() = throw(ArgumentError("lower requires a scalar expression; select a scalar entry, e.g. expr[1, 2]"))
@noinline lower_dims_error() = throw(ArgumentError("samples must have as many offsets as there are dimensions"))
@noinline unsampled_error() = throw(ArgumentError("lowering rules must read terms through samples, e.g. `f[0, 0]`"))
@noinline index_input_error() = throw(ArgumentError("grid indices are produced by lowering and cannot appear in its input, use coordinates instead"))
@noinline function coordinate_dims_error(a, n)
    throw(ArgumentError(LazyString("coordinate `", Coordinate(a), "` exceeds the number of dimensions ", n)))
end
@noinline rule_rank_error(op) = throw(ArgumentError(LazyString("the lowering rule of '", op, "' must return a scalar")))
@noinline function no_rule_error(op, args, n)
    throw(ArgumentError(LazyString("operator '", op, "' has no lowering rule for ", length(args), " arguments in ", n, " dimensions")))
end
