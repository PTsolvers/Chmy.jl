"""
    lower_rule(op, args, comps, offs, block)
    lower_rule(op, args, offs, block)
    lower_rule(op, args)

Expand the component `comps` of the call `op(args...)`, sampled at the offsets `offs` from the evaluation point in
the `block` of the domain, into an expression of terms sampled at offsets relative to this sample, e.g. `f[1//2, 0]`.
The calls combining the samples are kept, while the samples are lowered further. Rules of scalar operators can omit
`comps`, and rules of 1D operators can also omit `offs` and `block`.
"""
function lower_rule end

# scalar operators don't need the component
function lower_rule(op::Operator, args, comps, offs, block)
    isempty(comps) || expansion_error(op)
    return lower_rule(op, args, offs, block)
end

# 1D operators don't need the offsets and the block
function lower_rule(op::Operator, args, offs, block)
    length(offs) == 1 || no_rule_error(op, args, length(offs))
    return lower_rule(op, args)
end

lower_rule(op::Operator, args) = no_rule_error(op, args, 1)

# pointwise functions are evaluated with tensor arguments in component form one level deep,
# computing the entire result to read its component `comps`
function lower_rule(op::Union{Fun, BroadcastedFun}, args, comps, offs, block)
    result = evaluate(op, map(x -> components(x, length(offs)), args))
    isnothing(result) || return here(result[comps...], offs)
    # a scalar function of scalars reads its arguments at the evaluation point
    isempty(comps) || expansion_error(op)
    return makecall_unchecked(op, map(x -> here(x, offs), args))
end

# the value of `x` at the evaluation point of a rule
here(x::DTerm, offs) = makeinds(x, map(zero, offs))

"""
    lower(expr, block)

Lower the scalar expression `expr`, evaluated in the `block` of the domain, into samples of fields at offsets from the
evaluation point, e.g. `f[1//2, 0]`, expanding operator calls with their [`lower_rule`](@ref). Uniform terms are kept as
they are, and explicit locations can only be given to fields. Use [`locate`](@ref) to place the samples on the grid.
"""
function lower(expr::DTerm, block::Block{N}) where {N}
    offs = zero_offsets(Val(N))
    return lower(makeinds_unchecked(expr, offs), offs, block)
end
lower(::DTerm, ::Tuple{}) = lower_nodims_error()

# the result of a rule combines uniform terms and samples relative to `offs` with calls, which are kept as they are
function lower(term::DTerm, offs::NTuple{N, Rational{Int}}, block::Block{N})::DTerm where {N}
    return @match term begin
        DExpr(Call(_), _) => walkargs(x -> lower(x, offs, block), term)
        DExpr(Inds(_), _) => lowerterm(term, offs, block)
        _ => isuniform(term) ? term : unsampled_error()
    end
end

# lower the sample `term` relative to the sample at `base`, accumulating the offsets until the sampled term is a field,
# or a scalar expression that is expanded with its rule
function lowerterm(term::DTerm, base::NTuple{N, Rational{Int}}, block::Block{N})::DTerm where {N}
    shift, inner = @match term begin
        DExpr(Inds(shift), (inner,)) => (shift, inner)
        _ => unsampled_error()
    end
    length(shift) == N || lower_dims_error()
    tensorrank(inner) == 0 || lower_rank_error()
    offs = ntuple(a -> base[a] + shift[a], Val(N))
    return @match inner begin
        DExpr(Inds(_), _) => lowerterm(inner, offs, block)
        DExpr(Call(op), args) => lower_call(op, args, (), offs, block)
        DExpr(Comp(comps), (arg,)) => begin
            all(c -> 1 <= c <= N, comps) || throw(BoundsError(arg, comps))
            iscall(arg) ? lower_call(operation(arg), arguments(arg), comps, offs, block) : sample(inner, offs)
        end
        DExpr(Locs(_), (arg,)) => isuniform(arg) ? arg : isfield(arg) ? sample(inner, offs) : locs_field_error()
        ZeroTensor(_) => ZERO
        IdTensor(_) => ONE
        Index(_) => index_input_error()
        _ => sample(inner, offs)
    end
end

# the result of the rule is a scalar expression of samples relative to `offs`
function lower_call(op, args, comps, offs, block)
    term = lower_rule(op, args, comps, offs, block)::DTerm
    tensorrank(term) == 0 || rule_rank_error(op)
    return lower(term, offs, block)
end

# a field sampled at `offs`, uniform fields are not sampled
sample(field::DTerm, offs) = isuniform(field) ? field : makeinds_unchecked(field, offs)

# a stored scalar field, or a component of a stored tensor field
function isfield(term::DTerm)
    return @match term begin
        Tensor(_, _, _, _) => true
        DExpr(Comp(_), (Tensor(_, _, _, _),)) => true
        _ => false
    end
end

zero_offsets(::Val{N}) where {N} = ntuple(_ -> 0 // 1, Val(N))

"""
    locate(expr, locs)
    locate(expr, locs...)

Place the samples of the lowered expression `expr` on the staggered grid, evaluated at the locations `locs`: the sample
`f[o₁, …, oₙ]` becomes the grid sample `f[l₁, …, lₙ][i₁, …, iₙ]`, where the locations and the integer grid index offsets
follow from the offsets and `locs`. Explicit locations of fields must match the locations of their samples, and sampled
coordinates become expressions of grid indices.
"""
function locate(term::DTerm, locs::NTuple{N, Location})::DTerm where {N}
    return @match term begin
        DExpr(Inds(offs), (arg,)) => locate_sample(arg, offs, locs)
        DExpr(Call(_), _) => walkargs(x -> locate(x, locs), term)
        _ => isuniform(term) ? term : unlowered_error()
    end
end
locate(::DTerm, ::Tuple{}) = locate_nolocs_error()
locate(term::DTerm, locs::Vararg{Location, N}) where {N} = locate(term, locs)

function locate_sample(arg::DTerm, shift, locs::NTuple{N, Location}) where {N}
    length(shift) == N || lower_dims_error()
    offs = ntuple(a -> shift[a], Val(N))
    return @match arg begin
        Coordinate(a) => coordinate_value(offs, a, locs)
        DExpr(Locs(explicit), (field,)) && if isfield(field) end => begin
            length(explicit) == N || locs_length_error()
            explicit == sample_locations(offs, locs) || locs_mismatch_error()
            grid_sample(field, offs, locs)
        end
        _ => isfield(arg) ? grid_sample(arg, offs, locs) : unlowered_error()
    end
end

loc_offset(::Point) = 0 // 1
loc_offset(::Segment) = 1 // 2

# position of the sample relative to the grid index of the evaluation point,
# point `i` is located at `i`, and segment `i` at `i + 1//2`
sample_position(off, loc) = loc_offset(loc) + off

sample_location(x::Rational{Int}) = isinteger(x) ? Point() : Segment()

sample_locations(offs, locs) = map(sample_location, map(sample_position, offs, locs))

# the grid sample `f[locs...][inds...]` of the field `f` sampled at `offs` from the evaluation point at `locs`
function grid_sample(field::DTerm, offs, locs)
    xs = map(sample_position, offs, locs)
    return makeinds_unchecked(makelocs_unchecked(field, map(sample_location, xs)), map(floor, xs))
end

# a free coordinate is the distance of the sample from the lower wall in grid units,
# point `1` lies on the lower wall, so point `i` is at `i - 1` and segment `i` at `i - 1//2`
function coordinate_value(offs::NTuple{N, Rational{Int}}, a, locs::NTuple{N, Location}) where {N}
    1 <= a <= N || coordinate_dims_error(a, N)
    return offset_term(Index(a), sample_position(offs[a], locs[a]) - 1)
end

# canonical form `term`, `term + x`, or `term - x` of `term` shifted by `x`, with `x` positive
function offset_term(term::DTerm, x::Rational{Int})
    iszero(x) && return term
    return x > 0 ? term + stencil_constant(x) : term - stencil_constant(-x)
end

@noinline lower_nodims_error() = throw(ArgumentError("lowering requires at least one dimension"))
@noinline locate_nolocs_error() = throw(ArgumentError("locating requires at least one grid location"))
@noinline lower_rank_error() = throw(ArgumentError("lower requires a scalar expression; select a scalar entry, e.g. expr[1, 2]"))
@noinline lower_dims_error() = throw(ArgumentError("samples must have as many offsets as there are dimensions"))
@noinline unsampled_error() = throw(ArgumentError("lowering rules must read terms through samples, e.g. `f[0, 0]`"))
@noinline unlowered_error() = throw(ArgumentError("only lowered expressions can be located, see `lower`"))
@noinline locs_field_error() = throw(ArgumentError("explicit locations can only be given to fields"))
@noinline locs_length_error() = throw(ArgumentError("explicit locations must have the same length as the evaluation locations"))
@noinline locs_mismatch_error() = throw(ArgumentError("explicit locations of a field must match the locations where the field is sampled"))
@noinline index_input_error() = throw(ArgumentError("grid indices are produced by `locate` and cannot appear in the lowered expression, use coordinates instead"))
@noinline function coordinate_dims_error(a, n)
    throw(ArgumentError(LazyString("coordinate `", Coordinate(a), "` exceeds the number of evaluation locations ", n)))
end
@noinline rule_rank_error(op) = throw(ArgumentError(LazyString("the lowering rule of '", op, "' must return a scalar")))
@noinline function no_rule_error(op, args, n)
    throw(ArgumentError(LazyString("operator '", op, "' has no lowering rule for ", length(args), " arguments in ", n, " dimensions")))
end
