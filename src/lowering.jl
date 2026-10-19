struct LowerContext{N}
    locs::NTuple{N, Location}
    offs::NTuple{N, Rational{Int}}
    blck::Block{N}
end

"""
    stencil_rule(op, args[, ctx])

Distribute a stencil operation over `args` with a lowering context `ctx`, sampling the arguments at offsets
relative to the evaluation point, e.g. `f[1//2, 0]`. Context can be omitted for 1D rules.
"""
function stencil_rule end

stencil_rule(op::Operator, args, ctx::LowerContext) = _stencil_rule(op, args, ctx)

_stencil_rule(op, args::Tuple{DTerm}, ::LowerContext{1}) = stencil_rule(op, only(args))

# a pointwise function reads its arguments at the evaluation point
function stencil_rule(op::Fun, args::NTuple{M, DTerm}, ::LowerContext{N}) where {M, N}
    return makecall_unchecked(op, ntuple(k -> makeinds(args[k], zero_offsets(Val(N))), Val(M)))
end

"""
    lower(expr, locs)
    lower(expr, locs...)

Lower `expr` into a stencil form evaluated at the staggered grid locations `locs`. Tensor operations
in `expr` are first expanded with [`components`](@ref) in `length(locs)` dimensions. Each non-uniform
field is converted to a grid sample `f[locs...][offsets...]`, where the locations and the integer grid index
offsets are inferred from the offsets at which the field is sampled relative to the evaluation point.
"""
function lower(expr::DTerm, locs::NTuple{N, Location}) where {N}
    ctx = LowerContext(locs, zero_offsets(Val(N)), ntuple(_ -> Far(), Val(N)))
    return lower(expr, ctx)
end
lower(::DTerm, ::Tuple{}) = lower_nolocs_error()
lower(expr::DTerm, locs::Vararg{Location, N}) where {N} = lower(expr, locs)

function lower(term::DTerm, ctx::LowerContext{N}) where {N}
    return lower_at(scalar_components(term, N), ctx)
end

@noinline lower_nolocs_error() = throw(ArgumentError("lowering requires at least one grid location"))
@noinline locs_length_error() = throw(ArgumentError("explicit locations must have the same length as the evaluation locations"))
@noinline locs_mismatch_error() = throw(ArgumentError("explicit locations of a field must match the locations where the field is sampled"))
@noinline lower_rank_error() = throw(ArgumentError("lower requires a scalar expression; select a scalar entry, e.g. expr[1, 2]"))
@noinline lower_dims_error() = throw(ArgumentError("grid samples must have the same number of offsets as the evaluation locations"))
@noinline index_input_error() = throw(ArgumentError("grid indices are produced by lowering and cannot appear in the lowered expression, use coordinates instead"))
@noinline function coordinate_dims_error(a, n)
    throw(ArgumentError(LazyString("coordinate `", Coordinate(a), "` exceeds the number of evaluation locations ", n)))
end

function scalar_components(expr::DTerm, dims::Integer)
    tensorrank(expr) == 0 || lower_rank_error()
    return components(expr, dims)::DTerm
end

zero_offsets(::Val{N}) where {N} = ntuple(_ -> 0 // 1, Val(N))

# lower `expr` evaluated at `offsets` from the evaluation point at `locs`
#
# operators are expanded with stencil rules, which return samples at offsets relative to the point
# where the operator is evaluated
function lower_at(term::DTerm, ctx::LowerContext{N}) where {N}
    return @match term begin
        DExpr(Call(op), args) => lower_result(stencil_rule(op, args, ctx)::DTerm, ctx)
        DExpr(Locs(explicit), (arg,)) => begin
            isuniform(arg) && return arg
            length(explicit) == N || locs_length_error()
            explicit == sample_locations(ctx.offs, ctx.locs) || locs_mismatch_error()
            lower_at(arg, ctx)
        end
        DExpr(Inds(offs), (arg,)) => lower_sample(arg, offs, ctx)
        Coordinate(a) => coordinate_value(ctx.offs, a, ctx.locs)
        Index(_) => index_input_error()
        _ => isuniform(term) ? term : sample(term, ctx)
    end
end

# the result of a stencil rule combines samples and uniform terms with calls, which are kept as they are,
# while the samples are lowered, expanding the operators inside them
function lower_result(term::DTerm, ctx::LowerContext)::DTerm
    return @match term begin
        DExpr(Call(op), args) => lower_args(op, args, ctx)
        _ => lower_at(term, ctx)
    end
end

# specialize on arity to lower the arguments without a dynamic tuple map
function lower_args(op, args::NTuple{M, DTerm}, ctx)::DTerm where {M}
    return makecall_unchecked(op, ntuple(k -> lower_result(args[k], ctx), Val(M)))
end

# the offsets of a nested sample are relative to the offsets where the sample is evaluated
function lower_sample(arg, shift, ctx::LowerContext{N})::DTerm where {N}
    length(shift) == N || lower_dims_error()
    offs = ntuple(a -> ctx.offs[a] + shift[a], Val(N))
    return lower_at(arg, LowerContext(ctx.locs, offs, ctx.blck))
end

loc_offset(::Point) = 0 // 1
loc_offset(::Segment) = 1 // 2

# position of the sample relative to the grid index of the evaluation point,
# point `i` is located at `i`, and segment `i` at `i + 1//2`
sample_position(off, loc) = loc_offset(loc) + off

sample_location(x::Rational{Int}) = isinteger(x) ? Point() : Segment()

sample_locations(offs, locs) = map(sample_location, map(sample_position, offs, locs))

# convert a field sampled at `offsets` to the grid sample `f[locs...][inds...]`
function sample(field::DTerm, ctx::LowerContext{N}) where {N}
    xs = map(sample_position, ctx.offs, ctx.locs)
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
