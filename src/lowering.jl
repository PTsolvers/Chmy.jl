"""
    stencil_rule(op, args, dims)

Distribute a stencil operation over `args` in `dims = Val(N)` dimensions, sampling the arguments at offsets
relative to the evaluation point, e.g. `f[1//2, 0]`.
"""
function stencil_rule end

stencil_rule(op::Operator, args, dims::Val) = _stencil_rule(op, args, dims)

_stencil_rule(op, args::Tuple{DTerm}, ::Val{1}) = stencil_rule(op, only(args))

# a pointwise function is its own stencil
stencil_rule(op::Fun, args::Tuple{Vararg{DTerm}}, ::Val) = makecall_unchecked(op, args)

"""
    lower(expr, locs)
    lower(expr, locs...)

Lower `expr` into a stencil form evaluated at the staggered grid locations `locs`. Tensor operations
in `expr` are first expanded with [`components`](@ref) in `length(locs)` dimensions. Each non-uniform
field is converted to a grid sample `f[locs...][offsets...]`, where the locations and the integer grid index
offsets are inferred from the offsets at which the field is sampled relative to the evaluation point.
"""
function lower(expr::DTerm, locs::NTuple{N, Location}) where {N}
    return lower_at(scalar_components(expr, N), zero_offsets(Val(N)), locs)
end
lower(::DTerm, ::Tuple{}) = lower_nolocs_error()
lower(expr::DTerm, locs::Vararg{Location, N}) where {N} = lower(expr, locs)

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
# functions are applied pointwise, other operators are expanded with stencil rules, which return
# samples at offsets relative to the point where the operator is evaluated
function lower_at(expr::DTerm, offsets::NTuple{N, Rational{Int}}, locs::NTuple{N, Location})::DTerm where {N}
    return @match expr begin
        DExpr(Call(op), args) => begin
            if op isa Fun
                lower_args(op, args, offsets, locs)
            else
                lower_at(stencil_rule(op, args, Val(N))::DTerm, offsets, locs)
            end
        end
        DExpr(Locs(explicit), (arg,)) => begin
            isuniform(arg) && return arg
            length(explicit) == N || locs_length_error()
            explicit == sample_locations(offsets, locs) || locs_mismatch_error()
            lower_at(arg, offsets, locs)
        end
        DExpr(Inds(shift), (arg,)) => lower_sample(arg, shift, offsets, locs)
        Coordinate(a) => coordinate_value(offsets, a, locs)
        Index(_) => index_input_error()
        _ => isuniform(expr) ? expr : sample(expr, offsets, locs)
    end
end

# specialize on arity to lower the arguments without a dynamic tuple map
function lower_args(op, args::NTuple{M, DTerm}, offsets, locs)::DTerm where {M}
    return makecall_unchecked(op, ntuple(k -> lower_at(args[k], offsets, locs), Val(M)))
end

# the offsets of a nested sample are relative to the offsets where the sample is evaluated
function lower_sample(arg, shift, offsets::NTuple{N, Rational{Int}}, locs)::DTerm where {N}
    length(shift) == N || lower_dims_error()
    return lower_at(arg, ntuple(a -> offsets[a] + shift[a], Val(N)), locs)
end

loc_offset(::Point) = 0 // 1
loc_offset(::Segment) = 1 // 2

# position of the sample relative to the grid index of the evaluation point,
# point `i` is located at `i`, and segment `i` at `i + 1//2`
sample_position(offset, loc) = loc_offset(loc) + offset

sample_location(x::Rational{Int}) = isinteger(x) ? Point() : Segment()

sample_locations(offsets, locs) = map(sample_location, map(sample_position, offsets, locs))

# convert a field sampled at `offsets` to the grid sample `f[locs...][inds...]`
function sample(field::DTerm, offsets::NTuple{N, Rational{Int}}, locs::NTuple{N, Location}) where {N}
    xs = map(sample_position, offsets, locs)
    return makeinds_unchecked(makelocs_unchecked(field, map(sample_location, xs)), map(floor, xs))
end

# a free coordinate is the position of the evaluation point in grid index units
function coordinate_value(offsets::NTuple{N, Rational{Int}}, a, locs::NTuple{N, Location}) where {N}
    1 <= a <= N || coordinate_dims_error(a, N)
    return offset_term(Index(a), sample_position(offsets[a], locs[a]))
end

# canonical form `term`, `term + x`, or `term - x` of `term` shifted by `x`, with `x` positive
function offset_term(term::DTerm, x::Rational{Int})
    iszero(x) && return term
    return x > 0 ? term + stencil_constant(x) : term - stencil_constant(-x)
end
