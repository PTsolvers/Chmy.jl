"""
    stencil_rule(op, args, coords)

Distribute a stencil operation over `args`, sampling the arguments at symbolic coordinates
relative to `coords`, e.g. `f[ξ¹ + 1//2, ξ²]`.
"""
function stencil_rule end

stencil_rule(op::Operator, args, coords) = _stencil_rule(op, args, coords)

function _stencil_rule(op, args::Tuple{DTerm}, coords::Tuple{DTerm})
    return stencil_rule(op, only(args), only(coords))
end

function stencil_rule(op::Fun, args::Tuple{Vararg{DTerm}}, coords::NTuple{N, DTerm}) where {N}
    return makecall_unchecked(op, map(x -> x[coords...], args))
end

"""
    lower(expr, locs)
    lower(expr, locs...)

Lower `expr` into a stencil form evaluated at the staggered grid locations `locs`. Tensor operations
in `expr` are first expanded with [`components`](@ref) in `length(locs)` dimensions. Each non-uniform
field is converted to a grid sample `f[locs...][inds...]`, where the locations and the grid indices are
inferred from the coordinates at which the field is sampled relative to the evaluation point.
"""
function lower(expr::DTerm, locs::NTuple{N, Location}) where {N}
    return lower_at(scalar_components(expr, N), coordinates(Val(N)), locs)
end
lower(::DTerm, ::Tuple{}) = lower_nolocs_error()
lower(expr::DTerm, locs::Vararg{Location, N}) where {N} = lower(expr, locs)

@noinline lower_nolocs_error() = throw(ArgumentError("lowering requires at least one grid location"))
@noinline locs_length_error() = throw(ArgumentError("explicit locations must have the same length as the evaluation locations"))
@noinline locs_mismatch_error() = throw(ArgumentError("explicit locations of a field must match the locations where the field is sampled"))
@noinline lower_rank_error() = throw(ArgumentError("lower requires a scalar expression; select a scalar entry, e.g. expr[1, 2]"))
@noinline lower_dims_error() = throw(ArgumentError("grid samples must have the same number of coordinates as the evaluation locations"))
@noinline index_input_error() = throw(ArgumentError("grid indices are produced by lowering and cannot appear in the lowered expression, use coordinates instead"))
@noinline function coordinate_dims_error(a, n)
    throw(ArgumentError(LazyString("coordinate `", Coordinate(a), "` exceeds the number of evaluation locations ", n)))
end
@noinline function coordinate_form_error(term, a)
    throw(ArgumentError(LazyString("expected a coordinate in the form `", Coordinate(a), " ± k` with `k` a multiple of 1//2, got `", term, "`")))
end

function scalar_components(expr::DTerm, dims::Integer)
    tensorrank(expr) == 0 || lower_rank_error()
    return components(expr, dims)::DTerm
end

coordinates(::Val{N}) where {N} = ntuple(Coordinate, Val(N))

# lower `expr` evaluated at `coords`, the coordinates relative to the evaluation point at `locs`
#
# functions are applied pointwise, other operators are expanded with stencil rules, which return
# samples at coordinates relative to the evaluation point, so the result is lowered at the evaluation point
function lower_at(expr::DTerm, coords::NTuple{N, DTerm}, locs::NTuple{N, Location})::DTerm where {N}
    return @match expr begin
        DExpr(Call(op), args) => begin
            if op isa Fun
                lower_args(op, args, coords, locs)
            else
                lower_at(stencil_rule(op, args, coords)::DTerm, coordinates(Val(N)), locs)
            end
        end
        DExpr(Locs(explicit), (arg,)) => begin
            isuniform(arg) && return arg
            length(explicit) == N || locs_length_error()
            explicit == sample_locations(coords, locs) || locs_mismatch_error()
            lower_at(arg, coords, locs)
        end
        DExpr(Inds(), args) => lower_sample(args, coords, locs)
        Coordinate(a) => coordinate_value(coords, a, locs)
        Index(_) => index_input_error()
        _ => isuniform(expr) ? expr : sample(expr, coords, locs)
    end
end

# specialize on arity to lower the arguments without a dynamic tuple map
function lower_args(op, args::NTuple{M, DTerm}, coords, locs)::DTerm where {M}
    return makecall_unchecked(op, ntuple(k -> lower_at(args[k], coords, locs), Val(M)))
end

# the coordinates of a nested sample are relative to the coordinates where the sample is evaluated,
# so the offsets of both are summed up and folded into the canonical form `ξᵃ ± k`
function lower_sample(args::NTuple{M, DTerm}, coords::NTuple{N, DTerm}, locs)::DTerm where {M, N}
    M - 1 == N || lower_dims_error()
    arg, scoords = first(args), Base.tail(args)
    ncoords = ntuple(Val(N)) do a
        offset_term(Coordinate(a), fold_coordinate(scoords[a], a) + fold_coordinate(coords[a], a))
    end
    return lower_at(arg, ncoords, locs)
end

loc_offset(::Point) = 0 // 1
loc_offset(::Segment) = 1 // 2

# position of the sample along axis `a` relative to the grid index of the evaluation point,
# point `i` is located at `i`, and segment `i` at `i + 1//2`
sample_position(coord, loc, a) = loc_offset(loc) + fold_coordinate(coord, a)

sample_location(x::Rational{Int}) = isinteger(x) ? Point() : Segment()

function sample_positions(coords::NTuple{N, DTerm}, locs::NTuple{N, Location}) where {N}
    return map(sample_position, coords, locs, ntuple(identity, Val(N)))
end

sample_locations(coords, locs) = map(sample_location, sample_positions(coords, locs))

# convert a field sampled at `coords` to the grid sample `f[locs...][inds...]`
function sample(field::DTerm, coords::NTuple{N, DTerm}, locs::NTuple{N, Location}) where {N}
    xs = sample_positions(coords, locs)
    inds = ntuple(a -> offset_term(Index(a), floor(xs[a])), Val(N))
    return makeinds_unchecked(makelocs_unchecked(field, map(sample_location, xs)), inds)
end

# a free coordinate is the position of the evaluation point in grid index units
function coordinate_value(coords::NTuple{N, DTerm}, a, locs::NTuple{N, Location}) where {N}
    1 <= a <= N || coordinate_dims_error(a, N)
    return offset_term(Index(a), sample_position(coords[a], locs[a], a))
end

# canonical form `term`, `term + x`, or `term - x` of `term` shifted by `x`, with `x` positive
function offset_term(term::DTerm, x::Rational{Int})
    iszero(x) && return term
    return x > 0 ? term + stencil_constant(x) : term - stencil_constant(-x)
end

# offset `k` of the coordinate expression `ξᵃ ± k`
function fold_coordinate(term::DTerm, a::Int)::Rational{Int}
    m, k = fold_affine(term, a)
    isone(m) || coordinate_form_error(term, a)
    return k
end

# coefficient of `ξᵃ` and the constant offset of an affine coordinate expression
function fold_affine(term::DTerm, a::Int)::Tuple{Int, Rational{Int}}
    return @match term begin
        Coordinate(b) => b == a ? (1, 0 // 1) : coordinate_form_error(term, a)
        Literal(val) => begin
            val isa Union{Integer, Rational} || coordinate_form_error(term, a)
            k = Rational{Int}(val)
            denominator(k) <= 2 || coordinate_form_error(term, a)
            (0, k)
        end
        DExpr(Call(op), args) => fold_call(op, args, term, a)
        _ => coordinate_form_error(term, a)
    end
end

function fold_call(op, args, term, a)::Tuple{Int, Rational{Int}}
    if op isa Fun{typeof(+)}
        m, k = 0, 0 // 1
        for arg in args
            ma, ka = fold_affine(arg, a)
            m, k = m + ma, k + ka
        end
        return m, k
    elseif op isa Fun{typeof(-)} && length(args) == 1
        m, k = fold_affine(only(args), a)
        return -m, -k
    elseif op isa Fun{typeof(-)} && length(args) == 2
        m1, k1 = fold_affine(args[1], a)
        m2, k2 = fold_affine(args[2], a)
        return m1 - m2, k1 - k2
    end
    return coordinate_form_error(term, a)
end
