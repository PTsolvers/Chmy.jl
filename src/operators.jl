# the canonical call `op(args...)` of canonical arguments: calls with all tensor arguments in component form are
# evaluated in their number of dimensions, and the other calls with the rules of the operator; functions of literals
# without a rule are evaluated numerically
function makecall(op::Operator, args::NTuple{N, DTerm})::DTerm where {N}
    checkranks(op, args)
    dims = packed_dims(args)
    if !isnothing(dims)
        result = evaluate(op, args, dims)::Union{Nothing, DTerm}
        isnothing(result) || return result
    end
    result = evaluate(op, args)::Union{Nothing, DTerm}
    isnothing(result) || return result
    return something(fold_literals(op, args), makecall_unchecked(op, args))
end
makecall(op::Operator, args::Vararg{DTerm, N}) where {N} = makecall(op, args)
makecall_unchecked(op::Operator, args::NTuple{N, DTerm}) where {N} = DExpr(Call(op), args)

function makecomp(arg::DTerm, comp::NTuple{N, Int}) where {N}
    tensorrank(arg) == length(comp) || throw(ArgumentError("number of components must be equal to the rank of the tensor"))
    return makecomp_unchecked(arg, comp)
end
makecomp(arg::DTerm, comp::Vararg{Int, N}) where {N} = makecomp(arg, comp)
makecomp_unchecked(arg::DTerm, comp::NTuple{N, Int}) where {N} = DExpr(Comp(comp), (arg,))

# the sample of the scalar `arg` at the positions, lowered in component form in as many dimensions as there are positions
function makeinds(arg::DTerm, positions::NTuple{N, PositionLike}) where {N}
    tensorrank(arg) == 0 || throw(ArgumentError("only scalars can be arguments to indexing expression"))
    return sample(components(arg, N), map(asposition, positions))
end

asposition(p::Position) = p
asposition(p::Global) = Trace(p)
asposition((frame, offset)::Tuple{Global, Union{Integer, Rational}}) = Coord(frame, offset)

makeinds_unchecked(arg::DTerm, positions::NTuple{N, Position}) where {N} = DExpr(Inds(positions), (arg,))

function makepack(rank::Integer, dims::Integer, kind::TensorKind, data::Tuple{Vararg{DTerm}})
    rank > 0 || throw(ArgumentError("tensors in component form must have a positive rank"))
    dims > 0 || throw(ArgumentError("tensors in component form must have a positive number of dimensions"))
    if length(data) != ncomponents(kind, dims, rank)
        throw(ArgumentError("number of entries must match the number of stored components of the tensor"))
    end
    any(x -> tensorrank(x) > 0, data) && throw(ArgumentError("entries of a tensor in component form must be scalars"))
    return canonical_pack(rank, dims, kind, data)
end
makepack_unchecked(rank::Integer, dims::Integer, kind::TensorKind, data::Tuple{Vararg{DTerm}}) = DExpr(Pack(rank, dims, kind), data)

# the canonical term with the head `head` and the children `args`
function maketerm(head::Head, args::Tuple{Vararg{DTerm}})::DTerm
    return @match head begin
        Call(op) => makecall(op, args)
        Comp(comps) => component(only(args), comps)
        Inds(positions) => makeinds(only(args), positions)
        Pack(rank, dims, kind) => makepack(rank, dims, kind, args)
    end
end

# the number of dimensions of the arguments if all tensor arguments are in component form, and `nothing` otherwise
function packed_dims(args)
    dims = nothing
    for x in args
        tensorrank(x) == 0 && continue
        ispack(x) || return nothing
        d = packdims(x)
        if !isnothing(dims) && dims != d
            throw(DimensionMismatch("tensors in component form must have the same number of dimensions"))
        end
        dims = d
    end
    return dims
end

BroadcastedFun(op::Fun) = BroadcastedFun(op.f)
Fun(op::BroadcastedFun) = Fun(op.f)

function ⊡ end
function ⊗ end

function sym end
function asym end

function cof end

function gram end
function cogram end

# unless specified otherwise, we assume all operators are scalar
tensorrank(op::Operator, args) = 0

tensorrank(op::Fun, args) = tensorrank(op.f, args)
tensorrank(::BroadcastedFun, args) = maximum(tensorrank, args)

# fallback tensorrank definition, assuming scalar function by default
tensorrank(::Function, args) = 0

tensorrank(::typeof(+), args) = tensorrank(args[1])
tensorrank(::typeof(-), args) = tensorrank(args[1])
tensorrank(::typeof(*), args) = maximum(tensorrank, args)
tensorrank(::typeof(/), args) = tensorrank(args[1])
tensorrank(::typeof(//), args) = tensorrank(args[1])
tensorrank(::typeof(÷), args) = tensorrank(args[1])
tensorrank(::typeof(⋅), args) = tensorrank(args[1]) + tensorrank(args[2]) - 2
tensorrank(::typeof(×), args) = 1
tensorrank(::typeof(⊡), args) = tensorrank(args[1]) + tensorrank(args[2]) - 4
tensorrank(::typeof(⊗), args) = tensorrank(args[1]) + tensorrank(args[2])

tensorrank(::typeof(sym), args) = tensorrank(args[1])
tensorrank(::typeof(asym), args) = tensorrank(args[1])
tensorrank(::typeof(cof), args) = tensorrank(args[1])
tensorrank(::typeof(inv), args) = tensorrank(args[1])
tensorrank(::typeof(adjoint), args) = tensorrank(args[1])
tensorrank(::typeof(gram), args) = 2
tensorrank(::typeof(cogram), args) = 2
tensorrank(::typeof(diag), args) = 1

tensorrank(op::Union{ZeroTensor, IdTensor}, args) = op.rank

@inline function checkscalar(op, args)
    if any(x -> tensorrank(x) > 0, args)
        throw(ArgumentError(LazyString("all arguments to '", op, "' must be scalars")))
    end
    return
end

checkranks(op::Fun, args::NTuple{N, DTerm}) where {N} = checkranks(op.f, args)

function check_broadcast_ranks(op::BroadcastedFun, args::NTuple{N, DTerm}) where {N}
    rank = tensorrank(op, args)
    if any(x -> tensorrank(x) > 0 && tensorrank(x) != rank, args)
        throw(ArgumentError("all non-scalar broadcast arguments must have the same tensor rank"))
    end
    return rank
end
checkranks(op::BroadcastedFun, args::NTuple{N, DTerm}) where {N} = check_broadcast_ranks(op, args)

# fallback implementation for all functions and operators, accepting only scalar arguments
checkranks(op::Operator, args::NTuple{N, DTerm}) where {N} = checkscalar(op, args)
checkranks(f::Function, args::NTuple{N, DTerm}) where {N} = checkscalar(f, args)

for op in (:+, :-)
    @eval function checkranks(::typeof($op), args::NTuple{N, DTerm}) where {N}
        return allequal(tensorrank, args) || throw(ArgumentError(LazyString("all tensor arguments to '", $op, "' must have the same rank")))
    end
end
function checkranks(::typeof(*), args::NTuple{N, DTerm}) where {N}
    if count(x -> tensorrank(x) > 0, args) > 1
        throw(
            ArgumentError(
                """
                at most one tensor argument with rank > 0 allowed in multiplication, \
                consider using '⋅' for single contraction, '⊡' for double contraction, or '⊗' for outer product, \
                or use broadcasting to perform element-wise multiplication between tensors
                """
            )
        )
    end
    return
end
for op in (:/, ://, :÷)
    @eval function checkranks(::typeof($op), args::NTuple{N, DTerm}) where {N}
        if tensorrank(args[2]) > 0
            throw(ArgumentError(LazyString("denominator in '", $op, "' must be a scalar term")))
        end
        return
    end
end
for op in (:adjoint, :cof, :det, :inv, :tr, :diag, :sym, :asym, :gram, :cogram)
    @eval function checkranks(::typeof($op), args::NTuple{N, DTerm}) where {N}
        if tensorrank(args[1]) != 2 && tensorrank(args[1]) != 0
            throw(ArgumentError(LazyString("'", $op, "' can only be applied to scalars or rank-2 tensors")))
        end
        return
    end
end
function checkranks(::typeof(×), args::NTuple{N, DTerm}) where {N}
    if tensorrank(args[1]) != 1 || tensorrank(args[2]) != 1
        throw(ArgumentError("cross product '×' can only be applied to rank-1 tensors (vectors)"))
    end
    return
end
function checkranks(::typeof(⋅), args::NTuple{N, DTerm}) where {N}
    if tensorrank(args[1]) < 1 || tensorrank(args[2]) < 1
        throw(ArgumentError("dot product '⋅' requires both arguments to have rank at least 1"))
    end
    return
end
function checkranks(::typeof(⊡), args::NTuple{N, DTerm}) where {N}
    if tensorrank(args[1]) < 2 || tensorrank(args[2]) < 2
        throw(ArgumentError("double tensor contraction '⊡' requires both arguments to have rank at least 2"))
    end
    return
end
function checkranks(::typeof(⊗), args::NTuple{N, DTerm}) where {N}
    if tensorrank(args[1]) == 0 || tensorrank(args[2]) == 0
        throw(ArgumentError("outer product '⊗' cannot be applied to scalar terms"))
    end
    return
end

# shortcut for unary plus
Base.:+(x::DTerm) = x
Base.:*(x::DTerm) = x

# multiary operators
for op in (:+, :*, :max, :min)
    @eval Base.$op(args::Vararg{DTerm}) = makecall(Fun($op), args...)
end

# binary operators
for op in (:-, :/, ://, :÷, :^, :<, :<=, :>, :>=, :(==), :!=, :&, :|, :xor)
    @eval Base.$op(a::DTerm, b::DTerm) = makecall(Fun($op), a, b)
end

# values that are converted to terms in mixed binary operations: numbers, and the wall vectors
const TermLike = Union{Number, AnyNormal, AnyTangent}

# make operators callable, with numbers and wall vectors as arguments
(op::Operator)(args::Union{DTerm, TermLike}...) = makecall(op, map(asterm, args))

# mixed binary operations of terms and term-like values
for op in (:+, :-, :*, :max, :min, :/, ://, :÷, :^, :<, :<=, :>, :>=, :(==), :!=, :&, :|, :xor)
    @eval begin
        Base.$op(a::TermLike, b::DTerm) = $op(asterm(a), b)
        Base.$op(a::DTerm, b::TermLike) = $op(a, asterm(b))
    end
end

Base.adjoint(t::DTerm) = makecall(Fun(adjoint), t)

"""
    ⋅(a::DTerm, b::DTerm)

A single contraction operator, which contracts the last index of `a` with the first index of `b`.
"""
⋅(a::DTerm, b::DTerm) = makecall(Fun(⋅), a, b)

"""
    ⊡(a, b)

The double contraction operator, which contracts the last two indices of `a` with the first two indices of `b`.
"""
⊡(a::DTerm, b::DTerm) = makecall(Fun(⊡), a, b)

"""
    ⊗(a, b)

The outer product operator, which creates a tensor by combining all indices of `a` with all indices of `b`.
"""
⊗(a::DTerm, b::DTerm) = makecall(Fun(⊗), a, b)

# methods from LinearAlgebra
transpose(t::DTerm) = t'

for op in (:det, :tr, :diag)
    @eval $op(t::DTerm) = makecall(Fun($op), t)
end

Base.inv(t::DTerm) = makecall(Fun(inv), t)
×(a::DTerm, b::DTerm) = makecall(Fun(×), a, b)

for op in (:⋅, :⊡, :⊗, :×)
    @eval begin
        $op(a::TermLike, b::DTerm) = $op(asterm(a), b)
        $op(a::DTerm, b::TermLike) = $op(a, asterm(b))
    end
end

"""
    cof(t)

The cofactor matrix of a second-rank tensor, satisfying `cof(t) = det(t) * inv(t)'`.
"""
cof(t::DTerm) = makecall(Fun(cof), t)

"""
    adj(t)

The adjugate of a second-rank tensor, defined as the transpose of the cofactor matrix.
"""
adj(t::DTerm) = cof(t)'

"""
    sym(t)

The symmetric part of a second-rank tensor, defined as `(t + t') / 2`.
"""
sym(t::DTerm) = makecall(Fun(sym), t)

"""
    asym(t)

The antisymmetric part of a second-rank tensor, defined as `(t - t') / 2`.
"""
asym(t::DTerm) = makecall(Fun(asym), t)

for op in (:gram, :cogram)
    @eval $op(t::DTerm) = makecall(Fun($op), t)
end

# scalar unary operations
Base.:-(arg::DTerm) = makecall(Fun(-), (arg,))

for op in (
        :sqrt, :abs,
        :sin, :cos, :tan,
        :asin, :acos, :atan,
        :sind, :cosd, :tand,
        :asind, :acosd, :atand,
        :sinh, :cosh, :tanh,
        :asinh, :acosh, :atanh,
        :sinpi, :cospi, :tanpi,
        :cis, :cispi,
        :log, :log1p, :log2, :log10,
        :exp, :expm1, :exp2, :exp10,
    )
    @eval Base.$op(arg::DTerm) = makecall(Fun($op), arg)
end

broadcasted(f, arg::DTerm, args::DTerm...) = makecall(BroadcastedFun(f), (arg, args...))
broadcasted(f, a::DTerm, b::Number) = broadcasted(f, a, Literal(b))
broadcasted(f, a::Number, b::DTerm) = broadcasted(f, Literal(a), b)
function broadcasted(::typeof(Base.literal_pow), f, t::DTerm, ::Val{N}) where {N}
    return broadcasted(f, t, Literal(N))
end

Base.ifelse(cond::DTerm, x::DTerm, y::DTerm) = makecall(Fun(ifelse), cond, x, y)
Base.ifelse(cond::DTerm, x::Number, y::DTerm) = ifelse(cond, Literal(x), y)
Base.ifelse(cond::DTerm, x::DTerm, y::Number) = ifelse(cond, x, Literal(y))
Base.ifelse(cond::DTerm, x::Number, y::Number) = ifelse(cond, Literal(x), Literal(y))
