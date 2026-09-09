@inline function makecall(op::Operator, args::NTuple{N,DTerm}) where {N}
    checkranks(op, args)
    makecall_unchecked(op, args)
end
makecall(op::Operator, args::Vararg{DTerm,N}) where {N} = makecall(op, args)
makecall_unchecked(op::Operator, args::NTuple{N,DTerm}) where {N} = DExpr(Call(op), args)

function makecomp(arg::DTerm, comp::NTuple{N,Int}) where {N}
    tensorrank(arg) == length(comp) || throw(ArgumentError("number of components must be equal to the rank of the tensor"))
    makecomp_unchecked(arg, comp)
end
makecomp(arg::DTerm, comp::Vararg{Int,N}) where {N} = makecomp(arg, comp)
makecomp_unchecked(arg::DTerm, comp::NTuple{N,Int}) where {N} = DExpr(Comp(comp), (arg,))

function makelocs(arg::DTerm, locs::NTuple{N,Location}) where {N}
    tensorrank(arg) == 0 || throw(ArgumentError("only scalars can be arguments to location expression"))
    makelocs_unchecked(arg, locs)
end
makelocs(arg::DTerm, locs::Vararg{Location,N}) where {N} = makelocs(arg, locs)
makelocs_unchecked(arg::DTerm, locs::NTuple{N,Location}) where {N} = DExpr(Locs(locs), (arg,))

function makeinds(arg::DTerm, inds::NTuple{N,DTerm}) where {N}
    tensorrank(arg) == 0 || throw(ArgumentError("only scalars can be arguments to indexing expression"))
    makeinds_unchecked(arg, inds)
end
makeinds(arg::DTerm, inds::Vararg{DTerm,N}) where {N} = makeinds(arg, inds)
makeinds_unchecked(arg::DTerm, inds::NTuple{N,DTerm}) where {N} = DExpr(Inds(), (arg, inds...))

# make operators callable
(op::Operator)(args::DTerm...) = makecall(op, args)

struct Fun{F} <: Operator
    f::F
    function Fun(f)
        if Base.issingletontype(typeof(f))
            return new{typeof(f)}(f)
        else
            throw(ArgumentError("function must be a singleton type"))
        end
    end
end

function ⊡ end
function ⊗ end

function sym end
function asym end

function adj end

function gram end
function cogram end

# unless specified otherwise, we assume all operators are scalar
tensorrank(op::Operator, args) = 0

tensorrank(op::Fun, args) = tensorrank(op.f, args)

# fallback tensorrank definition, assuming scalar function by default
tensorrank(::Function, args) = 0

tensorrank(::typeof(+), args)  = tensorrank(args[1])
tensorrank(::typeof(-), args)  = tensorrank(args[1])
tensorrank(::typeof(*), args)  = maximum(tensorrank, args)
tensorrank(::typeof(/), args)  = tensorrank(args[1])
tensorrank(::typeof(//), args) = tensorrank(args[1])
tensorrank(::typeof(÷), args)  = tensorrank(args[1])
tensorrank(::typeof(⋅), args)  = tensorrank(args[1]) + tensorrank(args[2]) - 2
tensorrank(::typeof(×), args)  = 1
tensorrank(::typeof(⊡), args)  = tensorrank(args[1]) + tensorrank(args[2]) - 4
tensorrank(::typeof(⊗), args)  = tensorrank(args[1]) + tensorrank(args[2])

tensorrank(::typeof(sym), args)     = tensorrank(args[1])
tensorrank(::typeof(asym), args)    = tensorrank(args[1])
tensorrank(::typeof(adj), args)     = tensorrank(args[1])
tensorrank(::typeof(inv), args)     = tensorrank(args[1])
tensorrank(::typeof(adjoint), args) = tensorrank(args[1])
tensorrank(::typeof(gram), args)    = 2
tensorrank(::typeof(cogram), args)  = 2
tensorrank(::typeof(diag), args)    = 1

@inline function checkscalar(op, args)
    if any(x -> tensorrank(x) > 0, args)
        throw(ArgumentError(LazyString("all arguments to '", op, "' must be scalars")))
    end
end

checkranks(op::Fun, args::NTuple{N,DTerm}) where {N} = checkranks(op.f, args)

# fallback implementation for all functions and operators, accepting only scalar arguments
checkranks(op::Operator, args::NTuple{N,DTerm}) where {N} = checkscalar(op, args)
checkranks(f::Function, args::NTuple{N,DTerm}) where {N} = checkscalar(f, args)

for op in (:+, :-)
    @eval function checkranks(::typeof($op), args::NTuple{N,DTerm}) where {N}
        allequal(tensorrank, args) || throw(ArgumentError(LazyString("all tensor arguments to '", $op, "' must have the same rank")))
    end
end
function checkranks(::typeof(*), args::NTuple{N,DTerm}) where {N}
    if count(x -> tensorrank(x) > 0, args) > 1
        throw(ArgumentError("""
                            at most one tensor argument with rank > 0 allowed in multiplication, \
                            consider using '⋅' for single contraction, '⊡' for double contraction, or '⊗' for outer product, \
                            or use broadcasting to perform element-wise multiplication between tensors
                            """))
    end
end
for op in (:/, ://, :÷)
    @eval function checkranks(::typeof($op), args::NTuple{N,DTerm}) where {N}
        if tensorrank(args[2]) > 0
            throw(ArgumentError(LazyString("denominator in '", $op, "' must be a scalar term")))
        end
    end
end
for op in (:adjoint, :adj, :det, :inv, :tr, :diag, :sym, :asym, :gram, :cogram)
    @eval function checkranks(::typeof($op), args::NTuple{N,DTerm}) where {N}
        if tensorrank(args[1]) != 2 && tensorrank(args[1]) != 0
            throw(ArgumentError(LazyString("'", $op, "' can only be applied to scalars or rank-2 tensors")))
        end
    end
end
function checkranks(::typeof(×), args::NTuple{N,DTerm}) where {N}
    if tensorrank(args[1]) != 1 || tensorrank(args[2]) != 1
        throw(ArgumentError("cross product '×' can only be applied to rank-1 tensors (vectors)"))
    end
end
function checkranks(::typeof(⋅), args::NTuple{N,DTerm}) where {N}
    if tensorrank(args[1]) < 1 || tensorrank(args[2]) < 1
        throw(ArgumentError("dot product '⋅' requires both arguments to have rank at least 1"))
    end
end
function checkranks(::typeof(⊡), args::NTuple{N,DTerm}) where {N}
    if tensorrank(args[1]) < 2 || tensorrank(args[2]) < 2
        throw(ArgumentError("double tensor contraction '⊡' requires both arguments to have rank at least 2"))
    end
end
function checkranks(::typeof(⊗), args::NTuple{N,DTerm}) where {N}
    if tensorrank(args[1]) == 0 || tensorrank(args[2]) == 0
        throw(ArgumentError("outer product '⊗' cannot be applied to scalar terms"))
    end
end
# TODO: broadcasting

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

# mixed symbolic-numeric binary operations
for op in (:+, :-, :*, :max, :min, :/, ://, :÷, :^, :<, :<=, :>, :>=, :(==), :!=, :&, :|, :xor)
    @eval begin
        Base.$op(a::Number, b::DTerm) = $op(Literal(a), b)
        Base.$op(a::DTerm, b::Number) = $op(a, Literal(b))
    end
end

function Base.adjoint(t::DTerm)
    tensorrank(t) == 0 && return t
    return makecall(Fun(adjoint), t)
end

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
    @eval function $op(t::DTerm)
        tensorrank(t) == 0 && return t
        return makecall(Fun($op), t)
    end
end

Base.inv(t::DTerm) = makecall(Fun(inv), t)
×(a::DTerm, b::DTerm) = makecall(Fun(×), a, b)

"""
    adj(t)

The adjugate of a second-rank tensor, defined as the transpose of the cofactor matrix.
"""
function adj(t::DTerm)
    tensorrank(t) == 0 && return t
    return makecall(Fun(adj), t)
end

"""
    sym(t)

The symmetric part of a second-rank tensor, defined as `(t + t') / 2`.
"""
function sym(t::DTerm)
    tensorrank(t) == 0 && return t
    return makecall(Fun(sym), t)
end

"""
    asym(t)

The antisymmetric part of a second-rank tensor, defined as `(t - t') / 2`.
"""
function asym(t::DTerm)
    tensorrank(t) == 0 && return Literal(0)
    return makecall(Fun(asym), t)
end

for op in (:gram, :cogram)
    @eval function $op(t::DTerm)
        tensorrank(t) == 0 && return t^2
        return makecall(Fun($op), t)
    end
end

# scalar unary operations
isunaryminus(expr::DTerm) = isunary(expr) && operation(expr) === (-)

Base.:-(arg::DTerm) = makecall_unchecked(Fun(-), (arg,))

for op in (:sqrt, :abs,
           :sin, :cos, :tan,
           :asin, :acos, :atan,
           :sind, :cosd, :tand,
           :asind, :acosd, :atand,
           :sinh, :cosh, :tanh,
           :asinh, :acosh, :atanh,
           :sinpi, :cospi, :tanpi,
           :cis, :cispi,
           :log, :log1p, :log2, :log10,
           :exp, :expm1, :exp2, :exp10)
    @eval Base.$op(arg::DTerm) = makecall(Fun($op), arg)
end

# # overloading broadcasting
# function Base.Broadcast.broadcasted(f, args::Vararg{STerm})
#     return @makeop(:broadcasted, SFun(f), args...)
# end
# function Base.Broadcast.broadcasted(::typeof(Base.literal_pow), f, t::STerm, n::Val{N}) where {N}
#     return @makeop(:broadcasted, SRef(:^), t, SLiteral(N))
# end
# Base.Broadcast.broadcasted(f, a::STerm, b::Number) = Base.Broadcast.broadcasted(f, a, SLiteral(b))
# Base.Broadcast.broadcasted(f, a::Number, b::STerm) = Base.Broadcast.broadcasted(f, SLiteral(a), b)

Base.ifelse(cond::DTerm, x::DTerm, y::DTerm) = makecall(Fun(ifelse), cond, x, y)
Base.ifelse(cond::DTerm, x::Number, y::DTerm) = ifelse(cond, Literal(x), y)
Base.ifelse(cond::DTerm, x::DTerm, y::Number) = ifelse(cond, x, Literal(y))
Base.ifelse(cond::DTerm, x::Number, y::Number) = ifelse(cond, Literal(x), Literal(y))å

# location
Base.getindex(arg::DTerm, locs::Vararg{Location,N}) where {N} = makelocs(arg, locs...)

# indexing
Base.getindex(arg::DTerm, inds::Vararg{DTerm,N}) where {N} = makeinds(arg, inds...)
