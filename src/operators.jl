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

@inline function makecall(op::Fun, args::NTuple{N,DTerm}) where {N}
    checkranks(op, args)
    makecall_unchecked(op, args)
end
makecall(op::Fun, args::Vararg{DTerm,N}) where {N} = makecall(op, args)
makecall_unchecked(op::Fun, args::NTuple{N,DTerm}) where {N} = DExpr(Call(op), args)

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

function ⊡ end
function ⊗ end

function sym end
function asym end

function adj end

function gram end
function cogram end

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

@inline checkranks(op::Fun, args::NTuple{N,DTerm}) where {N} = checkranks(op.f, args)

# fallback implementation for all functions, accepting only scalar arguments
function checkranks(f::Function, args::NTuple{N,DTerm}) where {N}
    if any(x -> tensorrank(x) > 0, args)
        throw(ArgumentError(LazyString("all arguments to '", f, "' must be scalars")))
    end
end

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
    if tensorrank(args[1]) + tensorrank(args[2]) < 2
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
    ⊡(a, b)

The double contraction operator, which contracts the first two indices of `a` with the first two indices of `b`.
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

function Base.inv(t::DTerm)
    if iscall(t) && operation(t) === Fun(inv)
        return only(arguments(t))
    end
    return makecall(Fun(inv), t)
end

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

Base.:-(arg::DTerm) = makecall(Fun(-), arg)

Base.:-(::SLiteral{S}) where {S} = SLiteral{-S}()
Base.:-(arg::SLiteral{0}) = arg
Base.:-(arg::SZeroTensor) = arg

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
Base.ifelse(cond::DTerm, x::Number, y::Number) = ifelse(cond, Literal(x), Literal(y))

# tensor operations

# @generated function ⊡(t1::Tensor{D,2}, t2::Tensor{D,2}) where {D}
#     ex_p = Tuple(:(t1[$i, $j] * t2[$i, $j]) for i in 1:D, j in 1:D)
#     ex_c = Expr(:call, :+, ex_p...)
#     quote
#         @inline
#         return $ex_c
#     end
# end

# ⋅(v1::Vec{D}, v2::Vec{D}) where {D} = +(ntuple(i -> v1[i] * v2[i], Val(D))...)
# @generated function ⋅(t::Tensor{D,2}, v::Vec{D}) where {D}
#     vc(i) = Expr(:call, :+, Tuple(:(t[$i, $j] * v[$j]) for j in 1:D)...)
#     ex = Expr(:call, :(Vec{$D}), Tuple(vc(i) for i in 1:D)...)
#     quote
#         @inline
#         return $ex
#     end
# end
# @generated function ⋅(t1::Tensor{D,2}, t2::Tensor{D,2}) where {D}
#     vc(i, j) = Expr(:call, :+, Tuple(:(t1[$i, $k] * t2[$k, $j]) for k in 1:D)...)
#     ex = Expr(:call, :(Tensor{$D,2}), Tuple(vc(i, j) for i in 1:D, j in 1:D)...)
#     quote
#         @inline
#         return $ex
#     end
# end

# function ×(v1::Vec{3}, v2::Vec{3})
#     return Vec{3}(v1[2] * v2[3] - v1[3] * v2[2],
#                   v1[3] * v2[1] - v1[1] * v2[3],
#                   v1[1] * v2[2] - v1[2] * v2[1])
# end

# @generated function ⊗(v1::Vec{D}, v2::Vec{D}) where {D}
#     ex = Expr(:call, :(Tensor{$D,2}))
#     for j in 1:D, i in 1:D
#         push!(ex.args, :(v1[$i] * v2[$j]))
#     end
#     quote
#         @inline
#         return $ex
#     end
# end

# tr(t::Tensor{D,2}) where {D} = +(ntuple(i -> t[i, i], Val(D))...)

# Base.transpose(t::SymTensor{2}) = t
# Base.transpose(t::AltTensor{2}) = -t
# Base.transpose(t::DiagTensor{2}) = t
# @generated function Base.transpose(t::Tensor{D,2}) where {D}
#     ex = Expr(:call, :(Tensor{$D,2}), Tuple(:(t[$j, $i]) for i in 1:D, j in 1:D)...)
#     quote
#         @inline
#         return $ex
#     end
# end

# Base.adjoint(S::Tensor{D,2}) where {D} = transpose(S)

# det(t::Tensor{2,2}) = t[1, 1] * t[2, 2] - t[1, 2] * t[2, 1]
# det(t::SymTensor{2,2}) = t[1, 1] * t[2, 2] - t[1, 2]^2
# det(t::AltTensor{2,2}) = t[1, 2]^2
# function det(t::Tensor{3,2})
#     t[1, 1] * (t[2, 2] * t[3, 3] - t[2, 3] * t[3, 2]) +
#     t[1, 2] * (t[2, 3] * t[3, 1] - t[2, 1] * t[3, 3]) +
#     t[1, 3] * (t[2, 1] * t[3, 2] - t[2, 2] * t[3, 1])
# end

# det(::AltTensor{2,3}) = SLiteral(0)

# diag(t::Tensor{D,2}) where {D} = Vec(ntuple(i -> t[i, i], Val(D))...)

# adj(t::Tensor{2,2}) = Tensor{2,2}(t[2, 2], -t[2, 1], -t[1, 2], t[1, 1])
# adj(t::SymTensor{2,2}) = SymTensor{2,2}(t[2, 2], -t[1, 2], t[1, 1])
# adj(t::AltTensor{2,2}) = AltTensor{2,2}(-t[1, 2])
# function adj(t::Tensor{3,2})
#     c11 = t[2, 2] * t[3, 3] - t[2, 3] * t[3, 2]
#     c12 = t[2, 3] * t[3, 1] - t[2, 1] * t[3, 3]
#     c13 = t[2, 1] * t[3, 2] - t[2, 2] * t[3, 1]
#     c21 = t[1, 3] * t[3, 2] - t[1, 2] * t[3, 3]
#     c22 = t[1, 1] * t[3, 3] - t[1, 3] * t[3, 1]
#     c23 = t[1, 2] * t[3, 1] - t[1, 1] * t[3, 2]
#     c31 = t[1, 2] * t[2, 3] - t[1, 3] * t[2, 2]
#     c32 = t[1, 3] * t[2, 1] - t[1, 1] * t[2, 3]
#     c33 = t[1, 1] * t[2, 2] - t[1, 2] * t[2, 1]
#     return Tensor{3,2}(c11, c12, c13,
#                        c21, c22, c23,
#                        c31, c32, c33)
# end
# function adj(t::SymTensor{2,3})
#     c11 = t[2, 2] * t[3, 3] - t[2, 3]^2
#     c12 = t[2, 3] * t[1, 3] - t[1, 2] * t[3, 3]
#     c13 = t[1, 2] * t[2, 3] - t[2, 2] * t[1, 3]
#     c22 = t[1, 1] * t[3, 3] - t[1, 3]^2
#     c23 = t[1, 2] * t[1, 3] - t[1, 1] * t[2, 3]
#     c33 = t[1, 1] * t[2, 2] - t[1, 2]^2
#     return SymTensor{2,3}(c11,
#                           c12, c22,
#                           c13, c23, c33)
# end
# function adj(t::AltTensor{2,3})
#     c11 = t[2, 3]^2
#     c12 = -t[1, 3] * t[2, 3]
#     c13 = t[1, 2] * t[2, 3]
#     c22 = t[1, 3]^2
#     c23 = -t[1, 2] * t[1, 3]
#     c33 = t[1, 2]^2
#     return SymTensor{2,3}(c11,
#                           c12, c22,
#                           c13, c23, c33)
# end

# Base.inv(t::Tensor{D,2}) where {D} = adj(t) / det(t)

# sym(t::SymTensor) = t
# @generated function sym(t::Tensor{D,2}) where {D}
#     ex = Expr(:call, :(SymTensor{$D,2}))
#     foreach_nondecreasing(Val(D), Val(2)) do (i, j)
#         push!(ex.args, :(1 // 2 * (t[$i, $j] + t[$j, $i])))
#     end
#     quote
#         @inline
#         return $ex
#     end
# end

# @generated function asym(t::Tensor{D,2}) where {D}
#     ex = Expr(:call, :(AltTensor{$D,2}))
#     foreach_increasing(Val(D), Val(2)) do (i, j)
#         push!(ex.args, :(1 // 2 * (t[$i, $j] - t[$j, $i])))
#     end
#     quote
#         @inline
#         return $ex
#     end
# end

# """
#     gram(t)

# The Gramian of a second-rank tensor, defined as `t' ⋅ t`.
# """
# @generated function gram(t::Tensor{D,2}) where {D}
#     ex = Expr(:call, :(SymTensor{$D,2}))
#     foreach_nondecreasing(Val(D), Val(2)) do (i, j)
#         comp = Expr(:call, :+)
#         for k in 1:D
#             push!(comp.args, :(t[$i, $k] * t[$j, $k]))
#         end
#         push!(ex.args, comp)
#     end
#     quote
#         @inline
#         return $ex
#     end
# end

# """
#     cogram(t)

# The co-Gramian of a second-rank tensor, defined as `t ⋅ t'`.
# """
# @generated function cogram(t::Tensor{D,2}) where {D}
#     ex = Expr(:call, :(SymTensor{$D,2}))
#     foreach_nondecreasing(Val(D), Val(2)) do (i, j)
#         comp = Expr(:call, :+)
#         for k in 1:D
#             push!(comp.args, :(t[$k, $i] * t[$k, $j]))
#         end
#         push!(ex.args, comp)
#     end
#     quote
#         @inline
#         return $ex
#     end
# end

# # custom broadcasting for tensors
# @generated function Base.Broadcast.broadcasted(f::F, arg::Tensor{D,R}) where {F,D,R}
#     ex = Expr(:call, :(Tensor{$D,$R}))
#     for idx in CartesianIndices(ntuple(_ -> D, Val(R)))
#         I = Tuple(idx)
#         push!(ex.args, :(f(arg[$(I...)])))
#     end
#     quote
#         @inline
#         return $ex
#     end
# end

# TODO: perform checks as in a static version

# location
Base.getindex(arg::DTerm, locs::Vararg{Location,N}) where {N} = makelocs(arg, locs...)

# indexing
Base.getindex(arg::DTerm, inds::Vararg{DTerm,N}) where {N} = makeinds(arg, inds...)
