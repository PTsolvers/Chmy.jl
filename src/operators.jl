for op in (:+, :-)
    @eval function check_tensor_ranks(op::SRef{$(Meta.quot(op))}, args::Vararg{STerm})
        if !all(x -> tensorrank(x) == tensorrank(args[1]), args)
            throw(ArgumentError("all tensor arguments to '$op' must have the same rank"))
        end
    end
end
function check_tensor_ranks(::SRef{:*}, args::Vararg{STerm})
    if count(x -> tensorrank(x) > 0, args) > 1
        throw(ArgumentError("""
                            at most one tensor argument with rank > 0 allowed in multiplication, \
                            consider using '⋅' for single contraction, '⊡' for double contraction, or '⊗' for outer product, \
                            or use broadcasting to perform element-wise multiplication between tensors
                            """))
    end
end
for op in (:/, ://, :÷)
    @eval function check_tensor_ranks(op::SRef{$(Meta.quot(op))}, ::STerm, b::STerm)
        if tensorrank(b) > 0
            throw(ArgumentError("denominator in '$op' must be a scalar term"))
        end
    end
end
for op in (:adjoint, :adj, :det, :inv, :tr, :diag, :sym, :asym, :gram, :cogram)
    @eval function check_tensor_ranks(op::SRef{$(Meta.quot(op))}, t::STerm)
        if tensorrank(t) != 2 && tensorrank(t) != 0
            throw(ArgumentError("'$op' can only be applied to scalars or rank-2 tensors"))
        end
    end
end
function check_tensor_ranks(::SRef{:×}, a::STerm, b::STerm)
    if tensorrank(a) != 1 || tensorrank(b) != 1
        throw(ArgumentError("cross product '×' can only be applied to rank-1 tensors (vectors)"))
    end
end
function check_tensor_ranks(::SRef{:⋅}, a::STerm, b::STerm)
    if tensorrank(a) + tensorrank(b) < 2
        throw(ArgumentError("dot product '⋅' requires both arguments to have rank at least 1"))
    end
end
function check_tensor_ranks(::SRef{:⊡}, a::STerm, b::STerm)
    if tensorrank(a) < 2 || tensorrank(b) < 2
        throw(ArgumentError("double tensor contraction '⊡' requires both arguments to have rank at least 2"))
    end
end
function check_tensor_ranks(::SRef{:⊗}, a::STerm, b::STerm)
    if tensorrank(a) == 0 || tensorrank(b) == 0
        throw(ArgumentError("outer product '⊗' cannot be applied to scalar terms"))
    end
end
function check_tensor_ranks(::SRef{:broadcasted}, op::STerm, args::Vararg{STerm})
    maxrank = maximum(tensorrank, args)
    foreach(args) do arg
        tr = tensorrank(arg)
        if tr != 0 && tr != maxrank
            throw(ArgumentError("all tensor arguments to broadcasted '$op' must have the same rank, or be scalars"))
        end
    end
end

# operators for symbolic terms
makeop(op::Symbol, arg1, args...) = SExpr(Call(), SRef(op), arg1, args...)
canonop(op::Symbol, arg1, args...) = canonicalize(makeop(op, arg1, args...))

# multiary operators
for op in (:+, :*, :max, :min)
    @eval Base.$op(args::Vararg{STerm}) = canonop($(Meta.quot(op)), args...)
end

# binary operators
for op in (:-, :^, :<, :<=, :>, :>=, :(==), :!=, :&, :|, :xor)
    @eval Base.$op(a::STerm, b::STerm) = canonop($(Meta.quot(op)), a, b)
end

function Base.:/(a::STerm, b::STerm)
    (isstaticzero(a) && isstaticzero(b)) && throw(ArgumentError("division of zero by zero"))
    return canonop(:/, a, b)
end

for op in (://, :÷)
    @eval function Base.$op(a::STerm, b::STerm)
        (isstaticzero(a) && isstaticzero(b)) && throw(ArgumentError("division of zero by zero"))
        isstaticzero(a) && return SUniform(0)
        isstaticone(b) && return a
        a === b && return SUniform(1)
        return canonop($(Meta.quot(op)), a, b)
    end
end

# mixed symbolic-numeric binary operations
for op in (:+, :-, :*, :max, :min, :/, ://, :÷, :^, :<, :<=, :>, :>=, :(==), :!=, :&, :|, :xor)
    @eval begin
        Base.$op(a::Number, b::STerm) = $op(SUniform(a), b)
        Base.$op(a::STerm, b::Number) = $op(a, SUniform(b))
    end
end

function Base.adjoint(t::STerm)
    tensorrank(t) == 0 && return t
    return canonop(:adjoint, t)
end

"""
    ⊡(a, b)

The double contraction operator, which contracts the first two indices of `a` with the first two indices of `b`.
"""
⊡(a::STerm, b::STerm) = canonop(:⊡, a, b)

"""
    ⊗(a, b)

The outer product operator, which creates a tensor by combining all indices of `a` with all indices of `b`.
"""
⊗(a::STerm, b::STerm) = canonop(:⊗, a, b)

# methods from LinearAlgebra
transpose(t::STerm) = t'

for op in (:det, :tr, :diag)
    @eval function $op(t::STerm)
        tensorrank(t) == 0 && return t
        return canonop($(Meta.quot(op)), t)
    end
end

function Base.inv(t::STerm)
    if iscall(t) && operation(t) === SRef(:inv)
        return only(arguments(t))
    end
    return canonop(:inv, t)
end

×(a::STerm, b::STerm) = canonop(:×, a, b)

function _isopof(op, x, y)
    if isexpr(x) && operation(x) === op && first(arguments(x)) === y
        return true
    else
        return false
    end
end

_isinvof(a, b) = _isopof(SRef(:inv), a, b)
_isadjof(a, b) = _isopof(SRef(:adj), a, b)

"""
    ⋅(a, b)

The dot product operator, which contracts the last index of `a` with the first index of `b`.
"""
function ⋅(a::STerm, b::STerm)
    # a * I = a, I * b = b
    isidentity(a) && return b
    isidentity(b) && return a
    R = tensorrank(a) + tensorrank(b) - 2
    # a * inv(a) = I, inv(b) * b = I
    (_isinvof(a, b) || _isinvof(b, a)) && return SIdTensor{R}()
    # a * adj(a) = det(a) * I, adj(b) * b = det(b) * I
    _isadjof(a, b) && return det(b) * SIdTensor{R}()
    _isadjof(b, a) && return det(a) * SIdTensor{R}()
    return canonop(:⋅, a, b)
end

"""
    adj(t)

The adjugate of a second-rank tensor, defined as the transpose of the cofactor matrix.
"""
function adj(t::STerm)
    tensorrank(t) == 0 && return t
    return canonop(:adj, t)
end

"""
    sym(t)

The symmetric part of a second-rank tensor, defined as `(t + t') / 2`.
"""
function sym(t::STerm)
    tensorrank(t) == 0 && return t
    return canonop(:sym, t)
end

"""
    asym(t)

The antisymmetric part of a second-rank tensor, defined as `(t - t') / 2`.
"""
function asym(t::STerm)
    tensorrank(t) == 0 && return SUniform(0)
    return canonop(:asym, t)
end

for op in (:gram, :cogram)
    @eval function $op(t::STerm)
        tensorrank(t) == 0 && return t^2
        return canonop($(Meta.quot(op)), t)
    end
end

# scalar unary operations
isunaryminus(expr::STerm) = iscall(expr) && operation(expr) === SRef(:-) && arity(expr) == 1

function Base.:-(arg::STerm)
    if iscall(arg) && (operation(arg) === SRef(:-) || operation(arg) === SRef(:+))
        return canonop(:-, arg)
    end
    return makeop(:-, arg)
end
Base.:-(arg::SUniform{0}) = arg
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
    @eval function Base.$op(arg::STerm)
        return canonop($(Meta.quot(op)), arg)
    end
end

# overloading broadcasting
function Base.Broadcast.broadcasted(f, args::Vararg{STerm})
    return canonop(:broadcasted, SFun(f), args...)
end
function Base.Broadcast.broadcasted(::typeof(Base.literal_pow), f, t::STerm, n::Val{N}) where {N}
    return canonop(:broadcasted, SRef(:^), t, SUniform(N))
end
Base.Broadcast.broadcasted(f, a::STerm, b::Number) = Base.Broadcast.broadcasted(f, a, SUniform(b))
Base.Broadcast.broadcasted(f, a::Number, b::STerm) = Base.Broadcast.broadcasted(f, SUniform(a), b)

Base.ifelse(cond::STerm, x::STerm, y::STerm) = canonop(:ifelse, cond, x, y)
Base.ifelse(cond::STerm, x::Number, y::STerm) = ifelse(cond, SUniform(x), y)
Base.ifelse(cond::STerm, x::STerm, y::Number) = ifelse(cond, x, SUniform(y))
Base.ifelse(cond::STerm, x::Number, y::Number) = ifelse(cond, SUniform(x), SUniform(y))

# tensor operations
Base.:+(t::Tensor) = t
Base.:-(t::Tensor{D,R,K}) where {D,R,K} = Tensor{D,R,K}(map(-, t.components)...)

Base.:+(t1::Tensor{D,R,K}, t2::Tensor{D,R,K}) where {D,R,K} = Tensor{D,R,K}(map(+, t1.components, t2.components)...)
Base.:-(t1::Tensor{D,R,K}, t2::Tensor{D,R,K}) where {D,R,K} = Tensor{D,R,K}(map(-, t1.components, t2.components)...)

@generated function Base.:+(t1::Tensor{D,R}, t2::Tensor{D,R}) where {D,R}
    comps = Expr(:tuple)
    for idx in CartesianIndices(ntuple(_ -> D, Val(R)))
        I = Tuple(idx)
        push!(comps.args, :(t1[$(I...)] + t2[$(I...)]))
    end
    quote
        @inline
        return Tensor{D,R}($comps...)
    end
end

@generated function Base.:-(t1::Tensor{D,R}, t2::Tensor{D,R}) where {D,R}
    comps = Expr(:tuple)
    for idx in CartesianIndices(ntuple(_ -> D, Val(R)))
        I = Tuple(idx)
        push!(comps.args, :(t1[$(I...)] - t2[$(I...)]))
    end
    quote
        @inline
        return Tensor{D,R}($comps...)
    end
end

const NumberOrTerm = Union{Number,STerm}

Base.:*(s::NumberOrTerm, t::Tensor{D,O,S}) where {D,O,S} = Tensor{D,O,S}(map(x -> s * x, t.components)...)
Base.:*(t::Tensor{D,O,S}, s::NumberOrTerm) where {D,O,S} = Tensor{D,O,S}(map(x -> x * s, t.components)...)

Base.:/(t::Tensor{D,O,S}, s::NumberOrTerm) where {D,O,S} = Tensor{D,O,S}(map(x -> x / s, t.components)...)
Base.://(t::Tensor{D,O,S}, s::NumberOrTerm) where {D,O,S} = Tensor{D,O,S}(map(x -> x // s, t.components)...)
Base.:÷(t::Tensor{D,O,S}, s::NumberOrTerm) where {D,O,S} = Tensor{D,O,S}(map(x -> x ÷ s, t.components)...)

@generated function ⊡(t1::Tensor{D,2}, t2::Tensor{D,2}) where {D}
    idx1(i, j) = linear_index(t1, i, j)
    idx2(i, j) = linear_index(t2, i, j)
    ex_p = Tuple(:(t1.components[$(idx1(i, j))] * t2.components[$(idx2(i, j))]) for i in 1:D, j in 1:D)
    ex_c = Expr(:call, :+, ex_p...)
    quote
        @inline
        return $ex_c
    end
end

⋅(v1::Vec{D}, v2::Vec{D}) where {D} = +(ntuple(i -> v1[i] * v2[i], Val(D))...)
@generated function ⋅(t::Tensor{D,2}, v::Vec{D}) where {D}
    idx(i, j) = linear_index(t, i, j)
    vc(i) = Expr(:call, :+, Tuple(:(t.components[$(idx(i, j))] * v.components[$j]) for j in 1:D)...)
    ex = Expr(:call, :(Vec{$D}), Tuple(vc(i) for i in 1:D)...)
    quote
        @inline
        return $ex
    end
end
@generated function ⋅(t1::Tensor{D,2}, t2::Tensor{D,2}) where {D}
    idx1(i, j) = linear_index(t1, i, j)
    idx2(i, j) = linear_index(t2, i, j)
    vc(i, j) = Expr(:call, :+, Tuple(:(t1.components[$(idx1(i, k))] * t2.components[$(idx2(k, j))]) for k in 1:D)...)
    ex = Expr(:call, :(Tensor{$D,2}), Tuple(vc(i, j) for i in 1:D, j in 1:D)...)
    quote
        @inline
        return $ex
    end
end

function ×(v1::Vec{3}, v2::Vec{3})
    return Vec{3}(v1[2] * v2[3] - v1[3] * v2[2],
                  v1[3] * v2[1] - v1[1] * v2[3],
                  v1[1] * v2[2] - v1[2] * v2[1])
end

@generated function ⊗(v1::Vec{D}, v2::Vec{D}) where {D}
    ex = Expr(:call, :(Tensor{$D,2}))
    for j in 1:D, i in 1:D
        push!(ex.args, :(v1.components[$i] * v2.components[$j]))
    end
    quote
        @inline
        return $ex
    end
end

function tr(t::Tensor{D,2}) where {D}
    return +(ntuple(i -> t[i, i], Val(D))...)
end

Base.transpose(t::SymTensor{2}) = t
Base.transpose(t::AltTensor{2}) = -t
Base.transpose(t::DiagTensor{2}) = t
@generated function Base.transpose(t::Tensor{D,2}) where {D}
    idx(i, j) = linear_index(t, i, j)
    ex = Expr(:call, :(Tensor{$D,2}), Tuple(:(t.components[$(idx(j, i))]) for i in 1:D, j in 1:D)...)
    quote
        @inline
        return $ex
    end
end

Base.adjoint(S::Tensor) = transpose(S)

det(t::Tensor{2,2}) = t[1, 1] * t[2, 2] - t[1, 2] * t[2, 1]
det(t::SymTensor{2,2}) = t[1, 1] * t[2, 2] - t[1, 2]^2
det(t::AltTensor{2,2}) = t[1, 2]^2
function det(t::Tensor{3,2})
    t[1, 1] * (t[2, 2] * t[3, 3] - t[2, 3] * t[3, 2]) +
    t[1, 2] * (t[2, 3] * t[3, 1] - t[2, 1] * t[3, 3]) +
    t[1, 3] * (t[2, 1] * t[3, 2] - t[2, 2] * t[3, 1])
end

det(::AltTensor{2,3}) = SUniform(0)

diag(t::Tensor{D,2}) where {D} = Vec(ntuple(i -> t[i, i], Val(D))...)

adj(t::Tensor{2,2}) = Tensor{2,2}(t[2, 2], -t[1, 2], -t[2, 1], t[1, 1])
adj(t::SymTensor{2,2}) = SymTensor{2,2}(t[1, 1], -t[1, 2], t[2, 2])
adj(t::AltTensor{2,2}) = AltTensor{2,2}(-t[1, 2])
function adj(t::Tensor{3,2})
    c11 = t[2, 2] * t[3, 3] - t[2, 3] * t[3, 2]
    c12 = t[2, 3] * t[3, 1] - t[2, 1] * t[3, 3]
    c13 = t[2, 1] * t[3, 2] - t[2, 2] * t[3, 1]
    c21 = t[1, 3] * t[3, 2] - t[1, 2] * t[3, 3]
    c22 = t[1, 1] * t[3, 3] - t[1, 3] * t[3, 1]
    c23 = t[1, 2] * t[3, 1] - t[1, 1] * t[3, 2]
    c31 = t[1, 2] * t[2, 3] - t[1, 3] * t[2, 2]
    c32 = t[1, 3] * t[2, 1] - t[1, 1] * t[2, 3]
    c33 = t[1, 1] * t[2, 2] - t[1, 2] * t[2, 1]
    return Tensor{3,2}(c11, c21, c31,
                       c12, c22, c32,
                       c13, c23, c33)
end
function adj(t::SymTensor{2,3})
    c11 = t[2, 2] * t[3, 3] - t[2, 3]^2
    c12 = t[2, 3] * t[1, 3] - t[1, 2] * t[3, 3]
    c13 = t[1, 2] * t[2, 3] - t[2, 2] * t[1, 3]
    c22 = t[1, 1] * t[3, 3] - t[1, 3]^2
    c23 = t[1, 2] * t[1, 3] - t[1, 1] * t[2, 3]
    c33 = t[1, 1] * t[2, 2] - t[1, 2]^2
    return SymTensor{2,3}(c11,
                          c12, c22,
                          c13, c23, c33)
end
function adj(t::AltTensor{2,3})
    c11 = t[2, 3]^2
    c12 = -t[1, 3] * t[2, 3]
    c13 = t[1, 2] * t[2, 3]
    c22 = t[1, 3]^2
    c23 = -t[1, 2] * t[1, 3]
    c33 = t[1, 2]^2
    return SymTensor{2,3}(c11,
                          c12, c22,
                          c13, c23, c33)
end

Base.inv(t::Tensor) = adj(t) / det(t)

sym(t::SymTensor) = t
@generated function sym(t::Tensor{D,2}) where {D}
    idx(i, j) = linear_index(t, i, j)
    ex = Expr(:call, :(SymTensor{2,$D}))
    for j in 1:D, i in j:D
        push!(ex.args, :((t.components[$(idx(i, j))] + t.components[$(idx(j, i))]) // 2))
    end
    quote
        @inline
        return $ex
    end
end

@generated function asym(t::Tensor{D,2}) where {D}
    idx(i, j) = linear_index(t, i, j)
    ex = Expr(:call, :(AltTensor{2,$D}))
    for j in 1:D, i in j+1:D
        push!(ex.args, :((t.components[$(idx(i, j))] - t.components[$(idx(j, i))]) // 2))
    end
    quote
        @inline
        return $ex
    end
end

"""
    gram(t)

The Gramian of a second-rank tensor, defined as `t' ⋅ t`.
"""
@generated function gram(t::Tensor{D,2}) where {D}
    idx(i, j) = linear_index(t, i, j)
    ex = Expr(:call, :(SymTensor{2,$D}))
    for j in 1:D, i in j:D
        comp = Expr(:call, :+)
        for k in 1:D
            push!(comp.args, :(t.components[$(idx(i, k))] * t.components[$(idx(j, k))]))
        end
        push!(ex.args, comp)
    end
    quote
        @inline
        return $ex
    end
end

"""
    cogram(t)

The co-Gramian of a second-rank tensor, defined as `t ⋅ t'`.
"""
@generated function cogram(t::Tensor{D,2}) where {D}
    idx(i, j) = linear_index(t, i, j)
    ex = Expr(:call, :(SymTensor{2,$D}))
    for j in 1:D, i in j:D
        comp = Expr(:call, :+)
        for k in 1:D
            push!(comp.args, :(t.components[$(idx(k, i))] * t.components[$(idx(k, j))]))
        end
        push!(ex.args, comp)
    end
    quote
        @inline
        return $ex
    end
end

# custom broadcasting for tensors
Base.Broadcast.broadcasted(f::F, arg::Tensor{D,R,K}) where {F,D,R,K} = Tensor{D,R,K}(map(f, arg.components)...)
