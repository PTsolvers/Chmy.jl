for op in (:+, :-)
    @eval function _check_tensor_ranks(op::SRef{$(Meta.quot(op))}, args::Vararg{STerm})
        if !all(x -> tensorrank(x) == tensorrank(args[1]), args)
            throw(ArgumentError("all tensor arguments to '$op' must have the same rank"))
        end
    end
end

function _check_tensor_ranks(::SRef{:*}, args::Vararg{STerm})
    if count(x -> tensorrank(x) > 0, args) > 1
        throw(ArgumentError("""
                            at most one tensor argument with rank > 0 allowed in multiplication, \
                            consider using '⋅' for single contraction, '⊡' for double contraction, or '⊗' for outer product, \
                            or use broadcasting to perform element-wise multiplication between tensors
                            """))
    end
end

for op in (:/, ://, :÷)
    @eval function _check_tensor_ranks(op::SRef{$(Meta.quot(op))}, ::STerm, b::STerm)
        if tensorrank(b) > 0
            throw(ArgumentError("denominator in '$op' must be a scalar term"))
        end
    end
end

for op in (:adjoint, :adj, :det, :inv, :tr, :diag, :sym, :asym, :gram, :cogram)
    @eval function _check_tensor_ranks(op::SRef{$(Meta.quot(op))}, t::STerm)
        if tensorrank(t) != 2 && tensorrank(t) != 0
            throw(ArgumentError("'$op' can only be applied to scalars or rank-2 tensors"))
        end
    end
end

function _check_tensor_ranks(::SRef{:×}, a::STerm, b::STerm)
    if tensorrank(a) != 1 || tensorrank(b) != 1
        throw(ArgumentError("cross product '×' can only be applied to rank-1 tensors (vectors)"))
    end
end

function _check_tensor_ranks(::SRef{:⋅}, a::STerm, b::STerm)
    if tensorrank(a) + tensorrank(b) < 2
        throw(ArgumentError("dot product '⋅' requires both arguments to have rank at least 1"))
    end
end

function _check_tensor_ranks(::SRef{:⊡}, a::STerm, b::STerm)
    if tensorrank(a) < 2 || tensorrank(b) < 2
        throw(ArgumentError("double tensor contraction '⊡' requires both arguments to have rank at least 2"))
    end
end

function _check_scalar_ranks(op::SRef, args::Vararg{STerm})
    if any(x -> tensorrank(x) != 0, args)
        throw(ArgumentError("'$op' can only be applied to scalar terms, consider using broadcasting"))
    end
end

function _check_tensor_ranks(::SRef{:broadcasted}, op::STerm, args::Vararg{STerm})
    maxrank = maximum(tensorrank, args)
    foreach(args) do arg
        tr = tensorrank(arg)
        if tr != 0 && tr != maxrank
            throw(ArgumentError("all tensor arguments to broadcasted '$op' must have the same rank, or be scalars"))
        end
    end
end

# operators for symbolic terms
Base.:+(arg::STerm) = arg

# addition with filtering zeros
function Base.:+(args::Vararg{STerm})
    _check_tensor_ranks(SRef(:+), args...)
    nz_args = filter(!isstaticzero, args)
    length(nz_args) == 0 && return args[1]
    length(nz_args) == 1 && return first(nz_args)
    return SExpr(Call(), SRef(:+), nz_args...)
end

# multiplication with filtering ones and checking for zeros
function Base.:*(args::Vararg{STerm})
    _check_tensor_ranks(SRef(:*), args...)
    if any(isstaticzero, args)
        maxrank = maximum(tensorrank, args)
        return SZeroTensor{maxrank}()
    end
    nn_args = filter(x -> !isstaticone(x), args)
    length(nn_args) == 0 && return SUniform(1)
    length(nn_args) == 1 && return first(nn_args)
    return SExpr(Call(), SRef(:*), nn_args...)
end

# multiary operators
for op in (:max, :min)
    @eval function Base.$op(args::Vararg{STerm})
        _check_scalar_ranks(SRef($(Meta.quot(op))), args...)
        return SExpr(Call(), SRef($(Meta.quot(op))), args...)
    end
end

# subtraction with zero detection
function Base.:-(a::STerm, b::STerm)
    _check_tensor_ranks(SRef(:-), a, b)
    isstaticzero(a) && return -b
    isstaticzero(b) && return a
    a === b && return SZeroTensor{tensorrank(a)}()
    return SExpr(Call(), SRef(:-), a, b)
end

# division operators
for op in (:/, ://, :÷)
    @eval function Base.$op(a::STerm, b::STerm)
        _check_scalar_ranks(SRef($(Meta.quot(op))), a, b)
        (isstaticzero(a) && isstaticzero(b)) && throw(ArgumentError("division of zero by zero"))
        isstaticzero(a) && return SUniform(0)
        isstaticone(b) && return a
        a === b && return SUniform(1)
        return SExpr(Call(), SRef($(Meta.quot(op))), a, b)
    end
end

function Base.:^(a::STerm, b::STerm)
    _check_scalar_ranks(SRef(:^), a, b)
    isstaticzero(b) && return SUniform(1)
    isstaticzero(a) && return SUniform(0)
    isstaticone(a) && return SUniform(1)
    isstaticone(b) && return a
    return SExpr(Call(), SRef(:^), a, b)
end

# scalar binary operators
for op in (:<, :<=, :>, :>=, :(==), :!=, :&, :|, :xor)
    @eval function Base.$op(a::STerm, b::STerm)
        _check_scalar_ranks(SRef($(Meta.quot(op))), a, b)
        return SExpr(Call(), SRef($(Meta.quot(op))), a, b)
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
    _check_tensor_ranks(SRef(:adjoint), t)
    tensorrank(t) == 0 && return t
    SExpr(Call(), SRef(:adjoint), t)
end

# tensor double contraction
function ⊡(a::STerm, b::STerm)
    _check_tensor_ranks(SRef(:⊡), a, b)
    SExpr(Call(), SRef(:⊡), a, b)
end

# tensor outer product
⊗(a::STerm, b::STerm) = SExpr(Call(), SRef(:⊗), a, b)

# methods from LinearAlgebra
LinearAlgebra.transpose(t::STerm) = t'

for op in (:det, :tr, :diag)
    @eval function LinearAlgebra.$op(t::STerm)
        _check_tensor_ranks(SRef($(Meta.quot(op))), t)
        tensorrank(t) == 0 && return t
        SExpr(Call(), SRef($(Meta.quot(op))), t)
    end
end

function LinearAlgebra.inv(t::STerm)
    _check_tensor_ranks(SRef(:inv), t)
    SExpr(Call(), SRef(:inv), t)
end

function LinearAlgebra.:×(a::STerm, b::STerm)
    _check_tensor_ranks(SRef(:×), a, b)
    SExpr(Call(), SRef(:×), a, b)
end

function _isopof(op, x, y)
    if isexpr(x) && operation(x) === op && first(arguments(x)) === y
        return true
    else
        return false
    end
end

_isinvof(a, b) = _isopof(SRef(:inv), a, b)
_isadjof(a, b) = _isopof(SRef(:adj), a, b)

function LinearAlgebra.:⋅(a::STerm, b::STerm)
    _check_tensor_ranks(SRef(:⋅), a, b)
    isidentity(a) && return b
    isidentity(b) && return a
    R = tensorrank(a) + tensorrank(b) - 2
    (_isinvof(a, b) || _isinvof(b, a)) && return SIdTensor{R}()
    _isadjof(a, b) && return LinearAlgebra.det(b) * SIdTensor{R}()
    _isadjof(b, a) && return LinearAlgebra.det(a) * SIdTensor{R}()
    SExpr(Call(), SRef(:⋅), a, b)
end

function adj(t::STerm)
    _check_tensor_ranks(SRef(:adj), t)
    tensorrank(t) == 0 && return SUniform(1)
    SExpr(Call(), SRef(:adj), t)
end

function sym(t::STerm)
    _check_tensor_ranks(SRef(:sym), t)
    tensorrank(t) == 0 && return t
    SExpr(Call(), SRef(:sym), t)
end

function asym(t::STerm)
    _check_tensor_ranks(SRef(:asym), t)
    tensorrank(t) == 0 && return SUniform(0)
    SExpr(Call(), SRef(:asym), t)
end

for op in (:gram, :cogram)
    @eval function $op(t::STerm)
        _check_tensor_ranks(SRef($(Meta.quot(op))), t)
        tensorrank(t) == 0 && return t^2
        SExpr(Call(), SRef($(Meta.quot(op))), t)
    end
end

# scalar unary operations
Base.:-(arg::STerm) = SExpr(Call(), SRef(:-), arg)
Base.:-(arg::SUniform{0}) = arg
Base.:-(arg::SZeroTensor) = arg

function Base.:-(arg::SExpr{Call})
    if operation(arg) === SRef(:-) && length(arguments(arg)) == 1
        return first(arguments(arg))
    else
        return SExpr(Call(), SRef(:-), arg)
    end
end

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
        _check_scalar_ranks(SRef($(Meta.quot(op))), arg)
        SExpr(Call(), SRef($(Meta.quot(op))), arg)
    end
end

# overloading broadcasting
function Base.Broadcast.broadcasted(f, args::Vararg{STerm})
    _check_tensor_ranks(SRef(:broadcasted), SFun(f), args...)
    return SExpr(Call(), SRef(:broadcasted), SFun(f), args...)
end

function Base.Broadcast.broadcasted(::typeof(Base.literal_pow), f, t::STerm, n::Val{N}) where {N}
    return SExpr(Call(), SRef(:broadcasted), SRef(:^), t, SUniform(N))
end

Base.Broadcast.broadcasted(f, a::STerm, b::Number) = Base.Broadcast.broadcasted(f, a, SUniform(b))
Base.Broadcast.broadcasted(f, a::Number, b::STerm) = Base.Broadcast.broadcasted(f, SUniform(a), b)

function Base.ifelse(cond::STerm, x::STerm, y::STerm)
    _check_scalar_ranks(SRef(:ifelse), cond, x, y)
    SExpr(Call(), SRef(:ifelse), cond, x, y)
end

Base.ifelse(cond::STerm, x::Number, y::STerm)  = ifelse(cond, SUniform(x), y)
Base.ifelse(cond::STerm, x::STerm, y::Number)  = ifelse(cond, x, SUniform(y))
Base.ifelse(cond::STerm, x::Number, y::Number) = ifelse(cond, SUniform(x), SUniform(y))

# tensor operations

Base.:+(t::Tensor) = t
Base.:-(t::Tensor{R,D,K}) where {R,D,K} = Tensor{R,D,K}(map(-, t.components)...)

Base.:+(t1::Tensor{R,D,K}, t2::Tensor{R,D,K}) where {R,D,K} = Tensor{R,D,K}(map(+, t1.components, t2.components)...)
Base.:-(t1::Tensor{R,D,K}, t2::Tensor{R,D,K}) where {R,D,K} = Tensor{R,D,K}(map(-, t1.components, t2.components)...)

@generated function Base.:+(t1::Tensor{R,D}, t2::Tensor{R,D}) where {R,D}
    comps = Expr(:tuple)
    for idx in CartesianIndices(ntuple(_ -> D, Val(R)))
        I = Tuple(idx)
        push!(comps.args, :(t1[$(I...)] + t2[$(I...)]))
    end
    quote
        @inline
        return Tensor{R,D}($comps...)
    end
end

@generated function Base.:-(t1::Tensor{R,D}, t2::Tensor{R,D}) where {R,D}
    comps = Expr(:tuple)
    for idx in CartesianIndices(ntuple(_ -> D, Val(R)))
        I = Tuple(idx)
        push!(comps.args, :(t1[$(I...)] - t2[$(I...)]))
    end
    quote
        @inline
        return Tensor{R,D}($comps...)
    end
end

const NumberOrTerm = Union{Number,STerm}

Base.:*(s::NumberOrTerm, t::Tensor{O,D,S}) where {O,D,S} = Tensor{O,D,S}(map(x -> s * x, t.components)...)
Base.:*(t::Tensor{O,D,S}, s::NumberOrTerm) where {O,D,S} = Tensor{O,D,S}(map(x -> x * s, t.components)...)

Base.:/(t::Tensor{O,D,S}, s::NumberOrTerm) where {O,D,S} = Tensor{O,D,S}(map(x -> x / s, t.components)...)
Base.://(t::Tensor{O,D,S}, s::NumberOrTerm) where {O,D,S} = Tensor{O,D,S}(map(x -> x // s, t.components)...)
Base.:÷(t::Tensor{O,D,S}, s::NumberOrTerm) where {O,D,S} = Tensor{O,D,S}(map(x -> x ÷ s, t.components)...)

@generated function ⊡(t1::Tensor{2,D}, t2::Tensor{2,D}) where {D}
    idx1(i, j) = linear_index(t1, i, j)
    idx2(i, j) = linear_index(t2, i, j)
    ex_p = Tuple(:(t1.components[$(idx1(i, j))] * t2.components[$(idx2(i, j))]) for i in 1:D, j in 1:D)
    ex_c = Expr(:call, :+, ex_p...)
    quote
        @inline
        return $ex_c
    end
end

LinearAlgebra.:⋅(v1::Vec{D}, v2::Vec{D}) where {D} = +(ntuple(i -> v1[i] * v2[i], Val(D))...)

@generated function LinearAlgebra.:⋅(t::Tensor{2,D}, v::Vec{D}) where {D}
    idx(i, j) = linear_index(t, i, j)
    vc(i) = Expr(:call, :+, Tuple(:(t.components[$(idx(i, j))] * v.components[$j]) for j in 1:D)...)
    ex = Expr(:call, :(Vec{$D}), Tuple(vc(i) for i in 1:D)...)
    quote
        @inline
        return $ex
    end
end

@generated function LinearAlgebra.:⋅(t1::Tensor{2,D}, t2::Tensor{2,D}) where {D}
    idx1(i, j) = linear_index(t1, i, j)
    idx2(i, j) = linear_index(t2, i, j)
    vc(i, j) = Expr(:call, :+, Tuple(:(t1.components[$(idx1(i, k))] * t2.components[$(idx2(k, j))]) for k in 1:D)...)
    ex = Expr(:call, :(Tensor{2,$D}), Tuple(vc(i, j) for i in 1:D, j in 1:D)...)
    quote
        @inline
        return $ex
    end
end

function LinearAlgebra.:×(v1::Vec{3}, v2::Vec{3})
    return Vec{3}(v1[2] * v2[3] - v1[3] * v2[2],
                  v1[3] * v2[1] - v1[1] * v2[3],
                  v1[1] * v2[2] - v1[2] * v2[1])
end

@generated function ⊗(v1::Vec{D}, v2::Vec{D}) where {D}
    ex = Expr(:call, :(Tensor{2,$D}))
    for j in 1:D
        for i in 1:D
            push!(ex.args, :(v1.components[$i] * v2.components[$j]))
        end
    end
    quote
        @inline
        return $ex
    end
end

function LinearAlgebra.tr(t::Tensor{2,D}) where {D}
    return +(ntuple(i -> t[i, i], Val(D))...)
end

Base.transpose(t::SymTensor{2}) = t
Base.transpose(t::AltTensor{2}) = -t
Base.transpose(t::DiagTensor{2}) = t

@generated function Base.transpose(t::Tensor{2,D}) where {D}
    idx(i, j) = linear_index(t, i, j)
    ex = Expr(:call, :(Tensor{2,$D}), Tuple(:(t.components[$(idx(j, i))]) for i in 1:D, j in 1:D)...)
    quote
        @inline
        return $ex
    end
end

Base.adjoint(S::Tensor) = transpose(S)

LinearAlgebra.det(t::Tensor{2,2}) = t[1, 1] * t[2, 2] - t[1, 2] * t[2, 1]
LinearAlgebra.det(t::SymTensor{2,2}) = t[1, 1] * t[2, 2] - t[1, 2]^2
LinearAlgebra.det(t::AltTensor{2,2}) = t[1, 2]^2

function LinearAlgebra.det(t::Tensor{2,3})
    t[1, 1] * (t[2, 2] * t[3, 3] - t[2, 3] * t[3, 2]) +
    t[1, 2] * (t[2, 3] * t[3, 1] - t[2, 1] * t[3, 3]) +
    t[1, 3] * (t[2, 1] * t[3, 2] - t[2, 2] * t[3, 1])
end

LinearAlgebra.det(::AltTensor{2,3}) = SUniform(0)

LinearAlgebra.diag(t::Tensor{2,D}) where {D} = Vec(ntuple(i -> t[i, i], Val(D))...)

adj(t::Tensor{2,2}) = Tensor{2,2}(t[2, 2], -t[1, 2], -t[2, 1], t[1, 1])
adj(t::SymTensor{2,2}) = SymTensor{2,2}(t[1, 1], -t[1, 2], t[2, 2])
adj(t::AltTensor{2,2}) = AltTensor{2,2}(-t[1, 2])

function adj(t::Tensor{2,3})
    c11 = t[2, 2] * t[3, 3] - t[2, 3] * t[3, 2]
    c12 = t[2, 3] * t[3, 1] - t[2, 1] * t[3, 3]
    c13 = t[2, 1] * t[3, 2] - t[2, 2] * t[3, 1]
    c21 = t[1, 3] * t[3, 2] - t[1, 2] * t[3, 3]
    c22 = t[1, 1] * t[3, 3] - t[1, 3] * t[3, 1]
    c23 = t[1, 2] * t[3, 1] - t[1, 1] * t[3, 2]
    c31 = t[1, 2] * t[2, 3] - t[1, 3] * t[2, 2]
    c32 = t[1, 3] * t[2, 1] - t[1, 1] * t[2, 3]
    c33 = t[1, 1] * t[2, 2] - t[1, 2] * t[2, 1]
    return Tensor{2,3}(c11, c21, c31,
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

Base.inv(t::Tensor) = adj(t) / LinearAlgebra.det(t)

sym(t::SymTensor) = t

@generated function sym(t::Tensor{2,D}) where {D}
    idx(i, j) = linear_index(t, i, j)
    ex = Expr(:call, :(SymTensor{2,$D}))
    for j in 1:D
        for i in j:D
            push!(ex.args, :((t.components[$(idx(i, j))] + t.components[$(idx(j, i))]) // 2))
        end
    end
    quote
        @inline
        return $ex
    end
end

@generated function asym(t::Tensor{2,D}) where {D}
    idx(i, j) = linear_index(t, i, j)
    ex = Expr(:call, :(AltTensor{2,$D}))
    for j in 1:D
        for i in j+1:D
            push!(ex.args, :((t.components[$(idx(i, j))] - t.components[$(idx(j, i))]) // 2))
        end
    end
    quote
        @inline
        return $ex
    end
end

@generated function gram(t::Tensor{2,D}) where {D}
    idx(i, j) = linear_index(t, i, j)
    ex = Expr(:call, :(SymTensor{2,$D}))
    for j in 1:D
        for i in j:D
            comp = Expr(:call, :+)
            for k in 1:D
                push!(comp.args, :(t.components[$(idx(i, k))] * t.components[$(idx(j, k))]))
            end
            push!(ex.args, comp)
        end
    end
    quote
        @inline
        return $ex
    end
end

@generated function cogram(t::Tensor{2,D}) where {D}
    idx(i, j) = linear_index(t, i, j)
    ex = Expr(:call, :(SymTensor{2,$D}))
    for j in 1:D
        for i in j:D
            comp = Expr(:call, :+)
            for k in 1:D
                push!(comp.args, :(t.components[$(idx(k, i))] * t.components[$(idx(k, j))]))
            end
            push!(ex.args, comp)
        end
    end
    quote
        @inline
        return $ex
    end
end
