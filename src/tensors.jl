abstract type TensorKind end

struct NoKind <: TensorKind end
struct SymKind <: TensorKind end
struct AltKind <: TensorKind end
struct DiagKind <: TensorKind end

struct STensor{R,K,N} <: STerm end

tensorrank(::STensor{R}) where {R} = R
tensorkind(::STensor{<:Any,K}) where {K} = K
name(::STensor{<:Any,<:Any,N}) where {N} = N

tensorrank(::SIndex) = 0

tensorrank(::SUniform) = 0
tensorkind(::SUniform) = NoKind

"""
    STensor{R,K}(name)

Construct a symbolic tensor of rank `R` and kind `K` with the given name.
"""
STensor{R,K}(name::Symbol) where {R,K} = STensor{R,K,name}()

"""
    STensor{R}(name)

Construct a symbolic tensor of rank `R` and kind `NoKind` with the given name.
"""
STensor{R}(name::Symbol) where {R} = STensor{R,NoKind,name}()

"""
    SSymTensor{R}(name)

Construct a symbolic symmetric tensor of rank `R` with the given name.
"""
const SSymTensor{R} = STensor{R,SymKind}

"""
    SAltTensor{R}(name)

Construct a symbolic alternating tensor of rank `R` with the given name.
"""
const SAltTensor{R} = STensor{R,AltKind}

"""
    SDiagTensor{R}(name)

Construct a symbolic diagonal tensor of rank `R` with the given name.
"""
const SDiagTensor{R} = STensor{R,DiagKind}

"""
    SZeroTensor{R}()

Construct the zero tensor of rank `R`.
"""
struct SZeroTensor{R} <: STerm end
SZeroTensor{0}() = SUniform(0)

"""
    SIdTensor{R}()

Construct the identity tensor of rank `R`.
"""
struct SIdTensor{R} <: STerm end
SIdTensor{0}() = SUniform(1)

tensorrank(::SZeroTensor{R}) where {R} = R
tensorrank(::SIdTensor{R}) where {R} = R

"""
    SScalar(name)

Construct a symbolic scalar with the given name.
"""
const SScalar = STensor{0,NoKind}

"""
    SVec{D}(name)

Construct a symbolic vector of dimension `D` with the given name.
"""
const SVec = STensor{1,NoKind}

const IntegerOrSUniform = Union{Integer,SUniform}

sallunique(I::Tuple{SUniform}) = true
function sallunique(I)
    I[1] === I[2] && return false
    return sallunique(Base.tail(I))
end
function sinversion_count(I::NTuple{N,SUniform}) where {N}
    return inversion_count(map(value, I))
end

Base.getindex(s::SScalar) = s
Base.getindex(::SZeroTensor{R}, I::Vararg{IntegerOrSUniform,R}) where {R} = SUniform(0)
Base.getindex(t::SIdTensor{R}, I::Vararg{IntegerOrSUniform,R}) where {R} = Base.getindex(t, map(SUniform, I)...)
Base.getindex(::SIdTensor{R}, I::Vararg{SUniform,R}) where {R} = all(x -> x === I[1], I) ? SUniform(1) : SUniform(0)
Base.getindex(t::STensor{R}, I::Vararg{IntegerOrSUniform,R}) where {R} = Base.getindex(t, map(SUniform, I)...)
function Base.getindex(t::SDiagTensor{R,D}, I::Vararg{SUniform,R}) where {R,D}
    all(x -> x === I[1], I) || return SUniform(0)
    return SExpr(Comp(), t, I...)
end
function Base.getindex(t::SSymTensor{R}, I::Vararg{SUniform,R}) where {R}
    J = ssort(I; lt=isless_lex)
    SExpr(Comp(), t, J...)
end
function Base.getindex(t::SAltTensor{R}, I::Vararg{SUniform,R}) where {R}
    J = ssort(I; lt=isless_lex)
    sallunique(J) || return SUniform(0)
    v = SExpr(Comp(), t, J...)
    return iseven(sinversion_count(I)) ? v : -v
end
function Base.getindex(t::STensor{R}, I::Vararg{SUniform,R}) where {R}
    SExpr(Comp(), t, I...)
end

"""
    tensorrank(expr)

Return the rank of the tensor represented by the symbolic expression `expr`.
"""
function tensorrank end

tensorrank(expr::SExpr{Call}) = tensorrank(operation(expr), arguments(expr)...)
tensorrank(::SExpr{Comp}) = 0
tensorrank(::SExpr{Loc}) = 0
tensorrank(::SExpr{Ind}) = 0

tensorrank(::SFun, args...) = 0
tensorrank(::SRef, args...) = 0

tensorrank(::SRef{:adjoint}, t) = tensorrank(t)
tensorrank(::SRef{:broadcasted}, op, args...) = maximum(map(tensorrank, args))

tensorrank(::SRef{:+}, args...) = tensorrank(first(args))
tensorrank(::SRef{:*}, args...) = maximum(tensorrank, args)
tensorrank(::SRef{:-}, a)       = tensorrank(a)
tensorrank(::SRef{:-}, a, b)    = tensorrank(a)
tensorrank(::SRef{:⋅}, a, b)    = tensorrank(a) + tensorrank(b) - 2
tensorrank(::SRef{:×}, a, b)    = 1
tensorrank(::SRef{:⊡}, a, b)    = tensorrank(a) + tensorrank(b) - 4
tensorrank(::SRef{:⊗}, a, b)    = tensorrank(a) + tensorrank(b)

tensorrank(::SRef{:sym}, t)    = tensorrank(t)
tensorrank(::SRef{:asym}, t)   = tensorrank(t)
tensorrank(::SRef{:adj}, t)    = tensorrank(t)
tensorrank(::SRef{:inv}, t)    = tensorrank(t)
tensorrank(::SRef{:gram}, t)   = 2
tensorrank(::SRef{:cogram}, t) = 2
tensorrank(::SRef{:diag}, t)   = 1

tensorrank(::Gradient, t)   = tensorrank(t) + 1
tensorrank(::Divergence, t) = tensorrank(t) - 1
tensorrank(::Curl, t)       = 1

struct Tensor{D,R,K,C}
    components::C
end

Base.ndims(::Tensor{D}) where {D} = D

tensorrank(::Tensor{<:Any,R}) where {R} = R
dimensions(::Tensor{D}) where {D} = D
tensorkind(::Tensor{<:Any,<:Any,K}) where {K} = K

tensorrank(::Type{Tensor{<:Any,R}}) where {R} = R
dimensions(::Type{Tensor{D}}) where {D} = D
tensorkind(::Type{Tensor{<:Any,<:Any,K}}) where {K} = K

Base.length(t::Tensor) = length(t.components)

ncomponents(t::Tensor) = ncomponents(typeof(t))
ncomponents(tt::Type{Tensor}) = ncomponents(tensorkind(tt), Val(dimensions(tt)), Val(tensorrank(tt)))

ncomponents(::Type{NoKind}, ::Val{D}, ::Val{R}) where {D,R} = D^R
ncomponents(::Type{SymKind}, ::Val{D}, ::Val{R}) where {D,R} = binomial(D + R - 1, R)
ncomponents(::Type{AltKind}, ::Val{D}, ::Val{R}) where {D,R} = binomial(D, R)
ncomponents(::Type{DiagKind}, ::Val{D}, ::Val{R}) where {D,R} = D

"""
    SymTensor{D,R}(data...)

Construct a symmetric tensor of rank `R` and dimension `D` with the given components.
"""
const SymTensor{D,R} = Tensor{D,R,SymKind}

"""
    AltTensor{D,R}(data...)

Construct an alternating tensor of rank `R` and dimension `D` with the given components.
"""
const AltTensor{D,R} = Tensor{D,R,AltKind}

"""
    DiagTensor{D,R}(data...)

Construct a diagonal tensor of rank `R` and dimension `D` with the given components.
"""
const DiagTensor{D,R} = Tensor{D,R,DiagKind}

"""
    Vec{D}(data...)

Construct a vector of dimension `D` with the given components.
"""
const Vec{D} = Tensor{D,1}

isidentity(::STerm) = false
isidentity(::SIdTensor) = true

isstaticzero(::SZeroTensor) = true
isstaticzero(t::Tensor) = all(isstaticzero, t.components)

linear_index(t::Tensor, I::Vararg{Int}) = linear_index(typeof(t), I...)
linear_index(::Type{<:Tensor{D,R,K}}, I::Vararg{Int,R}) where {D,R,K} = linear_index(K, Val(D), I...)
function linear_index(::Type{SymKind}, ::Val, I::Vararg{Int,O}) where {O}
    J = sort(I)
    return 1 + sum(ntuple(k -> binomial(J[k] + k - 2, k), Val(O)))
end
function linear_index(::Type{AltKind}, ::Val, I::Vararg{Int,O}) where {O}
    J = sort(I)
    return 1 + sum(ntuple(k -> binomial(J[k] - 1, k), Val(O)))
end
function linear_index(::Type{NoKind}, ::Val{D}, I::Vararg{Int,O}) where {O,D}
    return LinearIndices(ntuple(_ -> D, Val(O)))[I...]
end
function linear_index(::Type{DiagKind}, ::Val{D}, I::Vararg{Int,O}) where {O,D}
    return I[1]
end

Base.getindex(t::Tensor{D,R,K}, I::Vararg{Int,R}) where {D,R,K} = t.components[linear_index(K, Val(D), I...)]
function Base.getindex(t::DiagTensor{D,R}, I::Vararg{Int,R}) where {D,R}
    all(==(I[1]), I) || return SUniform(0)
    return t.components[I[1]]
end
function Base.getindex(t::AltTensor{D,R}, I::Vararg{Int,R}) where {D,R}
    allunique(I) || return SUniform(0)
    J = sort(I)
    v = t.components[linear_index(AltKind, Val(D), J...)]
    return iseven(inversion_count(I)) ? v : -v
end

Vec(data::Vararg{STerm,M}) where {M} = Vec{M}(data...)

"""
    Tensor{D,R}(data...)

Construct a tensor of dimension `D` and rank `R` with the given components.
The kind of the tensor is automatically determined based on the symmetries of the components.
"""
Tensor{D,R}(data::Vararg{STerm,M}) where {D,R,M} = Tensor{D,R,NoKind}(data...)
function Tensor{D,R,K}(data::Vararg{STerm,M}) where {D,R,K,M}
    N = ncomponents(K, Val(D), Val(R))
    M == N || error("expected $N components to construct order-$R $(K) Tensor, got $M")
    construct_tensor(Tensor{D,R,K}, data)
end

"""
    Tensor{D}(s::STensor)

Construct a dimension-`D` component representation of symbolic tensor `s`.
The result has the same rank and the symmetries as `s`.
"""
Tensor{D}(s::SScalar) where {D} = s
@generated function Tensor{D}(s::STensor{R,K}) where {D,R,K}
    ex = Expr(:call, :(Tensor{$D,$R,$K}))
    comp_expr(I) = :(s[$(map(i -> :(SUniform($i)), I)...)])

    if K <: NoKind
        for idx in CartesianIndices(ntuple(_ -> D, Val(R)))
            I = Tuple(idx)
            push!(ex.args, comp_expr(I))
        end
    elseif K <: SymKind
        foreach_nondecreasing(Val(D), Val(R)) do I
            push!(ex.args, comp_expr(I))
        end
    elseif K <: AltKind
        foreach_increasing(Val(D), Val(R)) do I
            push!(ex.args, comp_expr(I))
        end
    elseif K <: DiagKind
        for i in 1:D
            I = ntuple(_ -> i, Val(R))
            push!(ex.args, comp_expr(I))
        end
    else
        error("unsupported tensor kind $K")
    end

    return ex
end

raw_tensor(::Type{Tensor{D,R,K}}, data::NTuple{N,STerm}) where {D,R,K,N} = Tensor{D,R,K,typeof(data)}(data)
@generated function Tensor{D}(::SZeroTensor{R}) where {D,R}
    data = Expr(:tuple, ntuple(_ -> :(SUniform(0)), Val(D))...)
    return :(raw_tensor(Tensor{$D,$R,DiagKind}, $data))
end
@generated function Tensor{D}(::SIdTensor{R}) where {D,R}
    data = Expr(:tuple, ntuple(_ -> :(SUniform(1)), Val(D))...)
    return :(raw_tensor(Tensor{$D,$R,DiagKind}, $data))
end

texpr(expr::STerm, ::Val) = expr
texpr(s::SScalar, ::Val) = s

texpr(::SRef{F}, ::Val) where {F} = F
texpr(s::Union{STensor,SZeroTensor,SIdTensor}, ::Val{D}) where {D} = :(Tensor{$D}($s))

texpr(expr::SExpr{Call}, d::Val) = texpr(operation(expr), d, arguments(expr))

function texpr(op::SRef, d::Val, args::NTuple{N,STerm}) where {N}
    return Expr(:call, texpr(op, d), map(arg -> texpr(arg, d), args)...)
end

# TODO: API for converting tensor calculus differential operators to the component form
function texpr(op::Gradient, d::Val, args::Tuple{STerm})
    arg = only(args)
    result = compute_scalar_gradient(PartialDerivative(op.op), arg, d)
    return :($result)
end

function texpr(op::Divergence, d::Val{D}, args::Tuple{STerm}) where {D}
    arg = Tensor{D}(only(args))
    result = compute_vector_divergence(PartialDerivative(op.op), arg, d)
    return :($result)
end

# TODO: generalise to tensor fields
function compute_scalar_gradient(∂::PartialDerivative, s::STerm, d::Val)
    return Vec(ntuple(i -> ∂(s, i), d)...)
end

# TODO: generalise to tensor fields
function compute_vector_divergence(∂::PartialDerivative, v::Vec, d::Val)
    return +(ntuple(i -> ∂(v[i], i), d)...)
end

function texpr(expr::SExpr{Comp}, d::Val)
    return Expr(:ref,
                _tensor_index_base_expr(argument(expr), d),
                map(_tensor_index_value_expr, indices(expr))...)
end

function texpr(expr::SExpr{Ind}, d::Val)
    return Expr(:ref,
                texpr(argument(expr), d),
                map(_tensor_dynamic_index_expr, indices(expr))...)
end

_tensor_index_base_expr(s::STensor, ::Val{D}) where {D} = :(Tensor{$D}($s))
_tensor_index_base_expr(expr::STerm, d::Val) = texpr(expr, d)

function _tensor_index_value_expr(::SUniform{Value}) where {Value}
    Value isa Integer || error("component index must be an integer, got $Value")
    return Int(Value)
end
_tensor_index_value_expr(i::STerm) = error("unsupported component index $i")

_tensor_dynamic_index_expr(i::SUniform) = _tensor_index_value_expr(i)
function _tensor_dynamic_index_expr(::SIndex)
    error("cannot convert indexed symbolic expression with placeholder indices to components")
end
_tensor_dynamic_index_expr(i::STerm) = error("unsupported symbolic index $i")

"""
    Tensor{D}(expr::STerm)

Convert a symbolic tensor expression `expr` to dimension-`D` component form by
expanding symbolic tensor leaves and evaluating the expression with `Tensor`
operations. Scalar symbolic values (including `SUniform` and `SScalar`) are
preserved as symbolic terms.
"""
@generated function Tensor{D}(expr::STerm) where {D}
    expri = expr.instance
    return quote
        Base.@_propagate_inbounds_meta
        $(texpr(expri, Val(D)))
    end
end

# construct the most specific tensor type based on the symmetries of the components
function construct_tensor(::Type{Tensor{D,R,DiagKind}}, data::NTuple{N,STerm}) where {D,R,N}
    all(isstaticzero, data) && return SZeroTensor{R}()
    all(isstaticone, data) && return SIdTensor{R}()
    return Tensor{D,R,DiagKind,typeof(data)}(data)
end
function construct_tensor(::Type{Tensor{D,R,AltKind}}, data::NTuple{N,STerm}) where {D,R,N}
    all(isstaticzero, data) && return SZeroTensor{R}()
    return Tensor{D,R,AltKind,typeof(data)}(data)
end
function construct_tensor(::Type{Tensor{D,R,SymKind}}, data::NTuple{N,STerm}) where {D,R,N}
    if isdiagonal(Val(D), Val(R), data)
        diag_comps = diagonal_from_symmetric(Tensor{D,R}, data)
        return construct_tensor(Tensor{D,R,DiagKind}, diag_comps)
    end
    return Tensor{D,R,SymKind,typeof(data)}(data)
end
function construct_tensor(::Type{Tensor{D,R,NoKind}}, data::NTuple{N,STerm}) where {D,R,N}
    if issymmetric(Val(D), Val(R), data)
        sym_comps = symmetric_components(Tensor{D,R}, data)
        return construct_tensor(Tensor{D,R,SymKind}, sym_comps)
    end

    if isalternating(Val(D), Val(R), data)
        alt_comps = alternating_components(Tensor{D,R}, data)
        return construct_tensor(Tensor{D,R,AltKind}, alt_comps)
    end

    return Tensor{D,R,NoKind,typeof(data)}(data)
end

@generated function issymmetric(::Val{D}, ::Val{R}, v::NTuple{N,STerm}) where {D,R,N}
    check = Expr(:&&)

    for idx in CartesianIndices(ntuple(_ -> D, Val(R)))
        I = Tuple(idx)
        J = sort(I)

        I == J && continue

        i = linear_index(NoKind, Val(D), I...)
        j = linear_index(NoKind, Val(D), J...)

        push!(check.args, :(v[$i] === v[$j]))
    end

    return check
end

@generated function isalternating(::Val{D}, ::Val{R}, v::NTuple{N,STerm}) where {D,R,N}
    check = Expr(:&&)

    for idx in CartesianIndices(ntuple(_ -> D, Val(R)))
        I = Tuple(idx)
        J = sort(I)

        i = linear_index(NoKind, Val(D), I...)

        if !allunique(J)
            push!(check.args, :(isstaticzero(v[$i])))
            continue
        end

        j = linear_index(NoKind, Val(D), J...)

        if i == j
            I != J && push!(check.args, :(isstaticzero(v[$i])))
            continue
        end

        if iseven(inversion_count(I))
            push!(check.args, :(v[$i] === v[$j]))
        else
            push!(check.args, :(v[$i] === -v[$j]))
        end
    end

    return check
end

@generated function isdiagonal(::Val{D}, ::Val{R}, v::NTuple{N,STerm}) where {D,R,N}
    check = Expr(:&&)
    for idx in CartesianIndices(ntuple(_ -> D, Val(R)))
        I = Tuple(idx)
        if issorted(I) && !allequal(I)
            i = linear_index(SymKind, Val(D), I...)
            push!(check.args, :(isstaticzero(v[$i])))
        end
    end
    return check
end

@generated function diagonal_from_symmetric(::Type{Tensor{D,R}}, v::NTuple{N,STerm}) where {D,R,N}
    expr = Expr(:tuple)
    for idx in 1:D
        I = ntuple(_ -> idx, Val(R))
        i = linear_index(SymKind, Val(D), I...)
        push!(expr.args, :(v[$i]))
    end
    return expr
end

@generated function symmetric_components(::Type{Tensor{D,R}}, v::NTuple{N,STerm}) where {D,R,N}
    expr = Expr(:tuple)
    foreach_nondecreasing(Val(D), Val(R)) do I
        i = linear_index(NoKind, Val(D), I...)
        push!(expr.args, :(v[$i]))
    end
    return expr
end

@generated function alternating_components(::Type{Tensor{D,R}}, v::NTuple{N,STerm}) where {D,R,N}
    expr = Expr(:tuple)
    foreach_increasing(Val(D), Val(R)) do I
        i = linear_index(NoKind, Val(D), I...)
        push!(expr.args, :(v[$i]))
    end
    return expr
end
