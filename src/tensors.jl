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

STensor{R,K}(name::Symbol) where {R,K} = STensor{R,K,name}()
STensor{R}(name::Symbol) where {R} = STensor{R,NoKind,name}()

const SSymTensor{R}  = STensor{R,SymKind}
const SAltTensor{R}  = STensor{R,AltKind}
const SDiagTensor{R} = STensor{R,DiagKind}

struct SZeroTensor{R} <: STerm end
SZeroTensor{0}() = SUniform(0)

struct SIdTensor{R} <: STerm end
SIdTensor{0}() = SUniform(1)

tensorrank(::SZeroTensor{R}) where {R} = R
tensorrank(::SIdTensor{R}) where {R} = R

const SScalar = STensor{0,NoKind}
const SVec = STensor{1,NoKind}

const IntegerOrSUniform = Union{Integer,SUniform}

function Base.getindex(t::STensor{R}, inds::Vararg{IntegerOrSUniform,R}) where {R}
    SExpr(Comp(), t, map(STerm, inds)...)
end

Base.getindex(s::SScalar) = s

tensorrank(expr::SExpr{Call}) = tensorrank(operation(expr), arguments(expr)...)
tensorrank(::SExpr{Comp}) = 0
tensorrank(::SExpr{Loc}) = 0
tensorrank(::SExpr{Ind}) = 0

tensorrank(::SFun, args...) = 0
tensorrank(::SRef, args...) = 0

tensorrank(::SRef{:adjoint}, t) = tensorrank(t)
tensorrank(::SRef{:broadcasted}, op, args...) = maximum(map(tensorrank, args))

tensorrank(::SRef{:+}, args...) = tensorrank(last(args))
tensorrank(::SRef{:*}, args...) = tensorrank(last(args))
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

struct Tensor{R,D,K,C}
    components::C
end

tensorrank(::Tensor{R}) where {R} = R
dimensions(::Tensor{<:Any,D}) where {D} = D
tensorkind(::Tensor{<:Any,<:Any,K}) where {K} = K

tensorrank(::Type{Tensor{R}}) where {R} = R
dimensions(::Type{Tensor{<:Any,D}}) where {D} = D
tensorkind(::Type{Tensor{<:Any,<:Any,K}}) where {K} = K

Base.length(t::Tensor) = length(t.components)

ncomponents(t::Tensor) = ncomponents(typeof(t))
ncomponents(tt::Type{Tensor}) = ncomponents(tensorkind(tt), Val(tensorrank(tt)), Val(dimensions(tt)))

ncomponents(::Type{NoKind}, ::Val{R}, ::Val{D}) where {R,D} = D^R
ncomponents(::Type{SymKind}, ::Val{R}, ::Val{D}) where {R,D} = binomial(D + R - 1, R)
ncomponents(::Type{AltKind}, ::Val{R}, ::Val{D}) where {R,D} = binomial(D, R)
ncomponents(::Type{DiagKind}, ::Val{R}, ::Val{D}) where {R,D} = D

const SymTensor{R,D}  = Tensor{R,D,SymKind}
const AltTensor{R,D}  = Tensor{R,D,AltKind}
const DiagTensor{R,D} = Tensor{R,D,DiagKind}

const Vec{D} = Tensor{1,D}

isidentity(::STerm) = false
isidentity(::SIdTensor) = true

isstaticzero(::SZeroTensor) = true
isstaticzero(t::Tensor) = all(isstaticzero, t.components)

linear_index(t::Tensor, I::Vararg{Int}) = linear_index(typeof(t), I...)

linear_index(::Type{<:Tensor{R,D,K}}, I::Vararg{Int,R}) where {R,D,K} = linear_index(K, Val(D), I...)

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

Base.getindex(t::Tensor{R,D,K}, I::Vararg{Int,R}) where {R,D,K} = t.components[linear_index(K, Val(D), I...)]

function Base.getindex(t::DiagTensor{R,D}, I::Vararg{Int,R}) where {R,D}
    all(==(I[1]), I) || return SUniform(0)
    return t.components[I[1]]
end

function Base.getindex(t::AltTensor{R,D}, I::Vararg{Int,R}) where {R,D}
    allunique(I) || return SUniform(0)
    J = sort(I)
    v = t.components[linear_index(AltKind, Val(D), J...)]
    return iseven(inversion_count(I)) ? v : -v
end

Base.getindex(::SZeroTensor{R}, I::Vararg{Int,R}) where {R} = SUniform(0)

Base.getindex(::SIdTensor{R}, I::Vararg{Int,R}) where {R} = all(==(I[1]), I) ? SUniform(1) : SUniform(0)

Vec(data::Vararg{STerm,M}) where {M} = Vec{M}(data...)

Tensor{O,D}(data::Vararg{STerm,M}) where {O,D,M} = Tensor{O,D,NoKind}(data...)

function Tensor{R,D,K}(data::Vararg{STerm,M}) where {R,D,K,M}
    N = ncomponents(K, Val(R), Val(D))
    M == N || error("expected $N components to construct order-$R $(K) Tensor, got $M")
    _construct_tensor(Tensor{R,D,K}, data)
end

function _construct_tensor(::Type{Tensor{R,D,DiagKind}}, data::NTuple{N,STerm}) where {R,D,N}
    all(isstaticzero, data) && return SZeroTensor{R}()
    all(isstaticone, data) && return SIdTensor{R}()
    return Tensor{R,D,DiagKind,typeof(data)}(data)
end

function _construct_tensor(::Type{Tensor{R,D,AltKind}}, data::NTuple{N,STerm}) where {R,D,N}
    all(isstaticzero, data) && return SZeroTensor{R}()
    return Tensor{R,D,AltKind,typeof(data)}(data)
end

function _construct_tensor(::Type{Tensor{R,D,SymKind}}, data::NTuple{N,STerm}) where {R,D,N}
    if _is_diagonal(Val(R), Val(D), data)
        diag_comps = _diagonal_from_symmetric(Tensor{R,D}, data)
        return _construct_tensor(Tensor{R,D,DiagKind}, diag_comps)
    end
    return Tensor{R,D,SymKind,typeof(data)}(data)
end

function _construct_tensor(::Type{Tensor{R,D,NoKind}}, data::NTuple{N,STerm}) where {R,D,N}
    if _is_symmetric(Val(R), Val(D), data)
        sym_comps = _symmetric_components(Tensor{R,D}, data)
        return _construct_tensor(Tensor{R,D,SymKind}, sym_comps)
    end

    if _is_alternating(Val(R), Val(D), data)
        alt_comps = _alternating_components(Tensor{R,D}, data)
        return _construct_tensor(Tensor{R,D,AltKind}, alt_comps)
    end

    return Tensor{R,D,NoKind,typeof(data)}(data)
end

@generated function _is_symmetric(::Val{R}, ::Val{D}, v::NTuple{N,STerm}) where {R,D,N}
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

@generated function _is_alternating(::Val{R}, ::Val{D}, v::NTuple{N,STerm}) where {R,D,N}
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

@generated function _is_diagonal(::Val{R}, ::Val{D}, v::NTuple{N,STerm}) where {R,D,N}
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

@generated function _diagonal_from_symmetric(::Type{Tensor{R,D}}, v::NTuple{N,STerm}) where {R,D,N}
    expr = Expr(:tuple)
    for idx in 1:D
        I = ntuple(_ -> idx, Val(R))
        i = linear_index(SymKind, Val(D), I...)
        push!(expr.args, :(v[$i]))
    end
    return expr
end

@generated function _symmetric_components(::Type{Tensor{R,D}}, v::NTuple{N,STerm}) where {R,D,N}
    expr = Expr(:tuple)
    foreach_nondecreasing(Val(D), Val(R)) do I
        i = linear_index(NoKind, Val(D), I...)
        push!(expr.args, :(v[$i]))
    end
    return expr
end

@generated function _alternating_components(::Type{Tensor{R,D}}, v::NTuple{N,STerm}) where {R,D,N}
    expr = Expr(:tuple)
    foreach_increasing(Val(D), Val(R)) do I
        i = linear_index(NoKind, Val(D), I...)
        push!(expr.args, :(v[$i]))
    end
    return expr
end
