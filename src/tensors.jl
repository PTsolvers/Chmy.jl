abstract type TensorKind end

struct NoKind <: TensorKind end
struct SymKind <: TensorKind end
struct AltKind <: TensorKind end
struct DiagKind <: TensorKind end
struct ZeroKind <: TensorKind end
struct IdKind <: TensorKind end

abstract type AbstractSTensor{R,K,U} <: STerm end

tensorrank(::AbstractSTensor{R}) where {R} = R
tensorkind(::AbstractSTensor{<:Any,K}) where {K} = K

struct STensor{R,K,U,N} <: AbstractSTensor{R,K,U} end

name(::STensor{<:Any,<:Any,<:Any,N}) where {N} = N

tensorrank(::SIndex) = 0
tensorrank(::SLiteral) = 0
tensorkind(::SLiteral) = NoKind
isuniform(::AbstractSTensor{<:Any,<:Any,true}) = true

"""
    STensor{R,K,U}(name)

Construct a symbolic tensor of rank `R`, kind `K`, and spatial uniformity `U`
with the given name.
"""
STensor{R,K,U}(name::Symbol) where {R,K,U} = STensor{R,K,U,name}()

"""
    STensor{R,K}(name)

Construct a symbolic tensor of rank `R` and kind `K` with the given name.

This shorthand creates a spatially varying tensor; use [`STensor{R,K,true}`](@ref)
or the `SU*` aliases for uniform symbolic tensors.
"""
STensor{R,K}(name::Symbol) where {R,K} = STensor{R,K,false,name}()

"""
    STensor{R}(name)

Construct a symbolic tensor of rank `R` and kind `NoKind` with the given name.
"""
STensor{R}(name::Symbol) where {R} = STensor{R,NoKind,false,name}()

"""
    SUTensor{R,K}(name)

Construct a symbolic spatially uniform tensor of rank `R` and kind `K`.
"""
const SUTensor{R,K} = STensor{R,K,true}

"""
    SSymTensor{R}(name)

Construct a symbolic symmetric tensor of rank `R` with the given name.
"""
const SSymTensor{R} = STensor{R,SymKind,false}

"""
    SUSymTensor{R}(name)

Construct a symbolic spatially uniform symmetric tensor of rank `R`.
"""
const SUSymTensor{R} = STensor{R,SymKind,true}

"""
    SAltTensor{R}(name)

Construct a symbolic alternating tensor of rank `R` with the given name.
"""
const SAltTensor{R} = STensor{R,AltKind,false}

"""
    SUAltTensor{R}(name)

Construct a symbolic spatially uniform alternating tensor of rank `R`.
"""
const SUAltTensor{R} = STensor{R,AltKind,true}

"""
    SDiagTensor{R}(name)

Construct a symbolic diagonal tensor of rank `R` with the given name.
"""
const SDiagTensor{R} = STensor{R,DiagKind,false}

"""
    SUDiagTensor{R}(name)

Construct a symbolic spatially uniform diagonal tensor of rank `R`.
"""
const SUDiagTensor{R} = STensor{R,DiagKind,true}

"""
    SZeroTensor{R}()

Construct the zero tensor of rank `R`.
"""
struct SZeroTensor{R} <: AbstractSTensor{R,ZeroKind,true} end
SZeroTensor{0}() = SLiteral(0)

"""
    SIdTensor{R}()

Construct the identity tensor of rank `R`.
"""
struct SIdTensor{R} <: AbstractSTensor{R,IdKind,true} end
SIdTensor{0}() = SLiteral(1)

tensorrank(::SZeroTensor{R}) where {R} = R
tensorrank(::SIdTensor{R}) where {R} = R

"""
    SScalar(name)

Construct a symbolic scalar with the given name.
"""
const SScalar = STensor{0,NoKind,false}

"""
    SUScalar(name)

Construct a symbolic spatially uniform scalar with the given name.
"""
const SUScalar = STensor{0,NoKind,true}

"""
    SVec(name)

Construct a symbolic vector with the given name.
"""
const SVec = STensor{1,NoKind,false}

"""
    SUVec(name)

Construct a symbolic spatially uniform vector with the given name.
"""
const SUVec = STensor{1,NoKind,true}

const IntegerOrSLiteral = Union{Integer,SLiteral}

sallunique(I::Tuple{SLiteral}) = true
function sallunique(I)
    I[1] === I[2] && return false
    return sallunique(Base.tail(I))
end
function sinversion_count(I::NTuple{N,SLiteral}) where {N}
    return inversion_count(map(value, I))
end

Base.getindex(::SZeroTensor{R}, I::Vararg{IntegerOrSLiteral,R}) where {R} = SLiteral(0)
Base.getindex(t::SIdTensor{R}, I::Vararg{IntegerOrSLiteral,R}) where {R} = Base.getindex(t, tuplemap(STerm, I)...)
Base.getindex(::SIdTensor{R}, I::Vararg{SLiteral,R}) where {R} = all(x -> x === I[1], I) ? SLiteral(1) : SLiteral(0)
Base.getindex(t::STensor{0}, ::Vararg{IntegerOrSLiteral,0}) = t
Base.getindex(t::STensor{R}, I::Vararg{IntegerOrSLiteral,R}) where {R} = Base.getindex(t, tuplemap(STerm, I)...)
function Base.getindex(t::SDiagTensor{R,D}, I::Vararg{SLiteral,R}) where {R,D}
    all(x -> x === I[1], I) || return SLiteral(0)
    return SExpr(Comp(), t, I...)
end
function Base.getindex(t::SSymTensor{R}, I::Vararg{SLiteral,R}) where {R}
    J = ssort(I; lt=isless_lex)
    return SExpr(Comp(), t, J...)
end
function Base.getindex(t::SAltTensor{R}, I::Vararg{SLiteral,R}) where {R}
    J = ssort(I; lt=isless_lex)
    sallunique(J) || return SLiteral(0)
    v = SExpr(Comp(), t, J...)
    return iseven(sinversion_count(I)) ? v : -v
end
function Base.getindex(t::STensor{R}, I::Vararg{SLiteral,R}) where {R}
    return SExpr(Comp(), t, I...)
end

"""
    tensorrank(expr)

Return the rank of the tensor represented by the symbolic expression `expr`.
"""
function tensorrank end

# tensor rank of operation depends on the operation
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
tensorrank(::SRef{:/}, a, b)    = tensorrank(a)
tensorrank(::SRef{://}, a, b)   = tensorrank(a)
tensorrank(::SRef{:÷}, a, b)    = tensorrank(a)
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
ncomponents(::Type{ZeroKind}, ::Val{D}, ::Val{R}) where {D,R} = 0
ncomponents(::Type{IdKind}, ::Val{D}, ::Val{R}) where {D,R} = 0

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
const Vec{D} = Tensor{D,1,NoKind}

const ZeroTensor{D,R} = Tensor{D,R,ZeroKind}
const IdTensor{D,R} = Tensor{D,R,IdKind}

isidentity(::STerm) = false
isidentity(::SIdTensor) = true

isstaticzero(::SZeroTensor) = true

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

Base.getindex(t::Tensor{D,R}, I::Vararg{IntegerOrSLiteral,R}) where {D,R} = t[tuplemap(STerm, I)...]

function literal_int(i::SLiteral)
    isstaticinteger(i) || throw(ArgumentError("tensor indices must be integer-valued SLiterals"))
    return Int(value(i))
end

function Base.getindex(t::Tensor{D,R,K}, I::Vararg{SLiteral,R}) where {D,R,K}
    return t.components[linear_index(K, Val(D), map(literal_int, I)...)]
end
function Base.getindex(t::DiagTensor{D,R}, I::Vararg{SLiteral,R}) where {D,R}
    J = map(literal_int, I)
    all(==(J[1]), J) || return SLiteral(0)
    return t.components[J[1]]
end
function Base.getindex(t::AltTensor{D,R}, I::Vararg{SLiteral,R}) where {D,R}
    J = map(literal_int, I)
    allunique(J) || return SLiteral(0)
    K = sort(J)
    v = t.components[linear_index(AltKind, Val(D), K...)]
    return iseven(inversion_count(J)) ? v : -v
end
function Base.getindex(::ZeroTensor{D,R}, I::Vararg{SLiteral,R}) where {D,R}
    return SLiteral(0)
end
function Base.getindex(::IdTensor{D,R}, I::Vararg{SLiteral,R}) where {D,R}
    J = map(literal_int, I)
    return all(==(J[1]), J) ? SLiteral(1) : SLiteral(0)
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
    M == N || error("expected $N components to construct rank-$R $(K) Tensor, got $M")
    construct_tensor(Tensor{D,R,K}, data)
end
ZeroTensor{D,R}() where {D,R} = Tensor{D,R,ZeroKind,Tuple{}}(())
IdTensor{D,R}() where {D,R} = Tensor{D,R,IdKind,Tuple{}}(())

"""
    Tensor{D}(s::STensor)

Construct a dimension-`D` component representation of symbolic tensor `s`.
The result has the same rank and the symmetries as `s`.
"""
Tensor{D}(t::Tensor{D}) where {D} = t
Tensor{D}(s::STensor{0}) where {D} = s
Tensor{D}(::SZeroTensor{R}) where {D,R} = ZeroTensor{D,R}()
Tensor{D}(::SIdTensor{R}) where {D,R} = IdTensor{D,R}()
@generated function Tensor{D}(s::STensor{R,K,U}) where {D,R,K,U}
    ex = Expr(:call, :(Tensor{$D,$R,$K}))
    comp_expr(I) = :(s[$(map(i -> :(SLiteral($i)), I)...)])
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
function Tensor{D}(expr::SExpr{Call}) where {D}
    args = tuplemap(Tensor{D}, arguments(expr))
    op = operation(expr)
    return Tensor{D}(op, args...)
end
function Tensor{D}(expr::SExpr{Comp}) where {D}
    arg = Tensor{D}(argument(expr))
    return arg[tuplemap(Tensor{D}, indices(expr))...]
end
function Tensor{D}(expr::SExpr{Loc}) where {D}
    arg = Tensor{D}(argument(expr))
    return arg[location(expr)...]
end
function Tensor{D}(expr::SExpr{Ind}) where {D}
    arg = Tensor{D}(argument(expr))
    return arg[tuplemap(Tensor{D}, indices(expr))...]
end
Tensor{D}(s::SRef) where {D} = s
Tensor{D}(sf::SFun) where {D} = sf
function Tensor{D}(::SRef{:broadcasted}, op::SFun, args::Vararg{Any,N}) where {D,N}
    all(arg -> tensorrank(arg) == 0, args) && return op.f(args...)
    return Base.Broadcast.broadcasted(op.f, args...)
end
function Tensor{D}(::SRef{:broadcasted}, op::SRef{F}, args::Vararg{Any,N}) where {D,F,N}
    all(arg -> tensorrank(arg) == 0, args) && return Tensor{D}(op, args...)
    return Base.Broadcast.broadcasted(getfield(Base, F), args...)
end
@generated function Tensor{D}(::SRef{F}, args::Vararg{Any,N}) where {D,F,N}
    ex = Expr(:call, F)
    for arg in args
        argi = arg.instance
        push!(ex.args, :($argi))
    end
    return ex
end
function Tensor{D}(sf::SFun, args::Vararg{Any,N}) where {D,N}
    return sf.f(tuplemap(Tensor{D}, args)...)
end
function Tensor{D}(op::STerm, args::Vararg{Any,N}) where {D,N}
    return evaluate(SExpr(Call(), op, args...))
end
Tensor{D}(s::SLiteral) where {D} = s
Tensor{D}(expr::SExpr) where {D} = SExpr(head(expr), tuplemap(Tensor{D}, children(expr))...)

# Materialize `value` into the requested tensor kind without collapsing it back
# to `ZeroTensor`, `DiagTensor`, etc. Boundary scalarization uses this when the
# left-hand side determines the component layout and the right-hand side may be a
# narrower tensor such as `SZeroTensor` or `SIdTensor`.
function tensor_with_kind(::Type{Tensor{D,R,K}}, value) where {D,R,K}
    t = Tensor{D}(value)
    tensorrank(t) == R || throw(ArgumentError("tensor rank $(tensorrank(t)) does not match requested rank $R"))
    components = tensor_kind_components(Tensor{D,R,K}, t)
    return Tensor{D,R,K,typeof(components)}(components)
end

@generated function tensor_kind_components(::Type{Tensor{D,R,K}}, t) where {D,R,K}
    K <: ZeroKind && return :(throw(ArgumentError("ZeroKind tensors do not store scalar components")))
    K <: IdKind && return :(throw(ArgumentError("IdKind tensors do not store scalar components")))

    comps = Expr(:tuple)
    comp_expr(I) = :(t[$(map(i -> :(SLiteral($i)), I)...)])
    if K <: NoKind
        for idx in CartesianIndices(ntuple(_ -> D, Val(R)))
            push!(comps.args, comp_expr(Tuple(idx)))
        end
    elseif K <: SymKind
        foreach_nondecreasing(Val(D), Val(R)) do I
            push!(comps.args, comp_expr(I))
        end
    elseif K <: AltKind
        foreach_increasing(Val(D), Val(R)) do I
            push!(comps.args, comp_expr(I))
        end
    elseif K <: DiagKind
        for i in 1:D
            push!(comps.args, comp_expr(ntuple(_ -> i, Val(R))))
        end
    else
        error("unsupported tensor kind $K")
    end
    return comps
end

# construct the most specific tensor type based on the symmetries of the components
function construct_tensor(::Type{Tensor{D,R,DiagKind}}, data::NTuple{N,STerm}) where {D,R,N}
    all(isstaticzero, data) && return ZeroTensor{D,R}()
    all(isstaticone, data) && return IdTensor{D,R}()
    return Tensor{D,R,DiagKind,typeof(data)}(data)
end
function construct_tensor(::Type{Tensor{D,R,AltKind}}, data::NTuple{N,STerm}) where {D,R,N}
    all(isstaticzero, data) && return ZeroTensor{D,R}()
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
    all(isstaticzero, data) && return ZeroTensor{D,R}()
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
    R < 2 && return :(false)
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
    R < 2 && return :(false)
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
    R < 2 && return :(false)
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
