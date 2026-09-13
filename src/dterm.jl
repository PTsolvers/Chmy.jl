@data HeadImpl begin
    export Call, Comp, Locs, Inds

    Call(Operator)
    Comp(Tuple{Vararg{Int}})
    Locs(Tuple{Vararg{Location}})
    Inds()
end
using .HeadImpl
const Head = HeadImpl.Type

@data DTermImpl begin
    export Literal, Index, Tensor, IdTensor, ZeroTensor, DExpr

    Literal(Number)
    Index(Int)
    struct Tensor
        rank::Int
        name::Symbol
        kind::TensorKind
        uniform::Bool
    end
    ZeroTensor(Int)
    IdTensor(Int)
    struct DExpr
        head::Head
        args::Tuple{Vararg{DTermImpl}}
        rank::Int
        hash::UInt
    end
end
using .DTermImpl
const DTerm = DTermImpl.Type

function DTermImpl.DExpr(head::Head, args::NTuple{N,DTerm}) where {N}
    rank = @match head begin
        Call(op) => tensorrank(op, args)::Int
        _ => 0
    end
    h = hash(head, hash(length(args), hash(:DExpr, UInt(0))))
    for k in eachindex(args)
        h = hash(args[k], h)
    end
    return DExpr(head, args, rank, h)
end

function (==ₛ)(a::DTerm, b::DTerm)
    hash(a) == hash(b) || return false
    @match (a, b) begin
        (Literal(x), Literal(y)) => isequal(x, y)::Bool
        (Tensor(ar, an, ak, au), Tensor(br, bn, bk, bu)) => ar == br &&
                                                            an == bn &&
                                                            ak == bk &&
                                                            au == bu
        (ZeroTensor(x), ZeroTensor(y)) => x == y
        (IdTensor(x), IdTensor(y)) => x == y
        (Index(i), Index(j)) => i == j
        (DExpr(ah, ac), DExpr(bh, bc)) => begin
            ah == bh || return false
            length(ac) == length(bc) || return false
            for k in eachindex(ac)
                ac[k]==ₛbc[k] || return false
            end
            true
        end
        _ => false
    end
end

"""
    isnegof(x::DTerm, y::DTerm)

Check whether canonical scalar expressions have equal bodies and opposite outer signs.
"""
function isnegof(x::DTerm, y::DTerm)::Bool
    if isliteral(x) && isliteral(y)
        return isnegof(value(x), value(y))
    elseif iscall(x) && isunaryminus(x)
        return only(arguments(x)) ==ₛ y
    elseif iscall(y) && isunaryminus(y)
        return x ==ₛ only(arguments(y))
    end
    return false
end
isnegof(x::T, y::S) where {T<:Number,S<:Number} = isequal(x, -y)::Bool

Base.isequal(a::DTerm, b::DTerm) = a==ₛb

function Base.hash(term::DTerm, h::UInt)::UInt
    @match term begin
        Literal(x) => begin
            h = hash(:Literal, h)
            if x isa Union{Int,Rational{Int},Float64}
                return hash(x, h)
            end
            return hash(x, h)::UInt
        end
        Index(i) => hash(i, hash(:Index, h))
        Tensor(rank, name, kind, uniform) => hash(uniform, hash(kind, hash(name, hash(rank, hash(:Tensor, h)))))
        ZeroTensor(rank) => hash(rank, hash(:ZeroTensor, h))
        IdTensor(rank) => hash(rank, hash(:IdTensor, h))
        DExpr(_, _, _, cached) => iszero(h) ? cached : hash(cached, h)
    end
end

isliteral(term::DTerm) = isa_variant(term, Literal)
isindex(term::DTerm) = isa_variant(term, Index)
istensor(term::DTerm) = isa_variant(term, Tensor) ||
                        isa_variant(term, ZeroTensor) ||
                        isa_variant(term, IdTensor)
isexpr(term::DTerm) = isa_variant(term, DExpr)

iscall(term::DTerm) = isa_variant(term, DExpr) && isa_variant(term.head, Call)
iscomp(term::DTerm) = isa_variant(term, DExpr) && isa_variant(term.head, Comp)
islocs(term::DTerm) = isa_variant(term, DExpr) && isa_variant(term.head, Locs)
isinds(term::DTerm) = isa_variant(term, DExpr) && isa_variant(term.head, Inds)

function head(term::DTerm)
    @match term begin
        DExpr(head, _) => head
        _ => throw(ArgumentError("only expressions have a head"))
    end
end

function args(term::DTerm)
    @match term begin
        DExpr(_, args) => args
        _ => throw(ArgumentError("only expressions have children"))
    end
end

function operation(term::DTerm)
    @match term begin
        DExpr(Call(op), _) => op
        _ => throw(ArgumentError("only call expressions have an operation"))
    end
end

function arguments(term::DTerm)
    @match term begin
        DExpr(Call(_), args) => args
        _ => throw(ArgumentError("only call expressions have arguments"))
    end
end

function isuniform(term::DTerm)
    @match term begin
        Literal(_) || Index(_) => true
        Tensor(_, _, _, uniform) => uniform
        ZeroTensor(_) || IdTensor(_) => true
        DExpr(_, args) => all(isuniform, args)
        _ => false
    end
end

function value(term::DTerm)
    @match term begin
        Literal(val) || Index(val) => val
        _ => throw(ArgumentError("only literals and grid indices can have a value"))
    end
end

function isstaticzero(term::DTerm)::Bool
    @match term begin
        Literal(val) => isequal(val, 0)::Bool
        ZeroTensor(_) => true
        _ => false
    end
end

function isstaticone(term::DTerm)::Bool
    @match term begin
        Literal(val) => isequal(val, 1)::Bool
        IdTensor(_) => true
        _ => false
    end
end

function Base.isinteger(term::DTerm)
    @match term begin
        Literal(val) => isinteger(val)
        _ => false
    end
end

function Base.isreal(term::DTerm)
    @match term begin
        Literal(val) => isreal(val)
        _ => false
    end
end

function arity(term::DTerm)
    @match term begin
        DExpr(Call(_), args) => length(args)
        _ => throw(ArgumentError("only function expressions have arity"))
    end
end

function isunary(term::DTerm)
    @match term begin
        DExpr(Call(_), args) => length(args) == 1
        _ => false
    end
end

function isbinary(term::DTerm)
    @match term begin
        DExpr(Call(_), args) => length(args) == 2
        _ => false
    end
end

function argument(term::DTerm)
    @match term begin
        DExpr(Comp(_), (arg,)) => arg
        DExpr(Locs(_), (arg,)) => arg
        DExpr(Inds(), (arg, _...)) => arg
        _ => throw(ArgumentError("only indexing expressions can have a single argument"))
    end
end

function components(term::DTerm)
    @match term begin
        DExpr(Comp(comps), _) => comps
        _ => throw(ArgumentError("only component expression can have components"))
    end
end

function locations(term::DTerm)
    @match term begin
        DExpr(Locs(locs), _) => locs
        _ => throw(ArgumentError("only location expression can have locations"))
    end
end

function indices(term::DTerm)
    @match term begin
        DExpr(Inds(), (_, inds...)) => inds
        _ => throw(ArgumentError("only indexing expression can have indices"))
    end
end

function tensorrank(term::DTerm)::Int
    @match term begin
        Tensor(rank, _, _, _) => rank
        ZeroTensor(rank) || IdTensor(rank) => rank
        DExpr(_, _, rank, _) => rank
        _ => 0
    end
end

function tensorname(term::DTerm)
    @match term begin
        Tensor(_, name, _, _) => name
        _ => throw(ArgumentError("only named tensors have a name"))
    end
end

function tensorkind(term::DTerm)
    @match term begin
        Tensor(_, _, kind, _) => kind
        _ => throw(ArgumentError("only named tensors have a kind"))
    end
end

function Base.getindex(term::DTerm, I::Vararg{Integer,N}) where {N}
    @match term begin
        Literal(_) => return term
        Index(_) => return term
        Tensor(_, _, kind, _) => begin
            if kind == Kind.Sym()
                return makecomp(term, sort(I))
            elseif kind == Kind.Alt()
                allunique(I) || return Literal(0)
                expr = makecomp(term, sort(I))
                return iseven(inversion_count(I)) ? expr : -expr
            elseif kind == Kind.Diag()
                return allequal(I) ? makecomp(term, I) : Literal(0)
            elseif kind == Kind.None()
                return makecomp(term, I)
            else
                error("unreachable")
            end
        end
        ZeroTensor(rank) => begin
            rank == N || throw(ArgumentError("number of subscripts not matching tensor rank"))
            return Literal(0)
        end
        IdTensor(rank) => begin
            rank == N || throw(ArgumentError("number of subscripts not matching tensor rank"))
            return allequal(I) ? Literal(1) : Literal(0)
        end
        _ => return makecomp(term, I)
    end
end
