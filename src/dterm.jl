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

    Literal(Any)
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
    end
end
using .DTermImpl
const DTerm = DTermImpl.Type

function (==ₛ)(a::DTerm, b::DTerm)
    @match (a, b) begin
        (Literal(x), Literal(y)) => (x == y)::Bool
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
                ac[k] ==ₛ bc[k] || return false
            end
            true
        end
        _ => false
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

isstaticzero(term::DTerm)::Bool = isa_variant(term, ZeroTensor)
isstaticone(term::DTerm)::Bool = isa_variant(term, IdTensor)

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
        DExpr(Comp(_), (arg, )) => arg
        DExpr(Locs(_), (arg, )) => arg
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
        DExpr(Comp(comps), _) => comps
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
        DExpr(Call(op), args) => tensorrank(op, args)::Int
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
