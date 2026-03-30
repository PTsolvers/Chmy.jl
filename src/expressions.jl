abstract type STerm end

isexpr(::STerm) = false
iscall(::STerm) = false
iscomp(::STerm) = false
isind(::STerm)  = false
isloc(::STerm)  = false
isnode(::STerm) = false

isstaticzero(::STerm) = false
isstaticone(::STerm) = false

"""
    isuniform(term)

Return `true` if `term` is spatially uniform, so location and grid indexing do
not change its value.
"""
isuniform(::STerm) = false

abstract type SExprHead end

struct Call <: SExprHead end

struct Comp <: SExprHead end

struct Loc <: SExprHead end

struct Ind <: SExprHead end

struct Node <: SExprHead end

"""
    SLiteral(value)

Construct a symbolic compile-time literal value.
"""
struct SLiteral{Value} <: STerm end
SLiteral(value) = SLiteral{value}()

function SLiteral(value::Real)
    isbits(value) || error("value must be isbits")
    if isinteger(value)
        value = Int(value)
    end
    sgn = value >= zero(value)
    value = abs(value)
    if isone(value)
        value = 1
    end
    return sgn ? SLiteral{value}() : -SLiteral{value}()
end

SLiteral(::StaticCoeff{Value}) where {Value} = SLiteral(Value)

isstaticliteral(s::STerm) = s isa SLiteral
isuniform(::SLiteral) = true

value(::SLiteral{Value}) where {Value} = Value

isstaticzero(s::SLiteral) = iszero(value(s))
isstaticone(s::SLiteral) = isone(value(s))

isstaticinteger(s::SLiteral) = isinteger(value(s))

struct SRef{F} <: STerm end
SRef(f::Symbol) = SRef{f}()
isuniform(::SRef) = true

struct SFun{F} <: STerm
    f::F
    function SFun(f)
        if Base.issingletontype(typeof(f))
            return new{typeof(f)}(f)
        else
            error("function must be a singleton type")
        end
    end
end
isuniform(::SFun) = true

function (f::SFun)(args::Vararg{STerm})
    SExpr(Call(), f, args...)
end

struct SIndex{I} <: STerm end
isuniform(::SIndex) = true

"""
    SIndex(i)

Create a static index for spatial dimension i.
"""
SIndex(I::Integer) = SIndex{I}()

struct SExpr{H,C} <: STerm
    head::H
    children::C
end

SExpr(head::SExprHead, children::Vararg{STerm}) = SExpr(head, children)

"""
    node(term)

Wrap `term` in a `Node` expression.

`Node` keeps the wrapped subtree structurally intact during expression evaluation.

```jldoctest
julia> using Chmy

julia> a, b = SScalar(:a), SScalar(:b)
(a, b)

julia> a + (a + b)
StaticExpression:
 2a + b

julia> a + node(a + b)
StaticExpression:
 a + (a + b)
```
"""
node(term::SExpr{Node}) = term
node(term::SLiteral) = term
node(term::STerm) = SExpr(Node(), term)
node(term) = node(STerm(term))

"""
    node_unwrap(term)

Recursively remove all `Node` wrappers from `term` and symbolically evaluate the
rebuilt expression.

This is the inverse of [`node`](@ref) for expression trees: once the wrappers
are gone, ordinary Chmy expression construction rules are re-applied, so the
result may be simplified or re-canonicalized.
"""
node_unwrap(term::STerm) = term
node_unwrap(expr::SExpr{Node}) = node_unwrap(argument(expr))
function node_unwrap(expr::SExpr)
    rebuilt = SExpr(head(expr), tuplemap(node_unwrap, children(expr))...)
    return evaluate(rebuilt)
end

SExpr(::Call, ::SRef{:*}, x::STerm) = x
SExpr(::Call, ::SRef{:+}, x::STerm) = x

function check_tensor_ranks(f::SRef, args...)
    if any(x -> tensorrank(x) != 0, args)
        throw(ArgumentError("'$f' can only be applied to scalar terms, consider using broadcasting"))
    end
end

function SExpr(::Call, f::SRef, args::STerm...)
    check_tensor_ranks(f, args...)
    return SExpr(Call(), (f, args...))
end

isexpr(::SExpr) = true

head(expr::SExpr) = expr.head
children(expr::SExpr) = expr.children
isuniform(expr::SExpr) = all(isuniform, children(expr))

iscall(expr::SExpr) = expr.head isa Call
operation(expr::SExpr{Call}) = first(expr.children)
arguments(expr::SExpr{Call}) = Base.tail(expr.children)
arity(expr::SExpr{Call}) = length(arguments(expr))

isnode(expr::SExpr) = expr.head isa Node
argument(expr::SExpr{Node}) = only(expr.children)

iscomp(expr::SExpr) = expr.head isa Comp
argument(expr::SExpr{Comp}) = first(expr.children)
indices(expr::SExpr{Comp}) = Base.tail(expr.children)

isind(expr::SExpr) = expr.head isa Ind
argument(expr::SExpr{Ind}) = first(expr.children)
indices(expr::SExpr{Ind}) = Base.tail(expr.children)

isloc(expr::SExpr) = expr.head isa Loc
argument(expr::SExpr{Loc}) = first(expr.children)
location(expr::SExpr{Loc}) = Base.tail(expr.children)

# Zero-argument indexing is an identity for symbolic terms.
Base.getindex(term::STerm) = term

# conversion to STerm
STerm(v::STerm) = v
STerm(v) = isbits(v) ? SLiteral(v) : error("value must be isbits")

# TermInterfaces.jl maketerm implementation
maketerm(::Type{SExpr}, head, children, ::Nothing) = SExpr(head, children...)
