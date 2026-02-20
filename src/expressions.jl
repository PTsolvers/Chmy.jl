abstract type STerm end

isexpr(::STerm) = false
iscall(::STerm) = false
iscomp(::STerm) = false
isind(::STerm)  = false
isloc(::STerm)  = false

isstaticzero(::STerm) = false
isstaticone(::STerm) = false

abstract type SExprHead end

struct Call <: SExprHead end

struct Comp <: SExprHead end

struct Loc <: SExprHead end

struct Ind <: SExprHead end

struct SUniform{Value} <: STerm end
SUniform(value) = SUniform{value}()

function SUniform(value::Real)
    isbits(value) || error("value must be isbits")
    if isinteger(value)
        value = Int(value)
    end
    sgn = value >= zero(value)
    value = abs(value)
    if isone(value)
        value = 1
    end
    return sgn ? SUniform{value}() : -SUniform{value}()
end

SUniform(::StaticCoeff{Value}) where {Value} = SUniform(Value)

isstaticuniform(s::STerm) = s isa SUniform

value(::SUniform{Value}) where {Value} = Value

isstaticzero(s::SUniform) = iszero(value(s))
isstaticone(s::SUniform) = isone(value(s))

isstaticinteger(s::SUniform) = isinteger(value(s))

struct SRef{F} <: STerm end
SRef(f::Symbol) = SRef{f}()

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

function (f::SFun)(args::Vararg{STerm})
    SExpr(Call(), f, args...)
end

struct SIndex{I} <: STerm end
SIndex(I::Integer) = SIndex{I}()

struct SExpr{H,C} <: STerm
    head::H
    children::C
end

SExpr(head::SExprHead, children::Vararg{STerm}) = SExpr(head, children)

SExpr(::Call, ::SRef{:*}, x::STerm) = x
SExpr(::Call, ::SRef{:+}, x::STerm) = x

isexpr(::SExpr) = true

head(expr::SExpr) = expr.head
children(expr::SExpr) = expr.children

iscall(expr::SExpr) = expr.head isa Call
operation(expr::SExpr{Call}) = first(expr.children)
arguments(expr::SExpr{Call}) = Base.tail(expr.children)
arity(expr::SExpr{Call}) = length(arguments(expr))

iscomp(expr::SExpr) = expr.head isa Comp
argument(expr::SExpr{Comp}) = first(expr.children)
indices(expr::SExpr{Comp}) = Base.tail(expr.children)

isind(expr::SExpr) = expr.head isa Ind
argument(expr::SExpr{Ind}) = first(expr.children)
indices(expr::SExpr{Ind}) = Base.tail(expr.children)

isloc(expr::SExpr) = expr.head isa Loc
argument(expr::SExpr{Loc}) = first(expr.children)
location(expr::SExpr{Loc}) = Base.tail(expr.children)

# indexing with static expressions
Base.getindex(t::STerm, inds::Vararg{STerm}) = SExpr(Ind(), t, inds...)

# for uniforms s[inds...] = s
Base.getindex(s::SUniform, ::Vararg{STerm}) = s

# conversion to STerm
STerm(v::STerm) = v
STerm(v) = isbits(v) ? SUniform(v) : error("value must be isbits")

# TermInterfaces.jl maketerm implementation
maketerm(::Type{SExpr}, head, children, ::Nothing) = SExpr(head, children...)
