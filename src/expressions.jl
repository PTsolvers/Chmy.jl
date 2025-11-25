abstract type STerm end

isexpr(::STerm) = false

abstract type SExprHead end

struct Call <: SExprHead end

struct Ind <: SExprHead end

struct Loc <: SExprHead end

struct SUniform{Value} <: STerm end
SUniform(value) = SUniform{value}()

struct Tag{Name,Inds} <: STerm end
Tag(name::Symbol, inds::Vararg{Integer}) = Tag{name,inds}()

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

struct SIndex{I} <: STerm end
SIndex(I::Union{Symbol,Integer}) = SIndex{I}()

struct SExpr{H,C} <: STerm
    head::H
    children::C
end

SExpr(head::SExprHead, children::Vararg{STerm}) = SExpr(head, children)

isexpr(::SExpr) = true

head(expr::SExpr) = expr.head
children(expr::SExpr) = expr.children

iscall(expr::SExpr) = expr.head isa Call
operation(expr::SExpr{Call}) = first(expr.children)
arguments(expr::SExpr{Call}) = Base.tail(expr.children)

isind(expr::SExpr) = expr.head isa Ind
argument(expr::SExpr{Ind}) = first(expr.children)
indices(expr::SExpr{Ind}) = Base.tail(expr.children)

isloc(expr::SExpr) = expr.head isa Loc
argument(expr::SExpr{Loc}) = first(expr.children)
location(expr::SExpr{Loc}) = Base.tail(expr.children)

Base.:+(arg::STerm) = arg

const UNARY_OPERATORS = (:-, :sqrt, :inv, :abs,
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

for op in UNARY_OPERATORS
    @eval Base.$op(arg::STerm) = SExpr(Call(), SRef($(Meta.quot(op))), arg)
end

const BINARY_OPERATORS = (:-, :/, ://, :÷, :^, :isless, :isequal, :<, :<=, :>, :>=, :(==), :!=, :&, :|, :xor)

for op in BINARY_OPERATORS
    @eval begin
        Base.$op(arg1::STerm, arg2::STerm) = SExpr(Call(), SRef($(Meta.quot(op))), arg1, arg2)
        Base.$op(arg1::STerm, arg2::Number) = $op(arg1, SUniform(arg2))
        Base.$op(arg1::Number, arg2::STerm) = $op(SUniform(arg1), arg2)
    end
end

const MULTIARY_OPERATORS = (:+, :*, :max, :min)

for op in MULTIARY_OPERATORS
    @eval begin
        Base.$op(args::Vararg{STerm}) = SExpr(Call(), SRef($(Meta.quot(op))), args...)
        Base.$op(a::Number, b::STerm) = $op(SUniform(a), b)
        Base.$op(a::STerm, b::Number) = $op(a, SUniform(b))
    end
end

Base.ifelse(cond::STerm, x::STerm, y::STerm) = SExpr(Call(), SRef(:ifelse), cond, x, y)
Base.ifelse(cond::STerm, x::Number, y::STerm) = SExpr(Call(), SRef(:ifelse), cond, SUniform(x), y)
Base.ifelse(cond::STerm, x::STerm, y::Number) = SExpr(Call(), SRef(:ifelse), cond, x, SUniform(y))
Base.ifelse(cond::STerm, x::Number, y::Number) = SExpr(Call(), SRef(:ifelse), cond, SUniform(x), SUniform(y))

Base.getindex(t::STerm, inds::Vararg{STerm}) = SExpr(Ind(), t, inds...)
Base.getindex(s::SUniform, ::Vararg{STerm}) = s
