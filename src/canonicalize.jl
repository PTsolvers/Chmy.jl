# Static evaluation

isstatic(::STerm) = false
isstatic(::SUniform) = true
isstatic(::SRef) = true
isstatic(::SFun) = true
isstatic(expr::SExpr) = all(isstatic, children(expr))

struct SEvalRule <: AbstractRule end

Base.@assume_effects :foldable function (::SEvalRule)(expr::SExpr)
    isstatic(expr) && return SUniform(compute(expr))
    return expr
end

seval(expr::STerm) = Postwalk(SEvalRule())(expr)

# Lexicographic ordering of terms for deterministic canonicalization

termrank(::SIndex)   = 1
termrank(::STensor)  = 2
termrank(::SRef)     = 3
termrank(::SFun)     = 4
termrank(::SUniform) = 5
termrank(::SExpr)    = 6

headrank(::Call) = 1
headrank(::Comp) = 2
headrank(::Ind) = 3
headrank(::Loc) = 4
headrank(expr::SExpr) = headrank(head(expr))

const SAtom = Union{SIndex,STensor,SRef,SFun,SUniform}

atomrank(::SIndex)   = 1
atomrank(::STensor)  = 2
atomrank(::SRef)     = 3
atomrank(::SFun)     = 4
atomrank(::SUniform) = 5

isless_atom(x::SAtom, y::SAtom)                   = isless(atomrank(x), atomrank(y))
isless_atom(::SIndex{I}, ::SIndex{J}) where {I,J} = isless(I, J)
isless_atom(::SRef{F1}, ::SRef{F2}) where {F1,F2} = isless(F1, F2)
isless_atom(x::SFun, y::SFun)                     = isless(nameof(x.f), nameof(y.f))
isless_atom(x::SUniform, y::SUniform)             = isless(value(x), value(y))
function isless_atom(x::STensor, y::STensor)
    rx = tensorrank(x)
    ry = tensorrank(y)
    rx == ry || return isless(rx, ry)

    nx = name(x)
    ny = name(y)
    nx == ny || return isless(nx, ny)

    if x !== y
        throw(ArgumentError("tensors with the same name must have the same rank and kind"))
    end

    return false
end

isless_tuple(::Tuple{}, ::Tuple{}) = false
isless_tuple(::Tuple{}, ::Tuple{Any,Vararg}) = true
isless_tuple(::Tuple{Any,Vararg}, ::Tuple{}) = false
function isless_tuple(xs::Tuple{X,Vararg}, ys::Tuple{Y,Vararg}) where {X,Y}
    xh = first(xs)
    yh = first(ys)
    if xh === yh
        return isless_tuple(Base.tail(xs), Base.tail(ys))
    end
    return isless_lex(xh, yh)
end

function isless_expr(x::SExpr, y::SExpr)
    hx = headrank(x)
    hy = headrank(y)
    hx == hy || return hx < hy
    return isless_tuple(children(x), children(y))
end

function isless_lex(x::STerm, y::STerm)
    x === y && return false

    rx = termrank(x)
    ry = termrank(y)
    rx == ry || return rx < ry

    if isexpr(x) && isexpr(y)
        return isless_expr(x, y)
    end
    return isless_atom(x, y)
end

# Static sorting of tuples of singleton types
_ssort_impl(args::Tuple, lt, by) = (sort!(collect(args); lt, by)...,)
@generated function ssort(args::Tuple; lt=isless, by=identity)
    sorted = _ssort_impl(args.instance, lt.instance, by.instance)
    return :($sorted)
end

# Monomial representation of products for canonicalization

struct Monomial{S,B}
    coeff::S
    powers::B
end

Monomial(term::STerm) = Monomial(StaticCoeff(1), Binding(term => SUniform(1)))

function Monomial(expr::SExpr{Call})
    coeff, powers = collect_powers!(expr)
    kv = ssort((pairs(powers)...,); lt=isless_lex, by=first)
    return Monomial(coeff, Binding(kv...))
end

isconstant(monomial::Monomial) = length(monomial.powers) == 0

Base.@assume_effects :foldable function collect_powers!(term::SExpr{Call},
                                                        coeff::StaticCoeff=StaticCoeff(1),
                                                        binding::Binding=Binding(),
                                                        power::STerm=SUniform(1))
    op = operation(term)
    if op === SRef(:*)
        # Flatten the tree and accumulate powers
        for arg in arguments(term)
            coeff, binding = collect_powers!(arg, coeff, binding, power)
        end
    elseif op === SRef(:/)
        # a / b is treated as a * b^-1, so powers in the denominator are negated
        num, den = arguments(term)
        coeff, binding = collect_powers!(num, coeff, binding, power)
        coeff, binding = collect_powers!(den, coeff, binding, -power)
    elseif op === SRef(:inv)
        # inv(a) is treated as a^-1, so powers are negated
        arg = only(arguments(term))
        coeff, binding = collect_powers!(arg, coeff, binding, -power)
    elseif isunaryminus(term) && isstatic(power)
        # Fold an unary minus into the coefficient for odd integer powers
        p = compute(power)
        if isinteger(p)
            isodd(p) && (coeff = -coeff) # (-a)^(2k+1) = -(a^(2k+1))
            coeff, binding = collect_powers!(only(arguments(term)), coeff, binding, power)
        end
    elseif op === SRef(:^)
        # Fold nested powers by multiplying exponents
        base, exp = arguments(term)
        coeff, binding = collect_powers!(base, coeff, binding, seval(power * exp))
    else
        # Non-product call: store or update its accumulated power in the binding
        if haskey(binding, term)
            binding = push(binding, term => seval(binding[term] + power))
        else
            binding = push(binding, term => power)
        end
    end
    return coeff, binding
end

# Fully static uniform literals can be folded into coeff at compile time
Base.@assume_effects :foldable function collect_powers!(term::SUniform, coeff, binding, power)
    base = value(term)
    pow = compute(power)
    # Preserve exact division of integers by promoting to Rational if possible
    if isinteger(base) && isinteger(pow) && pow < zero(pow)
        base = Rational(base)
    end
    coeff *= StaticCoeff(base^pow)
    return coeff, binding
end

Base.@assume_effects :foldable function collect_powers!(term::STerm, coeff, binding, power)
    # Non-call term: store or update its accumulated power in the binding
    if haskey(binding, term)
        binding = push(binding, term => seval(binding[term] + power))
    else
        binding = push(binding, term => power)
    end
    return coeff, binding
end

# Partition a base^power factor between numerator and denominator tuples.
# Keeping everything as tuples avoids heap allocations in the foldable path.
function splitpower(num::Tuple, den::Tuple, base::STerm, npow::STerm)
    # Negative exponents are represented as unary minus expressions.
    if isstaticzero(npow)
        return num, den
    elseif isunaryminus(npow)
        return num, (den..., base^-npow)
    else
        return (num..., base^npow), den
    end
end

# Consume (base, power) tuples recursively and accumulate factored tuples.
collect_factors(::Tuple{}, ::Tuple{}, num::Tuple, den::Tuple) = num, den
function collect_factors(exprs::Tuple{STerm,Vararg{STerm}},
                         data::Tuple{STerm,Vararg{STerm}},
                         num::Tuple,
                         den::Tuple)
    num_next, den_next = splitpower(num, den, first(exprs), first(data))
    return collect_factors(Base.tail(exprs), Base.tail(data), num_next, den_next)
end

# Coefficient 1 is implicit in products and should not introduce a factor.
function append_coeff(factors::Tuple, coeff)
    isone(coeff) && return factors
    return (factors..., SUniform(coeff))
end

Base.@assume_effects :foldable function abs_product_expr(monomial::Monomial)
    iszero(monomial.coeff) && return SUniform(0)

    num, den = collect_factors(monomial.powers.exprs,
                               monomial.powers.data,
                               (),
                               ())

    # We handle the sign separately by negating the product
    numc = append_coeff(num, abs(monomial.coeff))

    num_expr = *(numc...)
    den_expr = *(den...)

    return isstaticone(den_expr) ? num_expr : num_expr / den_expr
end

function product_expr(monomial::Monomial)
    abs_expr = abs_product_expr(monomial)
    return isnegative(monomial.coeff) ? -abs_expr : abs_expr
end

Base.@assume_effects :foldable function Base.isless(mx::Monomial, my::Monomial)
    # TODO
end

canonicalize_product(term::STerm) = term
function canonicalize_product(expr::SExpr{Call})
    op = operation(expr)
    (op === SRef(:*) || op === SRef(:/) || op === SRef(:inv)) || return expr
    return product_expr(Monomial(expr))
end
