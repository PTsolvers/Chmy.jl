# static evaluation
isstatic(::STerm) = false
isstatic(::SUniform) = true
isstatic(expr::SExpr{Call}) = all(isstatic, arguments(expr))

seval(term::STerm) = isstatic(term) ? SUniform(compute(term)) : term

# lex ordering of terms for deterministic canonicalization
termrank(::SIndex) = 1
termrank(::STensor) = 2
termrank(::SZeroTensor) = 3
termrank(::SIdTensor) = 4
termrank(::SRef) = 5
termrank(::SFun) = 6
termrank(::SUniform) = 7
termrank(::SExpr) = 8

headrank(::Call) = 1
headrank(::Comp) = 2
headrank(::Ind) = 3
headrank(::Loc) = 4
headrank(expr::SExpr) = headrank(head(expr))

isless_lex(::SIndex{I}, ::SIndex{J}) where {I,J} = isless(I, J)
isless_lex(::SRef{F1}, ::SRef{F2}) where {F1,F2} = isless(F1, F2)
isless_lex(x::SFun, y::SFun) = isless(nameof(x.f), nameof(y.f))
function isless_lex(x::STensor, y::STensor)
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
function isless_lex(x::STerm, y::STerm)
    x === y && return false

    tx = tensorrank(x)
    ty = tensorrank(y)
    tx == ty || return isless(tx, ty)

    # fully static terms can be compared at compile time
    if isstatic(x) && isstatic(y)
        return isless(compute(x), compute(y))
    end

    # different kinds of terms are compared by their rank
    rx = termrank(x)
    ry = termrank(y)
    rx == ry || return rx < ry

    # special logic for comparing expressions
    return isless_expr(x, y)
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

# static sorting of tuples of singleton types
_ssort_impl(args::Tuple, lt, by, order) = (sort!(collect(args); lt, by, order)...,)
@generated function ssort(args::Tuple; lt=Base.isless, by=Base.identity, order=Base.Order.Forward)
    sorted = _ssort_impl(args.instance, lt.instance, by.instance, order.instance)
    return :($sorted)
end

# monomial representation of products for canonicalization
struct Monomial{S,B}
    coeff::S
    powers::B
end
Monomial(::SUniform{C}) where {C} = Monomial(StaticCoeff(C), Binding())
Monomial(term::STerm) = Monomial(StaticCoeff(1), Binding(term => SUniform(1)))
function Monomial(expr::SExpr{Call})
    coeff, powers = collect_powers(expr)
    kv = ssort(pairstuple(powers); lt=isless_lex, by=first)
    return Monomial(coeff, Binding(kv...))
end

isconstant(monomial::Monomial) = length(monomial.powers) == 0

Base.iszero(monomial::Monomial) = iszero(monomial.coeff) || any(isstaticzero, keys(monomial.powers))

tensorrank(monomial::Monomial) = maximum(tensorrank, keys(monomial.powers); init=0)

function addpower(binding, term, power)
    if haskey(binding, term)
        # keep merged exponents canonicalized as we accumulate factors
        return push(binding, term => binding[term] + power)
    else
        return push(binding, term => power)
    end
end

collect_powers(term) = collect_powers(term, StaticCoeff(1), Binding(), SUniform(1))
Base.@assume_effects :foldable function collect_powers(term::SExpr{Call}, coeff, binding, npow)
    op = operation(term)
    if op === SRef(:*)
        # flatten the tree and accumulate powers
        coeff, binding = collect_powers(first(arguments(term)), coeff, binding, npow)
        rest = makeop(:*, Base.tail(arguments(term))...)
        coeff, binding = collect_powers(rest, coeff, binding, npow)
    elseif op === SRef(:/)
        # a / b is treated as a * b^-1, so powers in the denominator are negated
        num, den = arguments(term)
        coeff, binding = collect_powers(num, coeff, binding, npow)
        coeff, binding = collect_powers(den, coeff, binding, -npow)
    elseif op === SRef(:inv)
        arg = only(arguments(term))
        if tensorrank(arg) == 0
            # scalar inv(a) is treated as a^-1, so powers are negated
            coeff, binding = collect_powers(arg, coeff, binding, -npow)
        else
            binding = addpower(binding, term, npow)
        end
    elseif isunaryminus(term) && isstatic(npow)
        # fold an unary minus into the coefficient for odd integer powers, e.g. a * (-x)^3 -> -a * x^3
        p = compute(npow)
        if isinteger(p)
            isodd(p) && (coeff = -coeff)
            coeff, binding = collect_powers(only(arguments(term)), coeff, binding, npow)
        end
    elseif op === SRef(:^)
        # fold nested powers by multiplying exponents
        base, exp = arguments(term)
        newnpow = isstaticone(npow) ? exp : npow * exp
        coeff, binding = collect_powers(base, coeff, binding, newnpow)
    else
        binding = addpower(binding, term, npow)
    end
    return coeff, binding
end
function collect_powers(term, coeff, binding, npow)
    # fully static uniform literals can be folded into coeff at compile time
    if isstatic(term) && isstatic(npow)
        base = compute(term)
        powr = compute(npow)
        # preserve exact division of integers by promoting to Rational if possible
        if isinteger(base) && isinteger(powr) && powr < zero(powr)
            base = Rational(base)
        end
        coeff *= StaticCoeff(Base.literal_pow(^, base, Val(powr)))
    else
        binding = addpower(binding, term, npow)
    end
    return coeff, binding
end

# partition a base^power factor between numerator and denominator tuples
function splitpower(num, den, base, npow)
    if isstaticzero(npow)
        return num, den
    elseif isstaticone(npow)
        return (num..., base), den
    elseif isstatic(npow) && compute(npow) < zero(compute(npow))
        if isstaticone(-npow)
            return num, (den..., base)
        end
        return num, (den..., makeop(:^, base, -npow))
    else
        return (num..., makeop(:^, base, npow)), den
    end
end

# consume (base, power) tuples recursively and accumulate factored tuples
collect_factors(::Tuple{}, ::Tuple{}, num, den) = num, den
Base.@assume_effects :foldable function collect_factors(exprs, data, num, den)
    num_next, den_next = splitpower(num, den, first(exprs), first(data))
    return collect_factors(Base.tail(exprs), Base.tail(data), num_next, den_next)
end

function abs_product_expr(m::Monomial)
    iszero(m) && return SZeroTensor{tensorrank(m)}()

    num, den = collect_factors(keys(m.powers), values(m.powers), (), ())

    c = abs(m.coeff)

    if isempty(num)
        isempty(den) && return SUniform(c)
        isone(c) && return makeop(:inv, makeop(:*, den...))
        return makeop(:/, SUniform(c), makeop(:*, den...))
    end

    if isempty(den)
        isone(c) && return makeop(:*, num...)
        return makeop(:*, SUniform(c), num...)
    end

    expr = makeop(:/, makeop(:*, num...), makeop(:*, den...))

    return isone(c) ? expr : makeop(:*, SUniform(c), expr)
end

function STerm(monomial::Monomial)
    abs_expr = abs_product_expr(monomial)
    return isnegative(monomial.coeff) ? -abs_expr : abs_expr
end

degree(monomial::Monomial) = isconstant(monomial) ? SUniform(0) : +(values(monomial.powers)...)

# align monomials to the same ordered base set before grevlex comparison
function base_union(mx::Monomial, my::Monomial)
    px = pairstuple(mx.powers)
    py = pairstuple(my.powers)
    bx, by = base_union((), (), px, py)
    return Monomial(mx.coeff, bx), Monomial(my.coeff, by)
end
base_union(x, y, ::Tuple{}, ::Tuple{}) = Binding(x...), Binding(y...)
function base_union(x, y, ::Tuple{}, ty)
    fy = first(ty)
    base_union((x..., fy[1] => SUniform(0)), (y..., fy), (), Base.tail(ty))
end
function base_union(x, y, tx, ::Tuple{})
    fx = first(tx)
    base_union((x..., fx), (y..., fx[1] => SUniform(0)), Base.tail(tx), ())
end
function base_union(x, y, tx, ty)
    fx, fy = first(tx), first(ty)
    if fx[1] === fy[1]
        return base_union((x..., fx), (y..., fy), Base.tail(tx), Base.tail(ty))
    elseif isless_lex(fx[1], fy[1])
        return base_union((x..., fx), (y..., fx[1] => SUniform(0)), Base.tail(tx), ty)
    else
        return base_union((x..., fy[1] => SUniform(0)), (y..., fy), tx, Base.tail(ty))
    end
end

isless_grevlex(::Tuple{}, ::Tuple{}) = false
function isless_grevlex(x::Tuple, y::Tuple)
    # grevlex breaks ties from the last exponent backwards
    lx, ly = last(x), last(y)
    lx === ly && return isless_grevlex(Base.front(x), Base.front(y))
    return !isless_lex(lx, ly)
end

function Base.isless(mx::Monomial, my::Monomial)
    dx = degree(mx)
    dy = degree(my)

    if dx === dy
        bx, by = base_union(mx, my)
        return isless_grevlex(values(bx.powers), values(by.powers))
    end

    return isless_lex(dx, dy)
end

canonicalize_product(expr::STerm) = STerm(Monomial(expr))

collect_terms(expr::STerm) = collect_terms(expr, Binding(), StaticCoeff(1))
function collect_terms(expr::STerm, binding, add)
    # map each monomial basis to its accumulated scalar coefficient
    mon = Monomial(expr)
    return addpower(binding, mon.powers, add * mon.coeff)
end
Base.@assume_effects :foldable function collect_terms(expr::SExpr{Call}, binding, add)
    op = operation(expr)
    if op === SRef(:+)
        arg = first(arguments(expr))
        binding = collect_terms(arg, binding, add)
        rest = makeop(:+, Base.tail(arguments(expr))...)
        binding = collect_terms(rest, binding, add)
    elseif op === SRef(:-)
        if arity(expr) == 1
            arg = only(arguments(expr))
            binding = collect_terms(arg, binding, -add)
        else
            a, b = arguments(expr)
            binding = collect_terms(a, binding, add)
            binding = collect_terms(b, binding, -add)
        end
    else
        mon = Monomial(expr)
        binding = addpower(binding, mon.powers, add * mon.coeff)
    end
    return binding
end

build_tree(expr, ::Tuple{}) = expr
function build_tree(expr, monomials)
    mon = first(monomials)
    rest = Base.tail(monomials)
    # rebuild as `+`/`-` to preserve readable signs in the final tree
    if isnegative(mon.coeff)
        new_expr = makeop(:-, expr, abs_product_expr(mon))
    elseif iscall(expr) && operation(expr) === SRef(:+)
        new_expr = SExpr(Call(), children(expr)..., abs_product_expr(mon))
    else
        new_expr = makeop(:+, expr, abs_product_expr(mon))
    end
    return build_tree(new_expr, rest)
end

canonicalize_sum(term::STerm) = term
function canonicalize_sum(expr::SExpr{Call})
    binding = collect_terms(expr)
    monomials = map(x -> Monomial(x[2], x[1]), (pairs(binding)...,))
    nz_monomials = filter(!iszero, monomials)
    isempty(nz_monomials) && return SZeroTensor{tensorrank(expr)}()
    # sort by grevlex order and reconstruct the sum
    sorted = ssort(nz_monomials; order=Base.Order.Reverse)
    # build a tree of additions and subtractions from the sorted monomials
    first_expr = STerm(first(sorted))
    return build_tree(first_expr, Base.tail(sorted))
end

struct CanonicalizeRule <: AbstractRule end
function (::CanonicalizeRule)(expr::SExpr{Call})
    op = operation(expr)
    # normalize multiplicative and additive families with dedicated passes
    if op === SRef(:*) || op === SRef(:/) || op === SRef(:inv) || op === SRef(:^)
        return canonicalize_product(expr)
    elseif op === SRef(:+) || op === SRef(:-)
        return canonicalize_sum(expr)
    else
        return seval(expr)
    end
end

"""
    canonicalize(expr)

Return a deterministic canonical form of a symbolic expression `expr` with respect to algebraic operations.
`canonicalize` merges multiplicative factors into monomials, collects like terms in sums, and sorts 
terms with a stable ordering so structurally equivalent expressions map to the same tree.

Canonicalization is not recursively applied to subterms, for a recursive version use `simplify` instead.
"""
canonicalize(expr::STerm) = Passthrough(CanonicalizeRule())(expr)

"""
    simplify(expr)

Return a simplified form of a symbolic expression `expr` by recursively applying `canonicalize` to all subterms of `expr`.
"""
simplify(expr::STerm) = Postwalk(CanonicalizeRule())(expr)
