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

    if isstatic(x) && isstatic(y)
        return isless(compute(x), compute(y))
    end

    rx = termrank(x)
    ry = termrank(y)
    rx == ry || return rx < ry

    if isexpr(x) && isexpr(y)
        return isless_expr(x, y)
    end
    return isless_atom(x, y)
end

# Static sorting of tuples of singleton types
_ssort_impl(args::Tuple, lt, by, order) = (sort!(collect(args); lt, by, order)...,)
@generated function ssort(args::Tuple; lt=Base.isless, by=Base.identity, order=Base.Order.Forward)
    sorted = _ssort_impl(args.instance, lt.instance, by.instance, order.instance)
    return :($sorted)
end

# Monomial representation of products for canonicalization

struct Monomial{S,B}
    coeff::S
    powers::B
end

Monomial(term::STerm) = Monomial(StaticCoeff(1), Binding(term => SUniform(1)))

function Monomial(expr::SExpr{Call})
    coeff, powers = collect_powers(expr)
    kv = ssort((pairs(powers)...,); lt=isless_lex, by=first)
    return Monomial(coeff, Binding(kv...))
end

isconstant(monomial::Monomial) = length(monomial.powers) == 0

Base.@assume_effects :foldable function collect_powers(term::SExpr{Call},
                                                       coeff::StaticCoeff=StaticCoeff(1),
                                                       binding::Binding=Binding(),
                                                       power::STerm=SUniform(1))
    op = operation(term)
    if op === SRef(:*)
        # Flatten the tree and accumulate powers
        for arg in arguments(term)
            coeff, binding = collect_powers(arg, coeff, binding, power)
        end
    elseif op === SRef(:/)
        # a / b is treated as a * b^-1, so powers in the denominator are negated
        num, den = arguments(term)
        coeff, binding = collect_powers(num, coeff, binding, power)
        coeff, binding = collect_powers(den, coeff, binding, -power)
    elseif op === SRef(:inv)
        # inv(a) is treated as a^-1, so powers are negated
        arg = only(arguments(term))
        coeff, binding = collect_powers(arg, coeff, binding, -power)
    elseif isunaryminus(term) && isstatic(power)
        # Fold an unary minus into the coefficient for odd integer powers
        p = compute(power)
        if isinteger(p)
            isodd(p) && (coeff = -coeff) # (-a)^(2k+1) = -(a^(2k+1))
            coeff, binding = collect_powers(only(arguments(term)), coeff, binding, power)
        end
    elseif op === SRef(:^)
        # Fold nested powers by multiplying exponents
        base, exp = arguments(term)
        coeff, binding = collect_powers(base, coeff, binding, seval(power * exp))
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
Base.@assume_effects :foldable function collect_powers(term::SUniform, coeff, binding, power::SUniform)
    base = value(term)
    pow = compute(power)
    # Preserve exact division of integers by promoting to Rational if possible
    if isinteger(base) && isinteger(pow) && pow < zero(pow)
        base = Rational(base)
    end
    coeff *= StaticCoeff(base^pow)
    return coeff, binding
end

Base.@assume_effects :foldable function collect_powers(term::STerm, coeff, binding, power)
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
function prepend_coeff(factors::Tuple, coeff)
    isone(coeff) && return factors
    return (SUniform(coeff), factors...)
end

Base.@assume_effects :foldable function abs_product_expr(monomial::Monomial)
    iszero(monomial.coeff) && return SUniform(0)

    num, den = collect_factors(monomial.powers.exprs,
                               monomial.powers.data,
                               (),
                               ())

    # We handle the sign separately by negating the product
    numc = prepend_coeff(num, abs(monomial.coeff))

    num_expr = *(numc...)
    den_expr = *(den...)

    return isstaticone(den_expr) ? num_expr : num_expr / den_expr
end

function product_expr(monomial::Monomial)
    abs_expr = abs_product_expr(monomial)
    return isnegative(monomial.coeff) ? -abs_expr : abs_expr
end

degree(monomial::Monomial) = isconstant(monomial) ? SUniform(0) : seval(+(values(monomial.powers)...))

function base_union(mx::Monomial, my::Monomial)
    px = (pairs(mx.powers)...,)
    py = (pairs(my.powers)...,)
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

canonicalize_product(term::STerm) = term
function canonicalize_product(expr::SExpr{Call})
    op = operation(expr)
    (op === SRef(:*) || op === SRef(:/) || op === SRef(:inv)) || return expr
    return product_expr(Monomial(expr))
end

function collect_terms(expr::STerm, binding, add)
    mon = Monomial(expr)
    if haskey(binding, mon.powers)
        binding = push(binding, mon.powers => binding[mon.powers] + add * mon.coeff)
    else
        binding = push(binding, mon.powers => add * mon.coeff)
    end
    return binding
end
Base.@assume_effects :foldable function collect_terms(expr::SExpr{Call}, binding=Binding(), add=StaticCoeff(1))
    op = operation(expr)
    if op === SRef(:+)
        for arg in arguments(expr)
            binding = collect_terms(arg, binding, add)
        end
    elseif op === SRef(:-)
        if arity(expr) == 1
            binding = collect_terms(only(arguments(expr)), binding, -add)
        else
            a, b = arguments(expr)
            binding = collect_terms(a, binding, add)
            binding = collect_terms(b, binding, -add)
        end
    else
        mon = Monomial(expr)
        if haskey(binding, mon.powers)
            binding = push(binding, mon.powers => binding[mon.powers] + add * mon.coeff)
        else
            binding = push(binding, mon.powers => add * mon.coeff)
        end
    end
    return binding
end

build_tree(expr, ::Tuple{}) = expr
Base.@assume_effects :foldable function build_tree(expr, monomials)
    mon = first(monomials)
    rest = Base.tail(monomials)
    if isnegative(mon.coeff)
        new_expr = expr - abs_product_expr(mon)
    elseif operation(expr) === SRef(:+)
        new_expr = SExpr(Call(), children(expr)..., abs_product_expr(mon))
    else
        new_expr = expr + abs_product_expr(mon)
    end
    return build_tree(new_expr, rest)
end

function canonicalize_sum(expr::SExpr{Call})
    op = operation(expr)
    (op === SRef(:+) || op === SRef(:-)) || return expr
    binding = collect_terms(expr)
    kv = filter(!iszero ∘ last, (pairs(binding)...,))
    isempty(kv) && return SUniform(0)
    monomials = map(kv -> Monomial(kv[2], kv[1]), kv)
    # Sort by grevlex order and reconstruct the sum
    sorted = ssort(monomials; order=Base.Order.Reverse)
    # Build a tree of additions and subtractions from the sorted monomials
    return build_tree(product_expr(first(sorted)), Base.tail(sorted))
end
