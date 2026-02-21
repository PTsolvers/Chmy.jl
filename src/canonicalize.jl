# static evaluation
isstatic(::STerm) = false
isstatic(::SUniform) = true
isstatic(expr::SExpr{Call}) = all(isstatic, arguments(expr))

seval(term::STerm) = isstatic(term) ? SUniform(compute(term)) : term

# lex ordering of terms for deterministic canonicalization
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
    tx = tensorrank(x)
    ty = tensorrank(y)
    tx == ty || return isless(tx, ty)
    hx = headrank(x)
    hy = headrank(y)
    hx == hy || return hx < hy
    return isless_tuple(children(x), children(y))
end

function isless_lex(x::STerm, y::STerm)
    x === y && return false

    # fully static terms can be compared at compile time
    if isstatic(x) && isstatic(y)
        return isless(compute(x), compute(y))
    end

    # different kinds of terms are compared by their rank
    rx = termrank(x)
    ry = termrank(y)
    rx == ry || return rx < ry

    # special logic for comparing expressions
    if isexpr(x) && isexpr(y)
        return isless_expr(x, y)
    end

    # leaf terms
    return isless_atom(x, y)
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
    kv = ssort((pairs(powers)...,); lt=isless_lex, by=first)
    return Monomial(coeff, Binding(kv...))
end

isconstant(monomial::Monomial) = length(monomial.powers) == 0
tensorrank(monomial::Monomial) = maximum(tensorrank, keys(monomial.powers); init=0)

function addterm(binding, term, power)
    if haskey(binding, term)
        # keep merged exponents canonicalized as we accumulate factors
        return push(binding, term => binding[term] + power)
    else
        return push(binding, term => power)
    end
end

function collect_powers(term::SExpr{Call}, coeff=StaticCoeff(1), binding=Binding(), power=SUniform(1))
    op = operation(term)
    if op === SRef(:*)
        # flatten the tree and accumulate powers
        for arg in arguments(term)
            coeff, binding = collect_powers(arg, coeff, binding, power)
        end
    elseif op === SRef(:/)
        # a / b is treated as a * b^-1, so powers in the denominator are negated
        num, den = arguments(term)
        coeff, binding = collect_powers(num, coeff, binding, power)
        coeff, binding = collect_powers(den, coeff, binding, -power)
    elseif op === SRef(:inv)
        arg = only(arguments(term))
        if tensorrank(arg) == 0
            # scalar inv(a) is treated as a^-1, so powers are negated
            coeff, binding = collect_powers(arg, coeff, binding, -power)
        else
            binding = addterm(binding, term, power)
        end
    elseif isunaryminus(term) && isstatic(power)
        # fold an unary minus into the coefficient for odd integer powers, e.g. a * (-x)^3 -> -a * x^3
        p = compute(power)
        if isinteger(p)
            isodd(p) && (coeff = -coeff)
            coeff, binding = collect_powers(only(arguments(term)), coeff, binding, power)
        end
    elseif op === SRef(:^)
        # fold nested powers by multiplying exponents
        base, exp = arguments(term)
        coeff, binding = collect_powers(base, coeff, binding, power * exp)
    else
        binding = addterm(binding, term, power)
    end
    return coeff, binding
end
function collect_powers(term::STerm, coeff, binding, power)
    # fully static uniform literals can be folded into coeff at compile time
    if isstatic(term) && isstatic(power)
        base = compute(term)
        pow = compute(power)
        # preserve exact division of integers by promoting to Rational if possible
        if isinteger(base) && isinteger(pow) && pow < zero(pow)
            base = Rational(base)
        end
        coeff *= StaticCoeff(base^pow)
    else
        binding = addterm(binding, term, power)
    end
    return coeff, binding
end

# partition a base^power factor between numerator and denominator tuples.
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

# consume (base, power) tuples recursively and accumulate factored tuples.
collect_factors(::Tuple{}, ::Tuple{}, num, den) = num, den
function collect_factors(exprs, data, num, den)
    num_next, den_next = splitpower(num, den, first(exprs), first(data))
    return collect_factors(Base.tail(exprs), Base.tail(data), num_next, den_next)
end

function abs_product_expr(m::Monomial)
    iszero(m.coeff) && return SZeroTensor{tensorrank(m)}()

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
    return isnegative(monomial.coeff) ? makeop(:-, abs_expr) : abs_expr
end

degree(monomial::Monomial) = isconstant(monomial) ? SUniform(0) : +(values(monomial.powers)...)

# align monomials to the same ordered base set before grevlex comparison
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

function collect_terms(expr::STerm, binding, add)
    # map each monomial basis to its accumulated scalar coefficient
    mon = Monomial(expr)
    if haskey(binding, mon.powers)
        binding = push(binding, mon.powers => binding[mon.powers] + add * mon.coeff)
    else
        binding = push(binding, mon.powers => add * mon.coeff)
    end
    return binding
end
function collect_terms(expr::SExpr{Call}, binding=Binding(), add=StaticCoeff(1))
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
    kv = filter(!iszero ∘ last, (pairs(binding)...,))
    isempty(kv) && return SZeroTensor{tensorrank(expr)}()
    monomials = map(x -> Monomial(x[2], x[1]), kv)
    # sort by grevlex order and reconstruct the sum
    sorted = ssort(monomials; order=Base.Order.Reverse)
    # build a tree of additions and subtractions from the sorted monomials
    return build_tree(STerm(first(sorted)), Base.tail(sorted))
end

"""
    canonicalize(term)

Return a deterministic canonical form of a symbolic term. `canonicalize` folds
fully static subexpressions into `SUniform` values, merges multiplicative factors
into monomials, collects like terms in sums, and sorts terms with a stable ordering
so structurally equivalent expressions map to the same tree.
"""
canonicalize(expr::STerm) = expr
Base.@assume_effects :foldable function canonicalize(expr::SExpr)
    iscall(expr) || return expr
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
