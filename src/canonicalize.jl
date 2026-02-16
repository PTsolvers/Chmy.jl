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

_is_uniform_literal(term::STerm) = (term isa SUniform || iscall(term)) && isstatic(term)
_uniform_literal_value(term::STerm) = compute(term)::Real

function _is_uniform_integer(term::STerm)
    _is_uniform_literal(term) || return false
    return _uniform_literal_value(term) isa Integer
end

_term_rank(::SIndex) = 1
_term_rank(::STensor) = 2
_term_rank(::SRef) = 3
_term_rank(::SFun) = 4
_term_rank(::SUniform) = 5
_term_rank(::SExpr) = 6
_term_rank(::STerm) = 7

_head_rank(::Call) = 1
_head_rank(::Comp) = 2
_head_rank(::Ind) = 3
_head_rank(::Loc) = 4
_head_rank(expr::SExpr) = _head_rank(head(expr))

const SAtom = Union{SIndex,STensor,SRef,SFun,SUniform}

_atom_rank(::SIndex) = 1
_atom_rank(::STensor) = 2
_atom_rank(::SRef) = 3
_atom_rank(::SFun) = 4
_atom_rank(::SUniform) = 5

_isless_atom(x::SAtom, y::SAtom) = _atom_rank(x) < _atom_rank(y)
_isless_atom(::SIndex{I}, ::SIndex{J}) where {I,J} = I < J
_isless_atom(::SRef{F1}, ::SRef{F2}) where {F1,F2} = isless(F1, F2)
_isless_atom(x::SFun, y::SFun) = isless(nameof(x.f), nameof(y.f))
_isless_atom(x::SUniform, y::SUniform) = isless(value(x), value(y))
function _isless_atom(x::STensor, y::STensor)
    nx = name(x)
    ny = name(y)
    nx == ny || return isless(nx, ny)
    if x !== y
        throw(ArgumentError("tensors with the same name must have the same rank and kind"))
    end
    return false
end

_isless_term_tuple(::Tuple{}, ::Tuple{}) = false
_isless_term_tuple(::Tuple{}, ::Tuple{Any,Vararg}) = true
_isless_term_tuple(::Tuple{Any,Vararg}, ::Tuple{}) = false
function _isless_term_tuple(xs::Tuple{X,Vararg}, ys::Tuple{Y,Vararg}) where {X,Y}
    xh = first(xs)
    yh = first(ys)
    if xh === yh
        return _isless_term_tuple(Base.tail(xs), Base.tail(ys))
    end
    return _isless_lex(xh, yh)
end

function _isless_expr(x::SExpr, y::SExpr)
    hx = _head_rank(x)
    hy = _head_rank(y)
    hx == hy || return hx < hy
    return _isless_term_tuple(children(x), children(y))
end

function _isless_lex(x::STerm, y::STerm)
    x === y && return false

    rx = _term_rank(x)
    ry = _term_rank(y)
    rx == ry || return rx < ry

    if isexpr(x) && isexpr(y)
        return _isless_expr(x, y)
    elseif x isa SAtom && y isa SAtom
        return _isless_atom(x, y)
    else
        return _isless_atom(x, y)
    end
end

_cmp_ordered(x, y) = x == y ? 0 : (isless(x, y) ? -1 : 1)

function _cmp_power(x::STerm, y::STerm)
    x === y && return 0
    if _is_uniform_literal(x) && _is_uniform_literal(y)
        return _cmp_ordered(_uniform_literal_value(x), _uniform_literal_value(y))
    end
    if isexpr(x) && _is_uniform_integer(y)
        return 1
    elseif isexpr(y) && _is_uniform_integer(x)
        return -1
    end
    if _isless_lex(x, y)
        return -1
    elseif _isless_lex(y, x)
        return 1
    end
    return 0
end

_strip_power_base(term::STerm) = term
function _strip_power_base(term::SExpr{Call})
    operation(term) === SRef(:^) || return term
    return first(arguments(term))
end

_numeric_factor_score(v) = abs(log(abs(v)))

function _isless_uniform_factor(x::SUniform, y::SUniform)
    x === y && return false
    vx = value(x)
    vy = value(y)
    iszero(vx) && return true
    iszero(vy) && return false

    sx = _numeric_factor_score(vx)
    sy = _numeric_factor_score(vy)
    sx == sy || return isless(sx, sy)

    ax = abs(vx)
    ay = abs(vy)
    ax == ay || return isless(ax, ay)

    return isless(vx, vy)
end

function _isless_product_factor(x::STerm, y::STerm)
    x === y && return false

    if x isa SUniform
        return y isa SUniform ? _isless_uniform_factor(x, y) : true
    elseif y isa SUniform
        return false
    end

    return _isless_lex(_strip_power_base(x), _strip_power_base(y))
end

const SPower = Union{STerm,Real}

mutable struct Monomial
    neg::Bool
    coeff::Real
    powers::IdDict{STerm,SPower}
end

function Monomial(coeff::Real=1)
    return Monomial(coeff < zero(coeff), abs(coeff), IdDict{STerm,SPower}())
end

function Monomial(coeff::Real, powers::AbstractDict{STerm,<:Any})
    out = IdDict{STerm,SPower}()
    for (factor, power) in powers
        out[factor] = power
    end
    return Monomial(coeff < zero(coeff), abs(coeff), out)
end

_signed_coeff(monomial::Monomial) = monomial.neg ? -monomial.coeff : monomial.coeff

function _set_signed_coeff!(monomial::Monomial, coeff::Real)
    monomial.neg = coeff < zero(coeff)
    monomial.coeff = abs(coeff)
    return
end

function _scale_coeff!(monomial::Monomial, factor::Real)
    _set_signed_coeff!(monomial, _signed_coeff(monomial) * factor)
    return
end

_power_term(power::Real) = SUniform(power)
_power_term(power::STerm) = power

function _pow_real(base::Real, power::Real)
    if base isa Integer && power isa Integer && power < 0
        return Rational(base)^power
    end
    return base^power
end

_normalize_product_power(power::Real) = power
function _normalize_product_power(power::STerm)
    reduced = _canonicalize_power_term(seval(power))
    return _is_uniform_literal(reduced) ? _uniform_literal_value(reduced) : reduced
end

_neg_power(power::Real) = -power
_neg_power(power::STerm) = _normalize_product_power(seval(-power))

function _sort_product_terms_impl(expr::SExpr{Call})
    operation(expr) === SRef(:*) || return expr
    args = collect(arguments(expr))
    sort!(args; lt=_isless_product_factor)
    return *(args...)
end

sort_product_terms(expr::STerm) = expr
@generated function sort_product_terms(expr::SExpr{Call})
    expri = expr.instance
    sorted = _sort_product_terms_impl(expri)
    return :($sorted)
end

_canonicalize_power_term(term::STerm) = seval(term)
function _canonicalize_power_term(term::SExpr{Call})
    op = operation(term)
    if op === SRef(:+) || op === SRef(:-)
        return canonicalize_sum(term)
    elseif op === SRef(:*) || op === SRef(:/) || op === SRef(:inv)
        return canonicalize_product(term)
    else
        return seval(term)
    end
end

_normalize_power(power::STerm) = _canonicalize_power_term(seval(power))

function _add_monomial_power!(monomial::Monomial, base::STerm, power::SPower)
    merged = haskey(monomial.powers, base) ? monomial.powers[base] + power : power
    norm = _normalize_product_power(merged)
    if iszero(norm)
        pop!(monomial.powers, base, nothing)
    else
        monomial.powers[base] = norm
    end
    return
end

function _collect_monomial!(monomial::Monomial, term::STerm, power::SPower=1)
    power = _normalize_product_power(power)
    power isa Real && iszero(power) && return

    if iscall(term)
        op = operation(term)
        if op === SRef(:*)
            for arg in arguments(term)
                _collect_monomial!(monomial, arg, power)
            end
            return
        elseif op === SRef(:/)
            num, den = arguments(term)
            _collect_monomial!(monomial, num, power)
            _collect_monomial!(monomial, den, _neg_power(power))
            return
        elseif op === SRef(:inv)
            _collect_monomial!(monomial, only(arguments(term)), _neg_power(power))
            return
        elseif op === SRef(:-) && arity(term) == 1 && power isa Integer
            isodd(power) && (monomial.neg = !monomial.neg)
            _collect_monomial!(monomial, only(arguments(term)), power)
            return
        elseif op === SRef(:^)
            base, exp = arguments(term)
            _collect_monomial!(monomial, base, _normalize_product_power(seval(power * exp)))
            return
        end
    end

    if _is_uniform_literal(term) && power isa Real
        _scale_coeff!(monomial, _pow_real(_uniform_literal_value(term), power))
        return
    end

    _add_monomial_power!(monomial, term, power)
    return
end

function Monomial(term::STerm)
    monomial = Monomial(1)
    _collect_monomial!(monomial, term, 1)
    return monomial
end

_unit_monomial(monomial::Monomial) = Monomial(1, monomial.powers)

function _factor_with_power(base::STerm, power::SPower)
    power = _normalize_product_power(power)
    power isa Real && isone(power) && return base
    return base^power
end

function _sorted_product_expression(factors::Vector{STerm})
    isempty(factors) && return SUniform(1)
    return sort_product_terms(*(factors...))
end

function canonical_abs_product(monomial::Monomial)
    iszero(monomial.coeff) && return SUniform(0)

    numerator = STerm[]
    denominator = STerm[]

    for (base, power) in monomial.powers
        if power isa Real && iszero(power)
            continue
        elseif power isa Real && power < 0
            apow = -power
            iszero(apow) || push!(denominator, _factor_with_power(base, apow))
        else
            push!(numerator, _factor_with_power(base, power))
        end
    end

    isone(monomial.coeff) || push!(numerator, SUniform(monomial.coeff))

    num_expr = _sorted_product_expression(numerator)
    den_expr = _sorted_product_expression(denominator)

    return isone(den_expr) ? num_expr : num_expr / den_expr
end

function canonical_product(monomial::Monomial)
    abs_expr = canonical_abs_product(monomial)
    if iszero(monomial.coeff)
        return abs_expr
    end
    return monomial.neg ? -abs_expr : abs_expr
end

canonicalize_product(term::STerm) = term

function canonicalize_product(expr::SExpr{Call})
    op = operation(expr)
    (op === SRef(:*) || op === SRef(:/) || op === SRef(:inv)) || return expr

    return canonical_product(Monomial(expr))
end

_isconstant(monomial::Monomial) = isempty(monomial.powers)

function _add_sum_monomial!(monomials::Vector{Monomial}, index::IdDict{STerm,Int}, monomial::Monomial, sign::Real)
    unit = _unit_monomial(monomial)
    key = canonical_abs_product(unit)
    contrib = sign * _signed_coeff(monomial)
    if haskey(index, key)
        i = index[key]
        _set_signed_coeff!(monomials[i], _signed_coeff(monomials[i]) + contrib)
    else
        push!(monomials, Monomial(contrib, unit.powers))
        index[key] = length(monomials)
    end
    return
end

function _collect_sum_terms!(monomials::Vector{Monomial}, index::IdDict{STerm,Int}, term::STerm, sign::Real=1)
    if iscall(term)
        op = operation(term)
        if op === SRef(:+)
            for arg in arguments(term)
                _collect_sum_terms!(monomials, index, arg, sign)
            end
            return
        elseif op === SRef(:-)
            if arity(term) == 1
                _collect_sum_terms!(monomials, index, only(arguments(term)), -sign)
            else
                a, b = arguments(term)
                _collect_sum_terms!(monomials, index, a, sign)
                _collect_sum_terms!(monomials, index, b, -sign)
            end
            return
        end
    end

    _add_sum_monomial!(monomials, index, Monomial(term), sign)
    return
end

function _monomial_degree(monomial::Monomial)
    degree = SUniform(0)
    for power in values(monomial.powers)
        degree = seval(degree + power)
    end
    return degree
end

function _monomial_exponent(monomial::Monomial, factor::STerm)
    haskey(monomial.powers, factor) || return SUniform(0)
    return _power_term(monomial.powers[factor])
end

function _monomial_union_factors(mx::Monomial, my::Monomial)
    factors = STerm[]
    for factor in keys(mx.powers)
        push!(factors, factor)
    end
    for factor in keys(my.powers)
        push!(factors, factor)
    end

    sort!(factors; lt=_isless_lex)

    uniq = STerm[]
    for factor in factors
        if isempty(uniq) || (uniq[end] !== factor)
            push!(uniq, factor)
        end
    end
    return uniq
end

function _cmp_monomial_desc(mx::Monomial, my::Monomial)
    dcmp = _cmp_power(_monomial_degree(mx), _monomial_degree(my))
    dcmp == 0 || return dcmp

    factors = _monomial_union_factors(mx, my)
    for factor in Iterators.reverse(factors)
        cmp = _cmp_power(_monomial_exponent(mx, factor), _monomial_exponent(my, factor))
        cmp == 0 && continue
        return cmp < 0 ? 1 : -1
    end

    return 0
end

function _isless_sum(mx::Monomial, my::Monomial)
    xconst = _isconstant(mx)
    yconst = _isconstant(my)
    if xconst != yconst
        return !xconst
    elseif xconst && yconst
        return false
    end

    cmp = _cmp_monomial_desc(mx, my)
    if cmp != 0
        return cmp > 0
    end

    return _isless_lex(canonical_abs_product(mx), canonical_abs_product(my))
end

function _reconstruct_sum(monomials::Vector{Monomial})
    filter!(monomial -> !iszero(monomial.coeff), monomials)

    isempty(monomials) && return SUniform(0)

    sort!(monomials; lt=_isless_sum)

    terms = STerm[]
    isneg = Bool[]
    for monomial in monomials
        push!(terms, canonical_abs_product(monomial))
        push!(isneg, monomial.neg)
    end

    first_negative = findfirst(isneg)
    last_positive = findlast(!, isneg)

    if isnothing(first_negative)
        return +(terms...)
    elseif isnothing(last_positive)
        result = -terms[1]
        for term in @view terms[2:end]
            result = result - term
        end
        return result
    elseif last_positive < first_negative
        result = +(terms[1:first_negative-1]...)
        for term in @view terms[first_negative:end]
            result = result - term
        end
        return result
    end

    if isneg[1]
        result = -terms[1]
    else
        result = terms[1]
    end

    for i in 2:length(terms)
        term = terms[i]
        if isneg[i]
            result = result - term
        else
            result = result + term
        end
    end

    return result
end

canonicalize_sum(term::STerm) = term

function canonicalize_sum(expr::SExpr{Call})
    op = operation(expr)
    (op === SRef(:+) || op === SRef(:-)) || return expr

    monomials = Monomial[]
    index = IdDict{STerm,Int}()
    _collect_sum_terms!(monomials, index, expr, 1)
    return _reconstruct_sum(monomials)
end

struct CanonicalizeRule <: AbstractRule end

Base.@assume_effects :foldable function (::CanonicalizeRule)(expr::SExpr{Call})
    op = operation(expr)
    if op === SRef(:+) || op === SRef(:-)
        return canonicalize_sum(expr)
    elseif op === SRef(:*) || op === SRef(:/) || op === SRef(:inv)
        return canonicalize_product(expr)
    elseif op === SRef(:^)
        base, power = arguments(expr)
        return seval(base^_normalize_power(power))
    else
        return seval(expr)
    end
end

canonicalize(expr::STerm) = Postwalk(CanonicalizeRule())(expr)
