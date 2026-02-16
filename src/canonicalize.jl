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

_is_uniform_literal(::STerm) = false
_is_uniform_literal(::SUniform) = true

function _is_uniform_literal(expr::SExpr{Call})
    operation(expr) === SRef(:-) || return false
    arity(expr) == 1 || return false
    return only(arguments(expr)) isa SUniform
end

_uniform_literal_value(x::SUniform) = value(x)
_uniform_literal_value(expr::SExpr{Call}) = -value(only(arguments(expr)))

_is_uniform_zero(term::STerm) = _is_uniform_literal(term) && iszero(_uniform_literal_value(term))
_is_uniform_one(term::STerm) = _is_uniform_literal(term) && isone(_uniform_literal_value(term))

function _is_uniform_negative(term::STerm)
    _is_uniform_literal(term) || return false
    v = _uniform_literal_value(term)
    return v isa Real && v < zero(v)
end

function _is_uniform_integer(term::STerm)
    _is_uniform_literal(term) || return false
    return _uniform_literal_value(term) isa Integer
end

_abs_uniform(term::STerm) = SUniform(abs(_uniform_literal_value(term)))

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
_isless_atom(x::SFun, y::SFun) = isless(name(x.f), name(y.f))
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

    if x isa SExpr && y isa SExpr
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
    if x isa SExpr && _is_uniform_integer(y)
        return 1
    elseif y isa SExpr && _is_uniform_integer(x)
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

function _pow_uniform(base::STerm, exp::STerm)
    b = _uniform_literal_value(base)
    p = _uniform_literal_value(exp)
    if p isa Integer
        if b isa Integer && p < 0
            return SUniform(Rational(b)^p)
        end
        return SUniform(b^p)
    else
        return SUniform(b^p)
    end
end

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

function _add_exponent!(exps::AbstractDict{STerm,STerm}, base::STerm, power::STerm)
    merged = haskey(exps, base) ? seval(exps[base] + power) : power
    exps[base] = _normalize_power(merged)
    return
end

function _collect_product_factors!(exps::AbstractDict{STerm,STerm}, coef::Base.RefValue{STerm}, term::STerm, power::STerm=SUniform(1))
    _is_uniform_zero(power) && return

    if term isa SExpr{Call}
        op = operation(term)
        if op === SRef(:*)
            for arg in arguments(term)
                _collect_product_factors!(exps, coef, arg, power)
            end
            return
        elseif op === SRef(:/)
            num, den = arguments(term)
            _collect_product_factors!(exps, coef, num, power)
            _collect_product_factors!(exps, coef, den, _normalize_power(seval(-power)))
            return
        elseif op === SRef(:inv)
            _collect_product_factors!(exps, coef, only(arguments(term)), _normalize_power(seval(-power)))
            return
        elseif op === SRef(:^)
            base, exp = arguments(term)
            _collect_product_factors!(exps, coef, base, _normalize_power(seval(power * exp)))
            return
        end
    end

    if _is_uniform_literal(term) && _is_uniform_literal(power)
        coef[] = seval(coef[] * _pow_uniform(term, power))
        return
    end

    _add_exponent!(exps, term, power)
    return
end

function _factor_with_power(base::STerm, power::STerm)
    power = _normalize_power(power)
    _is_uniform_one(power) && return base
    return base^power
end

function _sorted_product_expression(factors::Vector{STerm})
    isempty(factors) && return SUniform(1)
    return sort_product_terms(*(factors...))
end

function _reconstruct_product(exps::AbstractDict{STerm,STerm}, coefficient::STerm)
    _is_uniform_zero(coefficient) && return SUniform(0)

    numerator = STerm[]
    denominator = STerm[]

    for (base, power) in exps
        _is_uniform_zero(power) && continue
        if _is_uniform_negative(power)
            apow = _abs_uniform(power)
            _is_uniform_zero(apow) || push!(denominator, _factor_with_power(base, apow))
        else
            push!(numerator, _factor_with_power(base, power))
        end
    end

    negcoef = false
    if _is_uniform_literal(coefficient)
        negcoef = _is_uniform_negative(coefficient)
        cabs = negcoef ? _abs_uniform(coefficient) : coefficient
        _is_uniform_one(cabs) || push!(numerator, cabs)
    else
        push!(numerator, coefficient)
    end

    num_expr = _sorted_product_expression(numerator)
    den_expr = _sorted_product_expression(denominator)

    result = _is_uniform_one(den_expr) ? num_expr : num_expr / den_expr
    return negcoef ? -result : result
end

canonicalize_product(term::STerm) = term

function canonicalize_product(expr::SExpr{Call})
    op = operation(expr)
    (op === SRef(:*) || op === SRef(:/) || op === SRef(:inv)) || return expr

    exps = IdDict{STerm,STerm}()
    coef = Ref{STerm}(SUniform(1))
    _collect_product_factors!(exps, coef, expr, SUniform(1))
    return _reconstruct_product(exps, coef[])
end

function _split_term_coefficient(term::STerm)
    if _is_uniform_literal(term)
        return term, SUniform(1)
    end

    if term isa SExpr{Call} && operation(term) === SRef(:*)
        coef = SUniform(1)
        rest = STerm[]
        for arg in arguments(term)
            if _is_uniform_literal(arg)
                coef = seval(coef * arg)
            else
                push!(rest, arg)
            end
        end

        if isempty(rest)
            return coef, SUniform(1)
        elseif length(rest) == 1
            return coef, only(rest)
        else
            return coef, canonicalize_product(*(rest...))
        end
    end

    return SUniform(1), term
end

function _collect_sum_terms!(coeffs::AbstractDict{STerm,STerm}, term::STerm, sign::STerm=SUniform(1))
    if term isa SExpr{Call}
        op = operation(term)
        if op === SRef(:+)
            for arg in arguments(term)
                _collect_sum_terms!(coeffs, arg, sign)
            end
            return
        elseif op === SRef(:-)
            if arity(term) == 1
                _collect_sum_terms!(coeffs, only(arguments(term)), seval(-sign))
            else
                a, b = arguments(term)
                _collect_sum_terms!(coeffs, a, sign)
                _collect_sum_terms!(coeffs, b, seval(-sign))
            end
            return
        end
    end

    coef, mono = _split_term_coefficient(term)
    contrib = seval(sign * coef)
    coeffs[mono] = haskey(coeffs, mono) ? seval(coeffs[mono] + contrib) : contrib
    return
end

function _collect_monomial_powers!(exps::AbstractDict{STerm,STerm}, term::STerm, power::STerm=SUniform(1))
    _is_uniform_zero(power) && return

    if term isa SExpr{Call}
        op = operation(term)
        if op === SRef(:*)
            for arg in arguments(term)
                _collect_monomial_powers!(exps, arg, power)
            end
            return
        elseif op === SRef(:/)
            num, den = arguments(term)
            _collect_monomial_powers!(exps, num, power)
            _collect_monomial_powers!(exps, den, seval(-power))
            return
        elseif op === SRef(:inv)
            _collect_monomial_powers!(exps, only(arguments(term)), seval(-power))
            return
        elseif op === SRef(:^)
            base, exp = arguments(term)
            _collect_monomial_powers!(exps, base, seval(power * exp))
            return
        end
    end

    _add_exponent!(exps, term, power)
    return
end

function _monomial_pairs(term::STerm)
    _, mono = _split_term_coefficient(term)
    mono === SUniform(1) && return Pair{STerm,STerm}[]

    exps = IdDict{STerm,STerm}()
    _collect_monomial_powers!(exps, mono, SUniform(1))

    pairs = Pair{STerm,STerm}[]
    for (factor, power) in exps
        _is_uniform_zero(power) && continue
        push!(pairs, factor => power)
    end

    sort!(pairs; lt=(a, b) -> _isless_lex(first(a), first(b)))
    return pairs
end

function _monomial_degree(monomial)
    degree = SUniform(0)
    for pair in monomial
        degree = seval(degree + last(pair))
    end
    return degree
end

function _monomial_exponent(monomial, factor::STerm)
    for pair in monomial
        first(pair) === factor && return last(pair)
    end
    return SUniform(0)
end

function _monomial_union_factors(mx, my)
    factors = STerm[]
    for pair in mx
        push!(factors, first(pair))
    end
    for pair in my
        push!(factors, first(pair))
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

function _cmp_monomial_desc(mx, my)
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

function _is_constant_term(term::STerm)
    _, mono = _split_term_coefficient(term)
    return mono === SUniform(1)
end

function _isless_sum_term(x::STerm, y::STerm)
    x === y && return false

    xconst = _is_constant_term(x)
    yconst = _is_constant_term(y)
    if xconst != yconst
        return !xconst
    elseif xconst && yconst
        cx, _ = _split_term_coefficient(x)
        cy, _ = _split_term_coefficient(y)
        return _cmp_power(cx, cy) < 0
    end

    mx = _monomial_pairs(x)
    my = _monomial_pairs(y)
    cmp = _cmp_monomial_desc(mx, my)
    if cmp != 0
        return cmp > 0
    end

    return _isless_lex(x, y)
end

_sort_sum_terms_impl(expr::STerm) = expr

function _sort_sum_terms_impl(expr::SExpr{Call})
    operation(expr) === SRef(:+) || return expr
    args = collect(arguments(expr))
    sort!(args; lt=_isless_sum_term)
    return +(args...)
end

@generated function sort_sum_terms(expr::SExpr{Call})
    expri = expr.instance
    sorted = _sort_sum_terms_impl(expri)
    return :($sorted)
end

sort_sum_terms(expr::STerm) = expr

function _sorted_sum_vector(terms::Vector{STerm})
    isempty(terms) && return STerm[]
    sorted = sort_sum_terms(+(terms...))
    if sorted isa SExpr{Call} && operation(sorted) === SRef(:+)
        return collect(arguments(sorted))
    else
        return STerm[sorted]
    end
end

function _compose_coeff_monomial(coef::STerm, mono::STerm)
    mono === SUniform(1) && return coef
    _is_uniform_one(coef) && return mono

    term = coef * mono
    if term isa SExpr{Call}
        op = operation(term)
        if op === SRef(:*) || op === SRef(:/) || op === SRef(:inv)
            return canonicalize_product(term)
        end
    end
    return term
end

function _reconstruct_sum(coeffs::AbstractDict{STerm,STerm})
    entries = Pair{STerm,STerm}[]
    for (mono, coeff) in coeffs
        _is_uniform_zero(coeff) && continue
        push!(entries, mono => coeff)
    end

    isempty(entries) && return SUniform(0)

    sort!(entries; lt=(a, b) -> _isless_sum_term(first(a), first(b)))

    terms = STerm[]
    isneg = Bool[]
    for entry in entries
        mono = first(entry)
        coeff = last(entry)
        if _is_uniform_negative(coeff)
            push!(terms, _compose_coeff_monomial(_abs_uniform(coeff), mono))
            push!(isneg, true)
        else
            push!(terms, _compose_coeff_monomial(coeff, mono))
            push!(isneg, false)
        end
    end

    first_negative = findfirst(identity, isneg)
    last_positive = findlast(x -> !x, isneg)

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

    coeffs = IdDict{STerm,STerm}()
    _collect_sum_terms!(coeffs, expr, SUniform(1))
    return _reconstruct_sum(coeffs)
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
