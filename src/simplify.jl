"""
    simplify(expr)

Simplify an expression with respect to algebraic rules.
"""
simplify(term::DTerm) = postwalk(canonicalize, term)

"""
    canonicalize(expr)

Computes a deterministic canonical representation of an expression.
"""
function canonicalize(term::DTerm)
    @match term begin
        Literal(x) => begin
            iszero(x) && return ZERO
            isone(x) && return ONE
            isinteger(x) && return Literal(Int(x))
        end
        ZeroTensor(0) => return ZERO
        IdTensor(0) => return ONE
        DExpr(Call(op), args) => begin
            # polynomial collection, including scalar inverses
            if isscalarop(op) || (op isa Fun{typeof(inv)} && tensorrank(args[1]) == 0)
                if op isa Union{Fun{typeof(+)}, Fun{typeof(-)}}
                    term = canonicalize_sum((term,), tensorrank(term))
                else
                    term = canonicalize_product((term,), tensorrank(term))
                end
                @match term begin
                    DExpr(Call(next_op), next_args) => begin
                        op, args = next_op, next_args
                    end
                    _ => return canonicalize(term)
                end
            elseif istensorop(op)
                result = canonicalize_tensor(term, op, args)
                isnothing(result) || return result
            elseif ismatrixop(op)
                result = canonicalize_matrix(term, op, args[1])
                isnothing(result) || return result
            end
            if op isa Fun && all(isliteral, args)
                return canonicalize(Literal(evaluate_literals(op, args)::Number))
            end
        end
        _ => nothing
    end
    return term
end

function canonicalize_tensor(term, op, args)
    a, b = args
    # zero annihilation takes precedence over inverse cancellation
    if isstaticzero(a) || isstaticzero(b)
        rank = tensorrank(term)
        isstaticzero(a) && tensorrank(a) == rank && return a
        isstaticzero(b) && tensorrank(b) == rank && return b
        return zero_expr(rank)
    end
    if op isa Fun{typeof(⋅)}
        ra, rb = tensorrank(a), tensorrank(b)
        # identity: I ⋅ a == a ⋅ I == a
        isstaticone(a) && ra == 2 && return b
        isstaticone(b) && rb == 2 && return a
        ra == rb == 2 || return term
        # inv(A) ⋅ A = I, cof(A)' ⋅ A = det(A) * I, and cof(A) ⋅ A' = det(A) * I
        for (x, y) in ((a, b), (b, a))
            @match x begin
                DExpr(Call(inner), (base,)) => begin
                    if inner isa Fun{typeof(inv)}
                        tensorrank(base) == 2 && base ==ₛ y && return IdTensor(2)
                    elseif inner isa Fun{typeof(adjoint)}
                        @match base begin
                            DExpr(Call(inner_op), (cof_base,)) => begin
                                inner_op isa Fun{typeof(cof)} && cof_base ==ₛ y && return det(cof_base) * IdTensor(2)
                            end
                            _ => nothing
                        end
                    elseif inner isa Fun{typeof(cof)}
                        canonicalize(base') ==ₛ y && return det(base) * IdTensor(2)
                    end
                end
                _ => nothing
            end
        end
    elseif op isa Fun{typeof(⊡)} && tensorrank(a) == tensorrank(b) == 2
        # trace: I ⊡ A = A ⊡ I = tr(A)
        isstaticone(a) && return canonicalize(tr(b))
        isstaticone(b) && return canonicalize(tr(a))
    elseif op isa Fun{typeof(×)} && a ==ₛ b
        return ZeroTensor(1) # v×v = 0
    end
    return nothing
end

function canonicalize_matrix(term, op, a)
    tensorrank(a) == 2 || return term
    kind = @match a begin
        Tensor(_, _, kind, _) => kind
        # identity and zero tensors are diagonal, with known scalar invariants
        IdTensor(_) => begin
            op isa Union{Fun{typeof(inv)}, Fun{typeof(cof)}} && return a
            op isa Fun{typeof(det)} && return ONE
            op isa Fun{typeof(diag)} && return IdTensor(1)
            Kind.Diag()
        end
        ZeroTensor(_) => begin
            op isa Union{Fun{typeof(tr)}, Fun{typeof(det)}} && return ZERO
            op isa Fun{typeof(diag)} && return ZeroTensor(1)
            Kind.Diag()
        end
        DExpr(Call(inner), inner_args) => begin
            ismatrixop(inner) || return term
            base = inner_args[1]
            tensorrank(base) == 2 || return term
            # involutions: inv(inv(A)) = A and A'' = A
            if (op isa Fun{typeof(inv)} && inner isa Fun{typeof(inv)}) ||
                    (op isa Fun{typeof(adjoint)} && inner isa Fun{typeof(adjoint)})
                return base
            end
            # explicit projections carry symmetry just like named tensor kinds
            if inner isa Fun{typeof(sym)}
                Kind.Sym()
            elseif inner isa Fun{typeof(asym)}
                Kind.Alt()
            elseif inner isa Fun{typeof(cof)}
                # cofactors preserve symmetry, but not antisymmetry
                @match base begin
                    Tensor(_, _, kind, _) => kind isa Union{Kind.Sym, Kind.Diag} ? kind : Kind.None()
                    _ => Kind.None()
                end
            else
                Kind.None()
            end
        end
        _ => Kind.None()
    end
    # transpose, projection idempotence, and mixed-projection cancellation
    if op isa Union{Fun{typeof(adjoint)}, Fun{typeof(sym)}}
        kind isa Union{Kind.Sym, Kind.Diag} && return a
        kind isa Kind.Alt && return op isa Fun{typeof(adjoint)} ? -a : ZeroTensor(2)
    elseif op isa Fun{typeof(asym)}
        isstaticzero(a) && return a
        kind isa Union{Kind.Sym, Kind.Diag} && return ZeroTensor(2)
        kind isa Kind.Alt && return a
    elseif kind isa Kind.Alt
        # antisymmetric tensors have zero trace and diagonal
        op isa Fun{typeof(tr)} && return ZERO
        op isa Fun{typeof(diag)} && return ZeroTensor(1)
    end
    return nothing
end

const Polynomial = Dict{Dict{DTerm, DTerm}, Rational{Int}}

function rational_value(term)
    return @match term begin
        Literal(x) => @match x begin
            x::Int => x // 1
            x::Rational{Int} => x
            x::Union{Integer, Rational} => Rational{Int}(x)
            _ => nothing
        end
        _ => nothing
    end
end

function integer_power(term)
    return @match term begin
        Literal(x) => @match x begin
            x::Int => x
            x::Union{Rational{Int}, Float64} => isinteger(x) ? Int(x) : nothing
            x::Real => isinteger(x) ? Int(x)::Int : nothing
            _ => nothing
        end
        _ => nothing
    end
end

function coefficient_expr(c)
    iszero(c) && return ZERO
    isone(c) && return ONE
    return isinteger(c) ? Literal(numerator(c)) : Literal(c)
end
zero_expr(rank) = iszero(rank) ? ZERO : ZeroTensor(rank)

function negate_power(power)
    p = integer_power(power)
    !isnothing(p) && return Literal(Base.checked_neg(p))
    @match power begin
        Literal(x) => return canonicalize(Literal(-x))
        _ => nothing
    end
    return canonicalize_sum((power,), 0, -1 // 1)
end

function add_exponents(x, y)
    cx, cy = rational_value(x), rational_value(y)
    !isnothing(cx) && !isnothing(cy) && return coefficient_expr(cx + cy)
    isliteral(x) && isliteral(y) && return canonicalize(Literal(value(x) + value(y)))
    return canonicalize_sum((x, y), 0)
end

function multiply_exponents(x, y)
    x ==ₛ ONE && return y
    y ==ₛ ONE && return x
    cx, cy = rational_value(x), rational_value(y)
    !isnothing(cx) && !isnothing(cy) && return coefficient_expr(cx * cy)
    isliteral(x) && isliteral(y) && return canonicalize(Literal(value(x) * value(y)))
    return canonicalize_product((x, y), 0)
end

function addpower!(powers, factor, power)
    power ==ₛ ZERO && return
    old = get(powers, factor, nothing)
    combined = isnothing(old) ? power : add_exponents(old, power)
    if combined ==ₛ ZERO
        delete!(powers, factor)
    else
        powers[factor] = combined
    end
    return
end

# additive children are already ordered by the scalar postwalk
# inspect the first coefficient without rebuilding the factor
function factor_leading_sign(term::DTerm)::Int
    c = rational_value(term)
    !isnothing(c) && return c < 0 ? -1 : 1
    @match term begin
        DExpr(Call(op), args) => begin
            if op isa Fun{typeof(-)} && length(args) == 1
                return -factor_leading_sign(first(args))
            elseif op isa Union{Fun{typeof(+)}, Fun{typeof(-)}}
                return factor_leading_sign(first(args))
            end
        end
        _ => nothing
    end
    return 1
end

function collect_monomial!(powers, coef, factor, power)
    p = integer_power(power)
    p == 0 && return coef
    c = rational_value(factor)
    !isnothing(c) && !isnothing(p) && return coef * c^p

    @match factor begin
        ZeroTensor(_) => !isnothing(p) && p > 0 && (coef = 0 // 1)
        IdTensor(0) => return coef
        DExpr(Call(op), args) => begin
            if op isa Fun{typeof(*)}
                for arg in args
                    coef = collect_monomial!(powers, coef, arg, power)
                end
                return coef
            elseif op isa Fun{typeof(/)}
                coef = collect_monomial!(powers, coef, args[1], power)
                return collect_monomial!(powers, coef, args[2], negate_power(power))
            elseif op isa Fun{typeof(inv)} && tensorrank(args[1]) == 0
                return collect_monomial!(powers, coef, args[1], negate_power(power))
            elseif op isa Fun{typeof(-)} && length(args) == 1 && !isnothing(p)
                isodd(p) && (coef = -coef)
                return collect_monomial!(powers, coef, args[1], power)
            elseif op isa Union{Fun{typeof(+)}, Fun{typeof(-)}} && !isnothing(p) && factor_leading_sign(factor) < 0
                positive = canonicalize_sum((factor,), tensorrank(factor), -1 // 1)
                isodd(p) && (coef = -coef)
                return collect_monomial!(powers, coef, positive, power)
            elseif op isa Fun{typeof(^)}
                exponent = canonicalize(args[2])
                return collect_monomial!(powers, coef, args[1], multiply_exponents(power, exponent))
            end
        end
        _ => nothing
    end

    addpower!(powers, factor, power)
    return coef
end

function collect_terms!(terms, scratch, term, scale)
    @match term begin
        DExpr(Call(op), args) => begin
            if op isa Fun{typeof(+)}
                for arg in args
                    scratch = collect_terms!(terms, scratch, arg, scale)
                end
                return scratch
            elseif op isa Fun{typeof(-)}
                if length(args) == 1
                    return collect_terms!(terms, scratch, args[1], -scale)
                end
                scratch = collect_terms!(terms, scratch, args[1], scale)
                return collect_terms!(terms, scratch, args[2], -scale)
            end
        end
        _ => nothing
    end

    powers = isnothing(scratch) ? Dict{DTerm, DTerm}() : scratch
    coef = collect_monomial!(powers, scale, term, ONE)
    if !iszero(coef)
        stored = getkey(terms, powers, nothing)
        if isnothing(stored)
            terms[powers] = coef
            return nothing
        end
        coef += terms[stored]
        if iszero(coef)
            delete!(terms, stored)
        else
            terms[stored] = coef
        end
    end
    empty!(powers)
    return powers
end

sorted_factors(powers) = sort!(collect(powers); lt = (x, y) -> first(x) <ₛ first(y))

function degree(factors)
    isempty(factors) && return ZERO
    length(factors) == 1 && return last(factors[1])
    numeric = 0 // 1
    for (_, power) in factors
        p = rational_value(power)
        isnothing(p) && return canonicalize_sum((power for (_, power) in factors), 0)
        numeric += p
    end
    return coefficient_expr(numeric)
end

struct OrderedMonomial
    coef::Rational{Int}
    factors::Vector{Pair{DTerm, DTerm}}
    degree::DTerm
end

function OrderedMonomial(coef, powers)
    factors = sorted_factors(powers)
    return OrderedMonomial(coef, factors, degree(factors))
end

function Base.isless(x::OrderedMonomial, y::OrderedMonomial)
    x.degree ==ₛ y.degree || return x.degree <ₛ y.degree
    i, j = length(x.factors), length(y.factors)
    while i > 0 || j > 0
        if i == 0
            return last(y.factors[j]) <ₛ ZERO
        elseif j == 0
            return ZERO <ₛ last(x.factors[i])
        end
        bx, px = x.factors[i]
        by, py = y.factors[j]
        if bx ==ₛ by
            px ==ₛ py || return py <ₛ px
            i -= 1
            j -= 1
        elseif bx <ₛ by
            return py <ₛ ZERO
        else
            return ZERO <ₛ px
        end
    end
    return false
end

# reconstruction only passes nonempty argument buffers
call_expr(op, args) = length(args) == 1 ? args[1] : DExpr(Call(Fun(op)), Tuple(args))

function abs_product_expr(coef, factors, num, den)
    empty!(num)
    empty!(den)
    c = abs(coef)
    for (base, power) in factors
        @match power begin
            Literal(p::Real) && if p < 0
            end => begin
                positive = negate_power(power)
                push!(den, positive ==ₛ ONE ? base : base^positive)
            end
            _ => push!(num, power ==ₛ ONE ? base : base^power)
        end
    end
    if isempty(num)
        isempty(den) && return coefficient_expr(c)
        denominator = call_expr(*, den)
        return isone(c) ? inv(denominator) : coefficient_expr(c) / denominator
    elseif isempty(den)
        isone(c) || pushfirst!(num, coefficient_expr(c))
        return call_expr(*, num)
    end
    ratio = call_expr(*, num) / call_expr(*, den)
    return isone(c) ? ratio : coefficient_expr(c) * ratio
end

function product_expr(coef, powers, rank)
    iszero(coef) && return zero_expr(rank)
    isempty(powers) && return coefficient_expr(coef)
    expr = abs_product_expr(coef, sorted_factors(powers), DTerm[], DTerm[])
    return coef < 0 ? -expr : expr
end

function sum_expr(terms, rank)
    isempty(terms) && return zero_expr(rank)
    if length(terms) == 1
        powers, coef = first(terms)
        return product_expr(coef, powers, rank)
    end
    ordered = Vector{OrderedMonomial}(undef, length(terms))
    for (i, (powers, coef)) in enumerate(terms)
        ordered[i] = OrderedMonomial(coef, powers)
    end
    sort!(ordered; rev = true)

    negative = first(ordered).coef < 0
    num, den, positive = DTerm[], DTerm[], DTerm[]
    for mon in ordered
        coef = negative ? -mon.coef : mon.coef
        expr = abs_product_expr(coef, mon.factors, num, den)
        if coef < 0
            expr = call_expr(+, positive) - expr
            empty!(positive)
        end
        push!(positive, expr)
    end
    expr = call_expr(+, positive)
    return negative ? -expr : expr
end

function canonicalize_product(args, rank)
    powers = Dict{DTerm, DTerm}()
    coef = 1 // 1
    for arg in args
        coef = collect_monomial!(powers, coef, arg, ONE)
    end
    return product_expr(coef, powers, rank)
end

function canonicalize_sum(args, rank, scale = 1 // 1)
    terms = Polynomial()
    scratch = nothing
    for arg in args
        scratch = collect_terms!(terms, scratch, arg, scale)
    end
    return sum_expr(terms, rank)
end

isscalarop(op) = op isa Union{Fun{typeof(+)}, Fun{typeof(-)}, Fun{typeof(*)}, Fun{typeof(/)}, Fun{typeof(^)}}
istensorop(op) = op isa Union{Fun{typeof(⋅)}, Fun{typeof(⊡)}, Fun{typeof(⊗)}, Fun{typeof(×)}}
function ismatrixop(op)
    return op isa Union{Fun{typeof(inv)}, Fun{typeof(cof)}, Fun{typeof(adjoint)}, Fun{typeof(det)}, Fun{typeof(tr)}, Fun{typeof(diag)}, Fun{typeof(sym)}, Fun{typeof(asym)}}
end

evaluate_literals(op, args) = op.f(map(value, args)...)
