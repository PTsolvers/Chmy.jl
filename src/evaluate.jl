"""
    evaluate(term)
    evaluate(op, args)

Rebuild `term` bottom-up through the canonical constructors. Terms built with operators, `getindex`, and `pack` are
already canonical, i.e. the representative of their class under the algebraic and tensor rules, so `evaluate` is
only needed for terms assembled from raw parts, e.g. `Literal(2.0)`.

The second method is the rule of the call `op(args...)` with canonical `args`, applied every time the call is
constructed: it returns the canonical form of the call, or `nothing` to keep the call as is. The rules of a function
`f` are methods of `evaluate(::typeof(f), args)`, and the rules of other operators are methods of
`evaluate(::MyOperator, args)`. A rule may construct any term except the call `op(args...)` itself.
"""
function evaluate(term::DTerm)::DTerm
    return @match term begin
        Literal(x) => canonical_literal(x)
        ZeroTensor(rank) => 𝒪(rank)
        IdTensor(rank) => ℐ(rank)
        DExpr(head, args) => maketerm(head, map(evaluate, args))
        _ => term
    end
end

# operators have no rules by default
evaluate(::Operator, args) = nothing
evaluate(op::Fun, args) = evaluate(op.f, args)

# functions of literals are evaluated numerically
evaluate(f, args) = fold_literals(f, args)

function fold_literals(f, args)
    (!isempty(args) && all(isliteral, args)) || return nothing
    return canonical_literal(f(map(value, args)...)::Number)
end

# integer values are `Int`
canonical_number(x::Number) = isinteger(x) ? Int(x) : x

# literals with canonical values, zero and one are shared
function canonical_literal(x::Number)
    y = canonical_number(x)
    iszero(y) && return ZERO
    isone(y) && return ONE
    return Literal(y)
end
# booleans are kept, so that conditions stay valid
canonical_literal(x::Bool) = Literal(x)

# broadcasting over scalars is a scalar call
evaluate(op::BroadcastedFun, args) = all(x -> tensorrank(x) == 0, args) ? makecall(Fun(op), args) : nothing

# arithmetic

# sums and differences collect like terms, products and powers collect the powers of like factors, and the remaining
# functions of literals are evaluated numerically
for (f, rule) in ((:+, :sum_rule), (:-, :sum_rule), (:*, :product_rule), (:/, :product_rule), (:^, :product_rule))
    @eval evaluate(::typeof($f), args) = arithmetic($rule, Fun($f), args)
end

function arithmetic(rule, op, args)
    raw = makecall_unchecked(op, args)
    term = rule((raw,), tensorrank(raw))
    # the collected terms are calls of functions
    iscall(term) || return term
    return something(fold_literals(operation(term).f, arguments(term)), term)
end

# tensor products

# products with a zero tensor are zero, which takes precedence over the identities
function zero_product(rank, a, b)
    isstaticzero(a) && tensorrank(a) == rank && return a
    isstaticzero(b) && tensorrank(b) == rank && return b
    (isstaticzero(a) || isstaticzero(b)) && return 𝒪(rank)
    return nothing
end

function evaluate(::typeof(⋅), (a, b)::Tuple{DTerm, DTerm})
    ra, rb = tensorrank(a), tensorrank(b)
    zero = zero_product(ra + rb - 2, a, b)
    isnothing(zero) || return zero
    # identity: I ⋅ a == a ⋅ I == a
    isstaticone(a) && ra == 2 && return b
    isstaticone(b) && rb == 2 && return a
    ra == rb == 2 || return nothing
    # inv(A) ⋅ A = I, cof(A)' ⋅ A = det(A) * I, and cof(A) ⋅ A' = det(A) * I
    for (x, y) in ((a, b), (b, a))
        isargof(inv, x, y) && return ℐ(2)
        isargof(cof, argof(adjoint, x), y) && return det(y) * ℐ(2)
        isargof(cof, x, y') && return det(y') * ℐ(2)
    end
    return nothing
end

function evaluate(::typeof(⊡), (a, b)::Tuple{DTerm, DTerm})
    zero = zero_product(tensorrank(a) + tensorrank(b) - 4, a, b)
    isnothing(zero) || return zero
    tensorrank(a) == tensorrank(b) == 2 || return nothing
    # trace: I ⊡ A = A ⊡ I = tr(A)
    isstaticone(a) && return tr(b)
    isstaticone(b) && return tr(a)
    return nothing
end

evaluate(::typeof(⊗), (a, b)::Tuple{DTerm, DTerm}) = zero_product(tensorrank(a) + tensorrank(b), a, b)

function evaluate(::typeof(×), (a, b)::Tuple{DTerm, DTerm})
    zero = zero_product(1, a, b)
    isnothing(zero) || return zero
    # v×v = 0
    return a ==ₛ b ? 𝒪(1) : nothing
end

# matrix functions

# the symmetry of a rank-2 tensor: named tensors have their kind, zero and identity tensors are diagonal, and the
# projections `sym` and `asym` are symmetric and alternating
function matrix_kind(a)
    return @match a begin
        Tensor(_, _, kind, _) => kind
        ZeroTensor(_) || IdTensor(_) => Kind.Diag()
        DExpr(Call(inner), (base,)) => projection_kind(inner, base)
        _ => Kind.None()
    end
end

projection_kind(op::Fun, base) = projection_kind(op.f, base)
projection_kind(op, base) = Kind.None()
projection_kind(::typeof(sym), base) = Kind.Sym()
projection_kind(::typeof(asym), base) = Kind.Alt()
# negation preserves symmetry
projection_kind(::typeof(-), base) = matrix_kind(base)
# cofactors preserve symmetry, but not antisymmetry
projection_kind(::typeof(cof), base) = cofactor_kind(matrix_kind(base))
cofactor_kind(kind::Union{Kind.Sym, Kind.Diag}) = kind
cofactor_kind(kind) = Kind.None()

# the argument of the call `f(base)`, and `nothing` for other terms
function argof(f, a)
    return @match a begin
        DExpr(Call(inner), (base,)) => inner == Fun(f) ? base : nothing
        _ => nothing
    end
end
argof(f, ::Nothing) = nothing

# whether `a` is the call `f(base)`
isargof(f, a, base) = (x = argof(f, a); !isnothing(x) && x ==ₛ base)

# the transpose and the symmetric and antisymmetric parts of a rank-2 tensor of known symmetry
transposed(::Union{Kind.Sym, Kind.Diag}, a) = a
transposed(::Kind.Alt, a) = -a
transposed(::Kind.None, a) = nothing

symmetric_part(::Union{Kind.Sym, Kind.Diag}, a) = a
symmetric_part(::Kind.Alt, a) = 𝒪(2)
symmetric_part(::Kind.None, a) = nothing

antisymmetric_part(::Union{Kind.Sym, Kind.Diag}, a) = 𝒪(2)
antisymmetric_part(::Kind.Alt, a) = a
antisymmetric_part(::Kind.None, a) = nothing

const MatrixFunction = Union{
    typeof(adjoint), typeof(sym), typeof(asym), typeof(inv), typeof(cof), typeof(det), typeof(tr), typeof(diag),
    typeof(gram), typeof(cogram),
}

evaluate(f::MatrixFunction, (a,)::Tuple{DTerm}) = tensorrank(a) == 0 ? scalar_rule(f, a) : matrix_rule(f, a)

# matrix functions are the identity on scalars unless specified otherwise
scalar_rule(f, a) = a
scalar_rule(::typeof(asym), a) = ZERO
scalar_rule(::typeof(cof), a) = ONE
scalar_rule(::Union{typeof(gram), typeof(cogram)}, a) = a^2
# scalar inverses are powers
scalar_rule(::typeof(inv), a) = arithmetic(product_rule, Fun(inv), (a,))

matrix_rule(f, a) = nothing

# involutions: inv(inv(A)) = A and A'' = A
function matrix_rule(::typeof(adjoint), a)
    base = argof(adjoint, a)
    return isnothing(base) ? transposed(matrix_kind(a), a) : base
end
matrix_rule(::typeof(inv), a) = isstaticone(a) ? a : argof(inv, a)

matrix_rule(::typeof(sym), a) = symmetric_part(matrix_kind(a), a)
matrix_rule(::typeof(asym), a) = antisymmetric_part(matrix_kind(a), a)
matrix_rule(::typeof(cof), a) = isstaticone(a) ? a : nothing

function matrix_rule(::typeof(det), a)
    isstaticone(a) && return ONE
    return isstaticzero(a) ? ZERO : nothing
end

# antisymmetric tensors have zero trace and diagonal
matrix_rule(::typeof(tr), a) = isstaticzero(a) || matrix_kind(a) == Kind.Alt() ? ZERO : nothing

function matrix_rule(::typeof(diag), a)
    isstaticone(a) && return ℐ(1)
    return isstaticzero(a) || matrix_kind(a) == Kind.Alt() ? 𝒪(1) : nothing
end

# polynomial collection

# the polynomial is reconstructed with raw calls, which are canonical by construction

const Polynomial = Dict{Dict{DTerm, DTerm}, Rational{Int}}

rawcall(f, args...) = makecall_unchecked(Fun(f), args)

function rational_value(term)
    return @match term begin
        Literal(x) => @match canonical_number(x) begin
            y::Union{Integer, Rational} => Rational{Int}(y)
            _ => nothing
        end
        _ => nothing
    end
end

function integer_power(term)
    c = rational_value(term)
    return !isnothing(c) && isinteger(c) ? Int(c) : nothing
end

function negate_power(power)
    p = integer_power(power)
    !isnothing(p) && return Literal(Base.checked_neg(p))
    @match power begin
        Literal(x) => return canonical_literal(-x)
        _ => nothing
    end
    return sum_rule((power,), 0, -1 // 1)
end

function add_exponents(x, y)
    cx, cy = rational_value(x), rational_value(y)
    !isnothing(cx) && !isnothing(cy) && return canonical_literal(cx + cy)
    isliteral(x) && isliteral(y) && return canonical_literal(value(x) + value(y))
    return sum_rule((x, y), 0)
end

function multiply_exponents(x, y)
    x ==ₛ ONE && return y
    y ==ₛ ONE && return x
    cx, cy = rational_value(x), rational_value(y)
    !isnothing(cx) && !isnothing(cy) && return canonical_literal(cx * cy)
    isliteral(x) && isliteral(y) && return canonical_literal(value(x) * value(y))
    return product_rule((x, y), 0)
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

# additive children are canonical, so their terms are already ordered
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
                positive = sum_rule((factor,), tensorrank(factor), -1 // 1)
                isodd(p) && (coef = -coef)
                return collect_monomial!(powers, coef, positive, power)
            elseif op isa Fun{typeof(^)}
                return collect_monomial!(powers, coef, args[1], multiply_exponents(power, args[2]))
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
        isnothing(p) && return sum_rule((power for (_, power) in factors), 0)
        numeric += p
    end
    return canonical_literal(numeric)
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
call_expr(op, args) = length(args) == 1 ? args[1] : rawcall(op, args...)

function abs_product_expr(coef, factors, num, den)
    empty!(num)
    empty!(den)
    c = abs(coef)
    for (base, power) in factors
        @match power begin
            Literal(p::Real) && if p < 0
            end => begin
                positive = negate_power(power)
                push!(den, positive ==ₛ ONE ? base : rawcall(^, base, positive))
            end
            _ => push!(num, power ==ₛ ONE ? base : rawcall(^, base, power))
        end
    end
    if isempty(num)
        isempty(den) && return canonical_literal(c)
        denominator = call_expr(*, den)
        return isone(c) ? rawcall(inv, denominator) : rawcall(/, canonical_literal(c), denominator)
    elseif isempty(den)
        isone(c) || pushfirst!(num, canonical_literal(c))
        return call_expr(*, num)
    end
    ratio = rawcall(/, call_expr(*, num), call_expr(*, den))
    return isone(c) ? ratio : rawcall(*, canonical_literal(c), ratio)
end

function product_expr(coef, powers, rank)
    iszero(coef) && return 𝒪(rank)
    isempty(powers) && return canonical_literal(coef)
    expr = abs_product_expr(coef, sorted_factors(powers), DTerm[], DTerm[])
    return coef < 0 ? rawcall(-, expr) : expr
end

function sum_expr(terms, rank)
    isempty(terms) && return 𝒪(rank)
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
            expr = rawcall(-, call_expr(+, positive), expr)
            empty!(positive)
        end
        push!(positive, expr)
    end
    expr = call_expr(+, positive)
    return negative ? rawcall(-, expr) : expr
end

function product_rule(args, rank)
    powers = Dict{DTerm, DTerm}()
    coef = 1 // 1
    for arg in args
        coef = collect_monomial!(powers, coef, arg, ONE)
    end
    return product_expr(coef, powers, rank)
end

function sum_rule(args, rank, scale = 1 // 1)
    terms = Polynomial()
    scratch = nothing
    for arg in args
        scratch = collect_terms!(terms, scratch, arg, scale)
    end
    return sum_expr(terms, rank)
end
