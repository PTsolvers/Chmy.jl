"""
    evaluate(term)
    evaluate(op, args)
    evaluate(op, args, dims)

Rebuild `term` bottom-up through the canonical constructors. Terms built with operators, `getindex`, and `pack` are
already canonical, i.e. the representative of their class under the algebraic and tensor rules, so `evaluate` is
only needed for terms assembled from raw parts, e.g. `Literal(2.0)`.

The second method is the rule of the call `op(args...)` with canonical `args`, applied every time the call is
constructed: it returns the canonical form of the call, or `nothing` if no rule applies, in which case a function of
literals is evaluated numerically, e.g. `sin(Literal(0))`, and other calls are kept as they are. The rules of a function
`f` are methods of `evaluate(::typeof(f), args)`, and the rules of other operators are methods of
`evaluate(::MyOperator, args)`. A rule may construct any term except the call `op(args...)` itself.

The third method is the rule of the call in `dims` dimensions, with its arguments in component form, applied when the
call is constructed with tensors in component form, see [`pack`](@ref), and by [`components`](@ref). It returns the
call in component form, a [`pack`](@ref) or a scalar, e.g. for the trace or the divergence, built from the arguments,
or `nothing`. By default, tensor-valued calls are evaluated with the rules of their components, see
[`component`](@ref).
"""
function evaluate(term::DTerm)::DTerm
    return @match term begin
        Literal(x) => canonical_literal(x)
        DExpr(head, args) => maketerm(head, map(evaluate, args))
        AddMul(_, _, _) => mapentries(evaluate, term)
        _ => term
    end
end

# operators have no rules by default
evaluate(::Operator, args) = nothing
evaluate(op::Fun, args) = evaluate(op.f, args)

# tensor-valued calls are evaluated component by component if every component has a rule
function evaluate(op, args, dims)
    rank = tensorrank(op, args)
    rank > 0 || return nothing
    data = DTerm[]
    complete = visit_components(Kind.None(), dims, rank) do I, state
        c = component(op, args, I)::Union{Nothing, DTerm}
        isnothing(c) || push!(data, c)
        return state && !isnothing(c)
    end
    return complete ? makepack(rank, dims, Kind.None(), Tuple(data)) : nothing
end
evaluate(op::Fun, args, dims) = evaluate(op.f, args, dims)

"""
    component(op, args, I)

The rule of the component `I` of the call `op(args...)` with canonical `args`, applied every time the component is
constructed, e.g. `grad(u)[1, 2]`: it returns the component, or `nothing` to keep the component of the call. These
rules hold in any number of dimensions, e.g. the component `(i, j)` of the gradient of `u` is the derivative of `u[j]`
along the axis `i`; components depending on the number of dimensions are evaluated once it is known, see
[`evaluate`](@ref), e.g. the trace or the divergence. A rule may construct any term except the component
`op(args...)[I...]` itself.
"""
component(op, args, I) = nothing
component(op::Fun, args, I) = component(op.f, args, I)

# the zero and identity tensors of rank zero are literals, and their components are the same in any rank
evaluate(op::ZeroTensor, args) = iszero(op.rank) ? ZERO : nothing
evaluate(op::IdTensor, args) = iszero(op.rank) ? ONE : nothing
component(::ZeroTensor, args, I) = ZERO
component(::IdTensor, args, I) = allequal(I) ? ONE : ZERO
evaluate(op::ZeroTensor, args, dims) = literal_pack(ZERO, dims, op.rank)
evaluate(op::IdTensor, args, dims) = literal_pack(ONE, dims, op.rank)

# functions have no rules by default
evaluate(f, args) = nothing

# functions of literals are evaluated numerically, other operators are not
fold_literals(::Operator, args) = nothing
fold_literals(op::Fun, args) = fold_literals(op.f, args)

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

# sums collect the coefficients of like terms, and products the powers of like bases, see `sumof` and `prodof`; the
# arguments of `+`, `*` and `^` can be vectors, so that sums and products are rebuilt without specialising on their number
evaluate(::typeof(+), args) = sumof(0, map(a -> a => 1, args), tensorrank(first(args)))
evaluate(::typeof(-), (a,)::Tuple{DTerm}) = scaled(-1, a)
evaluate(::typeof(-), (a, b)::Tuple{DTerm, DTerm}) = sumof(0, (a => 1, b => -1), tensorrank(a))
evaluate(::typeof(*), args) = prodof(1, map(a -> a => ONE, args), maximum(tensorrank, args))
evaluate(::typeof(/), (a, b)::Tuple{DTerm, DTerm}) = prodof(1, (a => ONE, b => Literal(-1)), tensorrank(a))
evaluate(::typeof(^), (a, p)) = power(a, p)

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

# the symmetry of a rank-2 tensor: named tensors have their kind, and calls the symmetry of their operator
function matrix_kind(a)
    return @match a begin
        Tensor(_, _, kind, _) => kind
        DExpr(Call(op), args) => matrix_kind(op, args)
        # scalar multiples preserve symmetry, and sums keep the symmetry common to their terms
        AddMul(_, _, dict) => mapreduce(matrix_kind, common_kind, Iterators.filter(k -> tensorrank(k) > 0, dict.keys))
        _ => Kind.None()
    end
end

common_kind(a::TensorKind, b::TensorKind) = a == b ? a : Kind.None()
common_kind(::Kind.Diag, b::Kind.Sym) = b
common_kind(a::Kind.Sym, ::Kind.Diag) = a

# the symmetry of the call `op(args...)`
matrix_kind(op::Fun, args) = matrix_kind(op.f, args)
matrix_kind(op, args) = Kind.None()
matrix_kind(::Union{ZeroTensor, IdTensor}, args) = Kind.Diag()
matrix_kind(::typeof(sym), (base,)::Tuple{DTerm}) = Kind.Sym()
matrix_kind(::typeof(asym), (base,)::Tuple{DTerm}) = Kind.Alt()
# cofactors preserve symmetry, but not antisymmetry
matrix_kind(::typeof(cof), (base,)::Tuple{DTerm}) = cofactor_kind(matrix_kind(base))
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
    typeof(adjoint),
    typeof(sym),
    typeof(asym),
    typeof(inv),
    typeof(cof),
    typeof(det),
    typeof(tr),
    typeof(diag),
    typeof(gram),
    typeof(cogram),
}

evaluate(f::MatrixFunction, (a,)::Tuple{DTerm}) = tensorrank(a) == 0 ? scalar_rule(f, a) : matrix_rule(f, a)

# matrix functions are the identity on scalars unless specified otherwise
scalar_rule(f, a) = a
scalar_rule(::typeof(asym), a) = ZERO
scalar_rule(::typeof(cof), a) = ONE
scalar_rule(::Union{typeof(gram), typeof(cogram)}, a) = a^2
# scalar inverses are powers
scalar_rule(::typeof(inv), a) = a^Literal(-1)

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

# sums and products

# coefficients are exact rationals, or floating-point and complex numbers
exact(x::Union{Integer, Rational}) = Rational{Int}(x)
exact(x::Number) = x

ispositive(x::Real) = x > 0
ispositive(x::Number) = false
isnegative(x::Real) = x < 0
isnegative(x::Number) = false

# the numeric power `x ^ p`, exact for integer powers
numpow(x, p) = isinteger(p) ? exact(x)^Int(p) : exact(x)^p

# the term `t` scaled by the number `c`, and the power `t ^ p`
scaled(c, t) = prodof(c, (t => ONE,), tensorrank(t))
power(t, p) = prodof(1, (t => p,), tensorrank(t))

# the entries of the dictionary `d` sorted by their keys, with the values mapped by `f`
function sorteddict(d, f = identity)
    entries = sort!(collect(d); by = first, lt = <ₛ)
    return LittleDict(DTerm[first(e) for e in entries], DTerm[f(last(e)) for e in entries])
end

# a sum is a call of `+` of its terms in order with the constant last, and a product is a call of `*` of its numeric
# factor and the powers of its bases, or a call of `^` if it is a single power, which would otherwise be its own argument
ispower(term) = isprod(term) && isone(term.coeff) && length(term.dict) == 1
addmul_operation(term) = ispower(term) ? Fun(^) : Fun(term.op)

function addmul_arguments(term)
    (; op, coeff, dict) = term
    if op === (+)
        args = DTerm[isstaticone(v) ? k : scaled(value(v), k) for (k, v) in dict]
        iszero(coeff) || push!(args, canonical_literal(coeff))
        return args
    end
    ispower(term) && return DTerm[only(dict.keys), only(dict.vals)]
    args = DTerm[isstaticone(p) ? b : power(b, p) for (b, p) in dict]
    isone(coeff) || pushfirst!(args, canonical_literal(coeff))
    return args
end

# the sum or the product with `f` applied to its terms or to its bases and powers, for maps commuting with the numeric
# coefficients, e.g. lowering, so that the arguments aren't built
mapentries(f, term) = mapentries(f, term.op, term)

function mapentries(f, ::typeof(+), term)
    (; coeff, dict) = term
    keys = DTerm[f(k) for k in dict.keys]
    all(i -> keys[i] === dict.keys[i], eachindex(keys)) && return term
    return sumof(coeff, zip(keys, map(value, dict.vals)), tensorrank(first(keys)))
end

function mapentries(f, ::typeof(*), term)
    (; coeff, dict) = term
    keys, vals = DTerm[f(k) for k in dict.keys], DTerm[f(v) for v in dict.vals]
    all(i -> keys[i] === dict.keys[i] && vals[i] === dict.vals[i], eachindex(keys)) && return term
    return prodof(coeff, zip(keys, vals), maximum(tensorrank, keys))
end

# a sum is a key of a sum only if it is scaled, i.e. its coefficient is not ±1
isunscaled(key, c) = issum(key) && isone(abs(c))

# the canonical sum `coeff + Σ c * t` of the pairs `t => c` with numeric coefficients `c`
function sumof(coeff, pairs, rank)
    terms = Dict{DTerm, Number}()
    c0 = exact(coeff)
    for (t, c) in pairs
        c0 = add!(terms, c0, t, exact(c))
    end
    # sums with the coefficient ±1 are flattened, all at once so that the result doesn't depend on the order
    while true
        unscaled = [k => c for (k, c) in terms if isunscaled(k, c)]
        isempty(unscaled) && break
        foreach(((k, _),) -> delete!(terms, k), unscaled)
        for (k, c) in unscaled
            c0 = add!(terms, c0, k, c)
        end
    end
    isempty(terms) && return iszero(c0) ? 𝒪(rank) : canonical_literal(c0)
    if iszero(c0) && length(terms) == 1
        # a single term is the term itself, or the product with its coefficient
        k, c = only(terms)
        return isone(c) ? k : scaled(c, k)
    end
    return packed(AddMul(+, canonical_number(c0), sorteddict(terms, canonical_literal), rank))
end

# add the term `t` with the coefficient `c` to `terms`, and return the constant term
function add!(terms, c0, t, c)
    # zero tensors vanish from the sum
    (iszero(c) || tensorrank(t) > 0 && isstaticzero(t)) && return c0
    return @match t begin
        Literal(x) => c0 + c * exact(x)
        # a sum scaled by a number other than ±1 is the product with the number, which keeps the sum as a term with
        # a positive leading coefficient
        AddMul(op, _, _) && if op === (+) && !isone(abs(c)) end => add!(terms, c0, scaled(c, t), 1)
        # the terms of a sum are collected, but not the terms of its scaled sums
        AddMul(op, s0, dict) && if op === (+) end => begin
            for (k, v) in dict
                addentry!(terms, k, c * exact(value(v)))
            end
            c0 + c * s0
        end
        # the coefficient of a product is collected, a scaled sum is kept as a term
        AddMul(_, p, dict) && if !isone(p) end => (addentry!(terms, prodof(1, dict, tensorrank(t)), c * p); c0)
        _ => (addentry!(terms, t, c); c0)
    end
end

function addentry!(terms, k, c)
    c = get(terms, k, 0) + c
    if iszero(c)
        delete!(terms, k)
    else
        terms[k] = c
    end
    return
end

# the canonical product `coeff * Π t ^ p` of the pairs `t => p`
function prodof(coeff, pairs, rank)
    powers = Dict{DTerm, DTerm}()
    c = exact(coeff)
    for (t, p) in pairs
        c = mul!(powers, c, t, p)
    end
    iszero(c) && return 𝒪(rank)
    isempty(powers) && return canonical_literal(c)
    if length(powers) == 1
        b, p = only(powers)
        if p ==ₛ ONE
            # a single base is the base itself, and negations and ±1 scalings of sums are distributed
            isone(c) && return b
            issum(b) && isone(abs(c)) && return sumof(0, (b => c,), rank)
        end
    end
    return packed(AddMul(*, canonical_number(c), sorteddict(powers), rank))
end

# multiply `powers` by the factor `t ^ p`, and return the numeric coefficient
function mul!(powers, c, t, p)
    isstaticzero(p) && return c
    isliteral(t) && isliteral(p) && return c * numpow(value(t), value(p))
    # zero tensors annihilate the product
    tensorrank(t) > 0 && isstaticzero(t) && return zero(c)
    integer = isinteger(p)
    @match t begin
        # the bases of a product are collected, and its coefficient unless it is negative and the power is not an integer
        AddMul(op, q, dict) && if op === (*) && (integer || ispositive(q)) end => begin
            isone(q) || (c = mul!(powers, c, canonical_literal(q), p))
            for (b, e) in dict
                c = mul!(powers, c, b, e * p)
            end
            return c
        end
        # integer powers of sums with a negative leading coefficient are powers of their negation
        AddMul(op, _, dict) && if op === (+) && integer && isnegative(value(first(dict.vals))) end => begin
            c *= numpow(-1, value(p))
            # negated directly, since `scaled` would normalise the sign of the sum again
            t = sumof(0, (t => -1,), tensorrank(t))
        end
        _ => nothing
    end
    old = get(powers, t, nothing)
    p = isnothing(old) ? p : old + p
    if isstaticzero(p)
        delete!(powers, t)
    else
        powers[t] = p
    end
    return c
end

# a sum or a product of tensors in component form is in component form
function packed(term)
    dims = packed_dims(term.dict.keys)
    isnothing(dims) && return term
    return pack(I -> component(term, I), Kind.None(), tensorrank(term), dims)
end
