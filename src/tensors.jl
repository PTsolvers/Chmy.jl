"""
    components(term, dims)

Return `term` in component form in `dims` spatial dimensions: tensors are a [`pack`](@ref) of their components
`term[I...]`, and every subterm is expanded the same way. Calls are evaluated in `dims` dimensions with the component
forms of their arguments, see [`evaluate`](@ref), so that nested operations are expanded at once, e.g. `sym(grad(u))`,
or scalars `tr(A)` and `divg(grad(u))`. Scalar calls without such a rule are kept with their arguments in component
form, and tensor-valued calls without a rule throw an error. Terms already in component form are returned unchanged.
"""
function components(t::DTerm, dims::Integer)::DTerm
    dims > 0 || throw(ArgumentError("dims must be a positive integer"))
    expand(x) = components(x, dims)
    return @match t begin
        DExpr(Pack(_, d, _), _) => d == dims ? walkargs(expand, t) : pack_dims_error(d, dims)
        Tensor(rank, _, kind, _) => pack(I -> component(t, I), kind, rank, dims)
        # rules build their results from the expanded arguments, and scalar calls without a rule are kept with them
        DExpr(Call(op), args) => begin
            expanded = map(expand, args)
            result = evaluate(op, expanded, dims)::Union{Nothing, DTerm}
            if isnothing(result)
                tensorrank(t) == 0 || expansion_error(op)
                result = expanded === args ? t : makecall(op, expanded)
            end
            result
        end
        # sums and products of tensors in component form are in component form, or zero tensors
        AddMul(_, _, _) => begin
            mapped = mapentries(expand, t)
            tensorrank(t) == 0 || ispack(mapped) ? mapped : expand(mapped)
        end
        # components are read from their argument in component form
        _ => walkargs(expand, t)
    end
end

@noinline function pack_dims_error(d, dims)
    throw(DimensionMismatch(LazyString("tensor in component form has ", d, " dimensions, expected ", dims)))
end
@noinline function expansion_error(op)
    throw(
        ArgumentError(
            LazyString(
                "tensor-valued operator '", op, "' has no component form, ",
                "see `Chmy.component(op, args, I)` and `Chmy.evaluate(op, args, dims)`",
            )
        )
    )
end

"""
    pack(array)
    pack(f, kind, rank, dims)

Construct a tensor in component form. The first method stores the entries of `array`, whose axes must
have equal lengths, converting numbers to literals. The second method calls `f(I)` at each canonical index
tuple `I` of a tensor with symmetry `kind` and rank `rank` in `dims` dimensions; a tensor of rank zero is its only
entry `f(())`.
"""
function pack(f::F, kind::TensorKind, rank::Integer, dims::Integer) where {F}
    iszero(rank) && return f(())::DTerm
    data = DTerm[]
    sizehint!(data, ncomponents(kind, dims, rank))
    foreach_component(kind, dims, rank) do I
        push!(data, f(I))
    end
    return makepack(rank, dims, kind, Tuple(data))
end
function pack(array::AbstractArray)
    allequal(size(array)) || throw(DimensionMismatch("all axes of a packed array must have the same length"))
    # column-major order matches the linear indices of the tensors with no symmetry
    data = map(asterm, vec(array))
    return makepack(ndims(array), size(array, 1), Kind.None(), Tuple(data))
end

asterm(x::DTerm) = x
asterm(x::Number) = canonical_literal(x)
asterm(v::Union{AnyNormal, AnyTangent}) = v()

ncomponents(::Kind.None, dims, rank) = dims^rank
ncomponents(::Kind.Sym, dims, rank) = binomial(dims + rank - 1, rank)
ncomponents(::Kind.Alt, dims, rank) = binomial(dims, rank)
ncomponents(::Kind.Diag, dims, rank) = iszero(rank) ? 1 : dims

# the entry of a tensor in component form at the index tuple `I`
pack_entry(kind::Union{Kind.None, Kind.Sym}, dims, data, I) = data[linear_index(kind, dims, I)]
function pack_entry(kind::Kind.Alt, dims, data, I)
    allunique(I) || return ZERO
    v = data[linear_index(kind, dims, I)]
    return iseven(inversion_count(I)) ? v : -v
end
pack_entry(::Kind.Diag, dims, data, I) = allequal(I) ? data[I[1]] : ZERO

# the component `I` of a call acting on every component of its tensor arguments alike, e.g. broadcasting
componentwise(op::Operator, args, I) = op(map(x -> componentof(x, I), args)...)
componentof(x, I) = tensorrank(x) == 0 ? x : component(x, I)

component(op::BroadcastedFun, args, I) = componentwise(Fun(op), args, I)
for op in (:+, :-, :*, :/, ://, :÷)
    @eval component(::typeof($op), args, I) = componentwise(Fun($op), args, I)
end

function component(::typeof(⊗), (a, b)::Tuple{DTerm, DTerm}, I)
    ra = tensorrank(a)
    return component(a, I[1:ra]) * component(b, I[(ra + 1):end])
end

# Chmy's symbolic adjoint is a transpose, without complex conjugation
component(::typeof(adjoint), (t,)::Tuple{DTerm}, (i, j)) = t[j, i]
component(::typeof(diag), (t,)::Tuple{DTerm}, (i,)) = t[i, i]
component(::typeof(sym), (t,)::Tuple{DTerm}, (i, j)) = (1 // 2) * (t[i, j] + t[j, i])
component(::typeof(asym), (t,)::Tuple{DTerm}, (i, j)) = (1 // 2) * (t[i, j] - t[j, i])

# only the stored components of the symmetric and antisymmetric parts are evaluated
evaluate(f::typeof(sym), args::Tuple{DTerm}, dims) = pack(I -> component(f, args, I), Kind.Sym(), 2, dims)
evaluate(f::typeof(asym), args::Tuple{DTerm}, dims) = pack(I -> component(f, args, I), Kind.Alt(), 2, dims)

evaluate(::typeof(⋅), (a, b)::Tuple{DTerm, DTerm}, dims) = contract_components(a, b, 1, dims)
evaluate(::typeof(⊡), (a, b)::Tuple{DTerm, DTerm}, dims) = contract_components(a, b, 2, dims)

function evaluate(::typeof(×), (a, b)::Tuple{DTerm, DTerm}, dims)
    dims == 3 || throw(ArgumentError("cross product requires dimension 3"))
    data = (
        a[2] * b[3] - a[3] * b[2],
        a[3] * b[1] - a[1] * b[3],
        a[1] * b[2] - a[2] * b[1],
    )
    return makepack_unchecked(1, 3, Kind.None(), data)
end

evaluate(::typeof(tr), (t,)::Tuple{DTerm}, dims) = sum(t[i, i] for i in 1:dims)

function evaluate(::typeof(det), (t,)::Tuple{DTerm}, dims)
    1 <= dims <= 3 || throw(ArgumentError("determinant is supported only in dimensions 1–3"))
    dims == 1 && return t[1, 1]
    dims == 2 && return t[1, 1] * t[2, 2] - t[1, 2] * t[2, 1]
    #! format: off
    return t[1, 1] * (t[2, 2] * t[3, 3] - t[2, 3] * t[3, 2]) +
           t[1, 2] * (t[2, 3] * t[3, 1] - t[2, 1] * t[3, 3]) +
           t[1, 3] * (t[2, 1] * t[3, 2] - t[2, 2] * t[3, 1])
    #! format: on
end

function evaluate(::typeof(cof), (t,)::Tuple{DTerm}, dims)
    1 <= dims <= 3 || throw(ArgumentError("cofactor matrix is supported only in dimensions 1–3"))
    dims == 1 && return literal_pack(ONE, dims, 2)
    return pack(Kind.None(), 2, dims) do (i, j)
        dims == 2 || return cofactor3(t, i, j)
        return i == j ? t[3 - i, 3 - j] : -t[3 - i, 3 - j]
    end
end

# the adjugate and the determinant are evaluated right away, since `t` is in component form
evaluate(::typeof(inv), (t,)::Tuple{DTerm}, dims) = adj(t) / det(t)

# the Gramian `t' ⋅ t` and the co-Gramian `t ⋅ t'`
function evaluate(::typeof(gram), (t,)::Tuple{DTerm}, dims)
    return pack(((i, j),) -> sum(t[k, i] * t[k, j] for k in 1:dims), Kind.Sym(), 2, dims)
end
function evaluate(::typeof(cogram), (t,)::Tuple{DTerm}, dims)
    return pack(((i, j),) -> sum(t[i, k] * t[j, k] for k in 1:dims), Kind.Sym(), 2, dims)
end

function linear_index(::Kind.Sym, dims, I)
    J = sort(I)
    idx = 1
    for k in eachindex(J)
        idx += binomial(J[k] + k - 2, k)
    end
    return idx
end
function linear_index(::Kind.Alt, dims, I)
    J = sort(I)
    idx = 1
    for k in eachindex(J)
        idx += binomial(J[k] - 1, k)
    end
    return idx
end
function linear_index(::Kind.None, dims, I)
    idx = 1
    stride = 1
    for i in I
        idx += (i - 1) * stride
        stride *= dims
    end
    return idx
end
linear_index(::Kind.Diag, dims, I) = I[1]

foreach_component(f, kind, dims, rank) = visit_components((I, _) -> (f(I); true), kind, dims, rank)

visit_components(f, ::Kind.Diag, dims, rank) = diagonal(f, dims, rank)
visit_components(f, ::Kind.Sym, dims, rank) = nondecreasing(f, dims, rank)
visit_components(f, ::Kind.Alt, dims, rank) = increasing(f, dims, rank)
visit_components(f, ::Kind.None, dims, rank) = cartesian(f, dims, rank)

function literal_pack(x::DTerm, dims, rank)
    kind = rank < 2 ? Kind.None() : Kind.Diag()
    return makepack_unchecked(rank, dims, kind, ntuple(_ -> x, ncomponents(kind, dims, rank)))
end

# a tensor in component form with the most structured storage supported by its entries
function canonical_pack(rank, dims, kind, data)
    detected = detect_kind(kind, dims, rank, data)
    kind == detected && return makepack_unchecked(rank, dims, kind, data)
    return pack(I -> pack_entry(kind, dims, data, I), detected, rank, dims)
end

function detect_kind(kind::TensorKind, dims, rank, data)
    rank < 2 && return kind
    kind == Kind.Diag() && return kind
    if kind == Kind.Alt()
        # Empty alternating storage (rank > dims) also represents a zero tensor.
        return all(isstaticzero, data) ? Kind.Diag() : kind
    end
    isdiag(kind, dims, rank, data) && return Kind.Diag()
    kind == Kind.Sym() && return kind
    issym(dims, rank, data) && return Kind.Sym()
    isalt(dims, rank, data) && return Kind.Alt()
    return kind
end

function issym(dims, rank, data)
    rank < 2 && return false
    return cartesian(dims, rank) do I, state
        J = sort(I)
        i = linear_index(Kind.None(), dims, I)
        j = linear_index(Kind.None(), dims, J)
        return state && data[i] ==ₛ data[j]
    end
end

function isalt(dims, rank, data)
    rank < 2 && return false
    return cartesian(dims, rank) do I, state
        x = data[linear_index(Kind.None(), dims, I)]
        allunique(I) || return state && isstaticzero(x)
        y = data[linear_index(Kind.None(), dims, sort(I))]
        return state && x ==ₛ (iseven(inversion_count(I)) ? y : -y)
    end
end

function isdiag(kind::TensorKind, dims, rank, data)
    rank < 2 && return false
    return visit_components(kind, dims, rank) do I, state
        i = linear_index(kind, dims, I)
        return state && (allequal(I) || isstaticzero(data[i]))
    end
end

function contraction_component(a, b, I, n, dims)
    free = tensorrank(a) - n
    left, right = I[1:free], I[(free + 1):end]
    terms = DTerm[]
    sizehint!(terms, ncomponents(Kind.None(), dims, n))
    foreach_component(Kind.None(), dims, n) do K
        push!(terms, a[left..., K...] * b[K..., right...])
    end
    return sum(terms)
end

function contract_components(a, b, n, dims)
    rank = tensorrank(a) + tensorrank(b) - 2n
    return pack(I -> contraction_component(a, b, I, n, dims), Kind.None(), rank, dims)
end

function cofactor3(t, row, col)
    # Remaining rows and columns in increasing order.
    r1, r2 = row == 1 ? 2 : 1, row == 3 ? 2 : 3
    c1, c2 = col == 1 ? 2 : 1, col == 3 ? 2 : 3
    minor = t[r1, c1] * t[r2, c2] - t[r1, c2] * t[r2, c1]
    return iseven(row + col) ? minor : -minor
end
