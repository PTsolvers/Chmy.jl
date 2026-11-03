"""
    components(term, dims)

Return `term` in component form in `dims` spatial dimensions, i.e. a [`pack`](@ref) of the components `term[I...]`,
which are not expanded further. Scalars and tensors already in component form are returned unchanged.
"""
function components(t::DTerm, dims::Integer)::DTerm
    dims > 0 || throw(ArgumentError("dims must be a positive integer"))
    tensorrank(t) == 0 && return t
    return @match t begin
        DExpr(Pack(_, d, _), _) => d == dims ? t : pack_dims_error(d, dims)
        Tensor(rank, _, kind, _) => pack(I -> makecomp_unchecked(t, I), kind, rank, dims)
        ZeroTensor(rank) => literal_pack(ZERO, dims, rank)
        IdTensor(rank) => literal_pack(ONE, dims, rank)
        # the kind of a tensor-valued expression is not known
        _ => pack(I -> makecomp_unchecked(t, I), Kind.None(), tensorrank(t), dims)
    end
end

@noinline function pack_dims_error(d, dims)
    throw(DimensionMismatch(LazyString("tensor in component form has ", d, " dimensions, expected ", dims)))
end
@noinline function expansion_error(op)
    throw(ArgumentError(LazyString("tensor-valued operator '", op, "' has no expansion to the component form")))
end

"""
    pack(array)
    pack(f, kind, rank, dims)

Construct a tensor in component form. The first method stores the entries of `array`, whose axes must
have equal lengths, converting numbers to literals. The second method calls `f(I)` at each canonical index
tuple `I` of a tensor with symmetry `kind` and rank `rank` in `dims` dimensions.
"""
function pack(f::F, kind::TensorKind, rank::Integer, dims::Integer) where {F}
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

# `evaluate_packed(op, args)` evaluates a function of tensors in component form, or returns `nothing` if the call has no
# expansion to the component form
evaluate_packed(op::Fun, args::Tuple{Vararg{DTerm}}) = evaluate_packed(op.f, args)
evaluate_packed(op, args::Tuple{Vararg{DTerm}}) = nothing

evaluate_packed(op::BroadcastedFun, args::Tuple{Vararg{DTerm}}) = pointwise(Fun(op), args)
for op in (:+, :-, :*, :/, ://, :÷)
    @eval evaluate_packed(::typeof($op), args::Tuple{Vararg{DTerm}}) = pointwise(Fun($op), args)
end

# a pointwise function of tensors in component form and scalars, whose symmetry is detected from the entries
function pointwise(op::Operator, args)
    t = args[findfirst(x -> tensorrank(x) > 0, args)]
    return pack(Kind.None(), tensorrank(t), packdims(t)) do I
        op(map(x -> tensorrank(x) == 0 ? x : x[I...], args)...)
    end
end

evaluate_packed(::typeof(⋅), args::Tuple{DTerm, DTerm}) = contract_components(args..., 1)
evaluate_packed(::typeof(⊡), args::Tuple{DTerm, DTerm}) = contract_components(args..., 2)

function evaluate_packed(::typeof(⊗), (a, b)::Tuple{DTerm, DTerm})
    ra, rb = tensorrank(a), tensorrank(b)
    return pack(Kind.None(), ra + rb, packdims(a)) do I
        a[I[1:ra]...] * b[I[(ra + 1):end]...]
    end
end

function evaluate_packed(::typeof(×), (a, b)::Tuple{DTerm, DTerm})
    packdims(a) == 3 || throw(ArgumentError("cross product requires dimension 3"))
    data = (
        a[2] * b[3] - a[3] * b[2],
        a[3] * b[1] - a[1] * b[3],
        a[1] * b[2] - a[2] * b[1],
    )
    return makepack_unchecked(1, 3, Kind.None(), data)
end

for op in (:adjoint, :tr, :diag, :det, :cof, :inv, :sym, :asym, :gram, :cogram)
    @eval evaluate_packed(f::typeof($op), (t,)::Tuple{DTerm}) = matrix_components(f, t)
end

# Chmy's symbolic adjoint is a transpose, without complex conjugation
matrix_components(::typeof(adjoint), t) = pack(I -> t[I[2], I[1]], Kind.None(), 2, packdims(t))

matrix_components(::typeof(tr), t) = +((t[i, i] for i in 1:packdims(t))...)

matrix_components(::typeof(diag), t) = pack(I -> t[I[1], I[1]], Kind.None(), 1, packdims(t))

function matrix_components(::typeof(det), t)
    d = packdims(t)
    1 <= d <= 3 || throw(ArgumentError("determinant is supported only in dimensions 1–3"))
    d == 1 && return t[1, 1]
    d == 2 && return t[1, 1] * t[2, 2] - t[1, 2] * t[2, 1]
    #! format: off
    return t[1, 1] * (t[2, 2] * t[3, 3] - t[2, 3] * t[3, 2]) +
           t[1, 2] * (t[2, 3] * t[3, 1] - t[2, 1] * t[3, 3]) +
           t[1, 3] * (t[2, 1] * t[3, 2] - t[2, 2] * t[3, 1])
    #! format: on
end

function matrix_components(::typeof(cof), t)
    d = packdims(t)
    1 <= d <= 3 || throw(ArgumentError("cofactor matrix is supported only in dimensions 1–3"))
    d == 1 && return literal_pack(ONE, d, 2)
    return pack(Kind.None(), 2, d) do (i, j)
        d == 2 || return cofactor3(t, i, j)
        return i == j ? t[3 - i, 3 - j] : -t[3 - i, 3 - j]
    end
end

# the adjugate and the determinant are evaluated right away, since `t` is in component form
matrix_components(::typeof(inv), t) = adj(t) / det(t)

matrix_components(::typeof(sym), t) = pack(((i, j),) -> (1 // 2) * (t[i, j] + t[j, i]), Kind.Sym(), 2, packdims(t))
matrix_components(::typeof(asym), t) = pack(((i, j),) -> (1 // 2) * (t[i, j] - t[j, i]), Kind.Alt(), 2, packdims(t))

# the Gramian `t' ⋅ t` and the co-Gramian `t ⋅ t'`
function matrix_components(::typeof(gram), t)
    d = packdims(t)
    return pack(((i, j),) -> +((t[k, i] * t[k, j] for k in 1:d)...), Kind.Sym(), 2, d)
end
function matrix_components(::typeof(cogram), t)
    d = packdims(t)
    return pack(((i, j),) -> +((t[i, k] * t[j, k] for k in 1:d)...), Kind.Sym(), 2, d)
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
        J = sort(I)
        i = linear_index(Kind.None(), dims, I)
        x = data[i]
        if !allunique(J)
            return state && isstaticzero(x)
        else
            j = linear_index(Kind.None(), dims, J)
            y = data[j]
            if iseven(inversion_count(I))
                return state && x ==ₛ y
            else
                return state && isnegof(x, y)
            end
        end
    end
end

function isdiag(kind::TensorKind, dims, rank, data)
    rank < 2 && return false
    return visit_components(kind, dims, rank) do I, state
        i = linear_index(kind, dims, I)
        return state && (allequal(I) || isstaticzero(data[i]))
    end
end

function contraction_component(a, b, I, n)
    free = tensorrank(a) - n
    left, right = I[1:free], I[(free + 1):end]
    terms = DTerm[]
    sizehint!(terms, ncomponents(Kind.None(), packdims(a), n))
    foreach_component(Kind.None(), packdims(a), n) do K
        push!(terms, a[left..., K...] * b[K..., right...])
    end
    return +(terms...)
end

function contract_components(a::DTerm, b::DTerm, n)
    rank = tensorrank(a) + tensorrank(b) - 2n
    rank == 0 && return contraction_component(a, b, (), n)
    return pack(I -> contraction_component(a, b, I, n), Kind.None(), rank, packdims(a))
end

function cofactor3(t, row, col)
    # Remaining rows and columns in increasing order.
    r1, r2 = row == 1 ? 2 : 1, row == 3 ? 2 : 3
    c1, c2 = col == 1 ? 2 : 1, col == 3 ? 2 : 3
    minor = t[r1, c1] * t[r2, c2] - t[r1, c2] * t[r2, c1]
    return iseven(row + col) ? minor : -minor
end
