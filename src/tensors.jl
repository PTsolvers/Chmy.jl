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

# the component `I` of a tensor in component form, or the scalar itself
component(t::DTerm, ::Tuple{}) = t
component(t::DTerm, I::Tuple) = t[I...]

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
asterm(x::Number) = Literal(x)

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

# the same tensor with other entries
repack(t::DTerm, data) = makepack_unchecked(tensorrank(t), packdims(t), tensorkind(t), Tuple(data))

# `evaluate(op, args)` evaluates a pointwise function of tensors in component form; returning `nothing` keeps
# the call `op(args...)` as is
evaluate(op::Fun, args::Tuple{Vararg{DTerm}}) = evaluate(op.f, args)
evaluate(f, args::Tuple{Vararg{DTerm}}) = nothing

function evaluate(op::BroadcastedFun, args::Tuple{Vararg{DTerm}})
    f = Fun(op)
    k = findfirst(x -> tensorrank(x) > 0, args)
    # broadcasting over scalars is a scalar call
    isnothing(k) && return makecall_unchecked(f, args)
    rank, dims = tensorrank(args[k]), packdims(args[k])
    return pack(broadcast_kind(args, rank), rank, dims) do I
        makecall_unchecked(f, map(x -> tensorrank(x) == 0 ? x : x[I...], args))
    end
end

function broadcast_kind(args, rank)
    # arbitrary functions preserve permutation symmetry
    sym_or_diag = rank >= 2 && all(args) do x
        tensorrank(x) == 0 || tensorkind(x) isa Union{Kind.Sym, Kind.Diag}
    end
    return sym_or_diag ? Kind.Sym() : Kind.None()
end

function evaluate(::typeof(+), args::Tuple{Vararg{DTerm}})
    tensorrank(first(args)) == 0 && return nothing
    length(args) == 1 && return only(args)
    t = first(args)
    kind = foldl(promote_kind, map(tensorkind, args))
    terms = DTerm[]
    sizehint!(terms, length(args))
    if all(x -> tensorkind(x) == kind, args)
        return repack(t, (component_sum_at!(terms, args, i) for i in eachindex(entries(t))))
    end
    return pack(kind, tensorrank(t), packdims(t)) do I
        component_sum_at!(terms, args, I)
    end
end
function evaluate(::typeof(-), args::Tuple{Vararg{DTerm}})
    t = first(args)
    tensorrank(t) == 0 && return nothing
    length(args) == 1 && return repack(t, map(-, entries(t)))
    return component_arithmetic(-, args...)
end
function evaluate(::typeof(*), args::Tuple{Vararg{DTerm}})
    # at most one factor is a tensor
    k = findfirst(x -> tensorrank(x) > 0, args)
    isnothing(k) && return nothing
    tensor = args[k]
    factors = Tuple(args[j] for j in eachindex(args) if j != k)
    isempty(factors) && return tensor
    scalar = length(factors) == 1 ? only(factors) : makecall_unchecked(Fun(*), factors)
    return repack(tensor, map(x -> scalar * x, entries(tensor)))
end
for op in (:/, ://, :÷)
    @eval function evaluate(::typeof($op), args::Tuple{Vararg{DTerm}})
        a, b = args
        # the denominator is scalar by the expression's construction contract
        tensorrank(a) == 0 && return nothing
        return repack(a, map(x -> $op(x, b), entries(a)))
    end
end

evaluate(::typeof(⋅), args::Tuple{DTerm, DTerm}) = contract_components(args..., 1)
evaluate(::typeof(⊡), args::Tuple{DTerm, DTerm}) = contract_components(args..., 2)

function evaluate(::typeof(⊗), (a, b)::Tuple{DTerm, DTerm})
    ra, rb = tensorrank(a), tensorrank(b)
    return pack(Kind.None(), ra + rb, packdims(a)) do I
        a[I[1:ra]...] * b[I[(ra + 1):end]...]
    end
end

function evaluate(::typeof(×), (a, b)::Tuple{DTerm, DTerm})
    packdims(a) == 3 || throw(ArgumentError("cross product requires dimension 3"))
    data = (
        a[2] * b[3] - a[3] * b[2],
        a[3] * b[1] - a[1] * b[3],
        a[1] * b[2] - a[2] * b[1],
    )
    return makepack_unchecked(1, 3, Kind.None(), data)
end

for op in (:transpose, :adjoint, :tr, :diag, :det, :cof, :inv, :sym, :asym, :gram, :cogram)
    @eval function evaluate(f::typeof($op), args::Tuple{DTerm})
        t = only(args)
        tensorrank(t) == 0 && return nothing
        return matrix_components(f, t)
    end
end

# transpose of a rank-2 tensor, preserving its known kind
function matrix_components(::typeof(transpose), t)
    tensorkind(t) isa Union{Kind.Sym, Kind.Diag} && return t
    tensorkind(t) == Kind.Alt() && return -t
    return pack(I -> t[I[2], I[1]], Kind.None(), 2, packdims(t))
end

# Chmy's symbolic adjoint is a transpose, without complex conjugation
matrix_components(::typeof(adjoint), t) = matrix_components(transpose, t)

function matrix_components(::typeof(tr), t)
    tensorkind(t) == Kind.Alt() && return ZERO
    return component_sum(DTerm[t[i, i] for i in 1:packdims(t)])
end

matrix_components(::typeof(diag), t) = pack(I -> t[I[1], I[1]], Kind.None(), 1, packdims(t))

function matrix_components(::typeof(det), t)
    d = packdims(t)
    1 <= d <= 3 || throw(ArgumentError("determinant is supported only in dimensions 1–3"))
    d == 1 && return t[1, 1]
    kind = tensorkind(t)
    if kind == Kind.Diag()
        return makecall_unchecked(Fun(*), entries(t))
    elseif kind == Kind.Alt()
        return d == 2 ? t[1, 2]^2 : ZERO
    elseif d == 2
        return kind == Kind.Sym() ?
            t[1, 1] * t[2, 2] - t[1, 2]^2 :
            t[1, 1] * t[2, 2] - t[1, 2] * t[2, 1]
    end
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
    kind = tensorkind(t)
    return pack(d == 3 && kind == Kind.Alt() ? Kind.Sym() : kind, 2, d) do (i, j)
        if kind == Kind.Diag()
            comps = DTerm[t[k, k] for k in 1:d if k != i]
            length(comps) == 1 && return only(comps)
            return makecall_unchecked(Fun(*), Tuple(comps))
        elseif d == 2
            return i == j ? t[3 - i, 3 - j] : -t[3 - i, 3 - j]
        end
        return cofactor3(t, i, j)
    end
end

# the adjugate and the determinant are evaluated right away, since `t` is in component form
matrix_components(::typeof(inv), t) = adj(t) / det(t)

function matrix_components(::typeof(sym), t)
    kind = tensorkind(t)
    kind isa Union{Kind.Sym, Kind.Diag} && return t
    kind == Kind.Alt() && return literal_pack(ZERO, packdims(t), 2)
    return pack(Kind.Sym(), 2, packdims(t)) do (i, j)
        i == j ? t[i, i] : (1 // 2) * (t[i, j] + t[j, i])
    end
end

function matrix_components(::typeof(asym), t)
    kind = tensorkind(t)
    kind == Kind.Alt() && return t
    kind isa Union{Kind.Sym, Kind.Diag} && return literal_pack(ZERO, packdims(t), 2)
    return pack(Kind.Alt(), 2, packdims(t)) do (i, j)
        (1 // 2) * (t[i, j] - t[j, i])
    end
end

# the Gramian `t' ⋅ t`
function matrix_components(::typeof(gram), t)
    d, diagonal = packdims(t), tensorkind(t) == Kind.Diag()
    return pack(diagonal ? Kind.Diag() : Kind.Sym(), 2, d) do (i, j)
        diagonal && return t[i, i]^2
        component_sum(DTerm[t[k, i] * t[k, j] for k in 1:d])
    end
end

# the co-Gramian `t ⋅ t'`
function matrix_components(::typeof(cogram), t)
    d, diagonal = packdims(t), tensorkind(t) == Kind.Diag()
    return pack(diagonal ? Kind.Diag() : Kind.Sym(), 2, d) do (i, j)
        diagonal && return t[i, i]^2
        component_sum(DTerm[t[i, k] * t[j, k] for k in 1:d])
    end
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

# the most structured storage supported by the entries of a tensor in component form
function canonicalize_pack(t::DTerm, rank, dims, kind, data)
    detected = detect_kind(kind, dims, rank, data)
    kind == detected && return t
    kind == Kind.Alt() && detected == Kind.Diag() && return literal_pack(ZERO, dims, rank)
    return pack(detected, rank, dims) do I
        data[linear_index(kind, dims, I)]
    end
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

promote_kind(::T, ::T) where {T <: TensorKind} = T()
promote_kind(::Kind.None, ::Kind.None) = Kind.None()

promote_kind(::TensorKind, ::Kind.None) = Kind.None()
promote_kind(::Kind.Sym, ::Kind.Diag) = Kind.Sym()
promote_kind(::Kind.Alt, ::Kind.Diag) = Kind.None()
promote_kind(::Kind.Alt, ::Kind.Sym) = Kind.None()
promote_kind(::Kind.Diag, ::Kind.Sym) = Kind.Sym()

promote_kind(a, b) = promote_kind(b, a)

function component_arithmetic(f::F, a::DTerm, b::DTerm) where {F}
    if tensorkind(a) == tensorkind(b)
        data = map(entries(a), entries(b)) do x, y
            makecall_unchecked(Fun(f), (x, y))
        end
        return repack(a, data)
    end
    kind = promote_kind(tensorkind(a), tensorkind(b))
    return pack(kind, tensorrank(a), packdims(a)) do I
        makecall_unchecked(Fun(f), (a[I...], b[I...]))
    end
end

function component_sum_at!(terms, tensors, index)
    empty!(terms)
    for t in tensors
        # linear data indices are used only when all storage kinds match
        x = index isa Int ? entries(t)[index] : t[index...]
        push!(terms, x)
    end
    return component_sum(terms)
end

function component_sum(terms)
    n = length(terms)
    n == 1 && return only(terms)
    # Base.ntuple unrolls lengths up to ten, avoiding element boxing when
    # copying a short argument buffer
    args = n <= 10 ? ntuple(i -> terms[i], n) : Tuple(terms)
    return makecall_unchecked(Fun(+), args)
end

function contraction_component(a, b, I, n)
    free = tensorrank(a) - n
    left, right = I[1:free], I[(free + 1):end]
    terms = DTerm[]
    sizehint!(terms, ncomponents(Kind.None(), packdims(a), n))
    foreach_component(Kind.None(), packdims(a), n) do K
        push!(terms, a[left..., K...] * b[K..., right...])
    end
    return component_sum(terms)
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
