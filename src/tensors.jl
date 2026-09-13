struct TensorComponents
    dims::Int
    rank::Int
    kind::TensorKind
    data::Vector{DTerm}
end

"""
    components(expr, dims)

Convert `expr` to scalar component form in `dims` spatial dimensions.
"""
function components(t::DTerm, dims::Integer)::Union{TensorComponents,DTerm}
    dims > 0 || throw(ArgumentError("dims must pe positive integer"))
    @match t begin
        Tensor(rank, _, kind, _) => begin
            iszero(rank) && return t
            return TensorComponents(kind, rank, dims) do I
                makecomp_unchecked(t, I)
            end
        end
        ZeroTensor(rank) => begin
            iszero(rank) && return ZERO
            return literal_components(ZERO, dims, rank)
        end
        IdTensor(rank) => begin
            iszero(rank) && return ONE
            return literal_components(ONE, dims, rank)
        end
        DExpr(Comp(I), (arg,)) => begin
            istensor(arg) && return t
            result = components(arg, dims)
            if result isa TensorComponents
                return result[I...]
            else
                return result
            end
        end
        DExpr(Call(op), args) => begin
            lowered = map(arg -> components(arg, dims), args)
            return evaluate(op, lowered, dims)
        end
        DExpr(_, args) => begin
            lowered = map(arg -> components(arg, dims), args)
            return DExpr(head(t), lowered::Tuple{Vararg{DTerm}})
        end
        _ => return t
    end
end

"""
    simplify(t::TensorComponents)

Simplify each stored scalar entry, then detect and extract the most structured storage supported by those entries.
"""
function simplify(t::TensorComponents)
    data = map(simplify, t.data)
    kind = detect_kind(t.kind, t.dims, t.rank, data)
    if t.kind == kind
        return TensorComponents(t.dims, t.rank, kind, data)
    end
    if t.kind == Kind.Alt() && kind == Kind.Diag()
        return literal_components(ZERO, t.dims, t.rank)
    end
    return TensorComponents(kind, t.rank, t.dims) do I
        data[linear_index(t.kind, t.dims, I)]
    end
end

"""
    TensorComponents(f, kind, rank, dims)

Make a component representation of tensor by calling `f(I)` at each canonical index tuple.
"""
function TensorComponents(f::F, kind::TensorKind, rank::Integer, dims::Integer) where {F}
    data = DTerm[]
    sizehint!(data, ncomponents(kind, dims, rank))
    foreach_component(kind, dims, rank) do I
        push!(data, f(I))
    end
    return TensorComponents(dims, rank, kind, data)
end

tensorrank(t::TensorComponents) = t.rank
tensorkind(t::TensorComponents) = t.kind

Base.ndims(t::TensorComponents) = t.dims

ncomponents(t::TensorComponents) = length(t.data)

components(t::TensorComponents) = t.data

isuniform(t::TensorComponents) = all(isuniform, t.data)

"""
    getindex(t::TensorComponents, inds...)

Return the component of a tensor.
"""
function Base.getindex(t::TensorComponents, I::Vararg{Integer,N}) where {N}
    N == t.rank || throw(ArgumentError("number of subscripts not matching tensor rank"))
    if t.kind == Kind.Sym()
        idx = linear_index(t.kind, t.dims, I)
        return t.data[idx]
    elseif t.kind == Kind.Alt()
        allunique(I) || return ZERO
        idx = linear_index(t.kind, t.dims, I)
        v = t.data[idx]
        return iseven(inversion_count(I)) ? v : -v
    elseif t.kind == Kind.Diag()
        return allequal(I) ? t.data[I[1]] : ZERO
    elseif t.kind == Kind.None()
        return t.data[linear_index(t.kind, t.dims, I)]
    else
        error("unreachable")
    end
end

Base.:-(a::TensorComponents, b::TensorComponents) = component_arithmetic(-, a, b)
Base.:-(a::TensorComponents) = TensorComponents(a.dims, a.rank, a.kind, map(-, a.data))

Base.:+(a::TensorComponents) = a
Base.:+(a::TensorComponents, b::TensorComponents) = component_arithmetic(+, a, b)
function Base.:+(a::TensorComponents, b::TensorComponents, c::TensorComponents, rest::TensorComponents...)
    tensors = (a, b, c, rest...)
    kind = foldl(promote_kind, map(tensorkind, tensors))
    terms = DTerm[]
    sizehint!(terms, length(tensors))
    if all(t -> t.kind == kind, tensors)
        data = DTerm[component_sum_at!(terms, tensors, i) for i in eachindex(a.data)]
        return TensorComponents(a.dims, a.rank, kind, data)
    end
    return TensorComponents(kind, a.rank, a.dims) do I
        component_sum_at!(terms, tensors, I)
    end
end

Base.:*(s::DTerm, t::TensorComponents) = TensorComponents(t.dims, t.rank, t.kind, DTerm[s * x for x in t.data])
Base.:*(s::Number, t::TensorComponents) = Literal(s) * t

for op in (:*, :/, ://, :÷)
    @eval begin
        function Base.$op(t::TensorComponents, s::DTerm)
            return TensorComponents(t.dims, t.rank, t.kind, DTerm[$op(x, s) for x in t.data])
        end
        Base.$op(t::TensorComponents, s::Number) = $op(t, Literal(s))
    end
end

"""
    ⋅(a::TensorComponents, b::TensorComponents)

Contract the last index of `a` with the first index of `b`.
"""
⋅(a::TensorComponents, b::TensorComponents) = contract_components(a, b, 1)

"""
    ⊡(a, b)

Contract the last two indices of `a` with the first two indices of `b`.
"""
⊡(a::TensorComponents, b::TensorComponents) = contract_components(a, b, 2)

"""
    ⊗(a, b)

Form the outer product, placing all indices of `a` before those of `b`.
"""
function ⊗(a::TensorComponents, b::TensorComponents)
    rank = a.rank + b.rank
    return TensorComponents(Kind.None(), rank, a.dims) do I
        a[I[1:a.rank]...] * b[I[(a.rank+1):end]...]
    end
end

"""
    cross(a::TensorComponents, b::TensorComponents)
    ×(a, b)

Compute the cross product of two three-dimensional component vectors.
"""
function ×(a::TensorComponents, b::TensorComponents)
    a.dims == 3 || throw(ArgumentError("cross product requires dimension 3"))
    data = DTerm[a[2] * b[3] - a[3] * b[2],
                 a[3] * b[1] - a[1] * b[3],
                 a[1] * b[2] - a[2] * b[1]]
    return TensorComponents(3, 1, Kind.None(), data)
end

"""
    tr(t::TensorComponents)

Return the scalar trace of a rank-2 component tensor.
"""
function tr(t::TensorComponents)
    t.kind == Kind.Alt() && return ZERO
    return component_sum(DTerm[t[i, i] for i in 1:t.dims])
end

"""
    diag(t::TensorComponents)

Extract the diagonal of a rank-2 tensor `t` as a vector.
"""
function diag(t::TensorComponents)
    return TensorComponents(t.dims, 1, Kind.None(), DTerm[t[i, i] for i in 1:t.dims])
end

"""
    transpose(t::TensorComponents)

Transpose a rank-2 tensor `t`, preserving its known kind.
"""
function Base.transpose(t::TensorComponents)
    t.kind isa Union{Kind.Sym,Kind.Diag} && return t
    t.kind == Kind.Alt() && return -t
    return TensorComponents(I -> t[I[2], I[1]], Kind.None(), 2, t.dims)
end

# Chmy's symbolic adjoint is a transpose, without complex conjugation.
Base.adjoint(t::TensorComponents) = transpose(t)

"""
    det(t::TensorComponents)

Return a determinant of a rank-2 tensor `t`.
"""
function det(t::TensorComponents)
    d = t.dims
    1 <= d <= 3 || throw(ArgumentError("determinant is supported only in dimensions 1–3"))
    d == 1 && return t[1, 1]
    if t.kind == Kind.Diag()
        return makecall_unchecked(Fun(*), Tuple(t.data))
    elseif t.kind == Kind.Alt()
        return d == 2 ? t[1, 2]^2 : ZERO
    elseif d == 2
        return t.kind == Kind.Sym() ? t[1, 1] * t[2, 2] - t[1, 2]^2 :
                                      t[1, 1] * t[2, 2] - t[1, 2] * t[2, 1]
    end
    return t[1, 1] * (t[2, 2] * t[3, 3] - t[2, 3] * t[3, 2]) +
           t[1, 2] * (t[2, 3] * t[3, 1] - t[2, 1] * t[3, 3]) +
           t[1, 3] * (t[2, 1] * t[3, 2] - t[2, 2] * t[3, 1])
end

"""
    adj(t)

Adjugate (transposed cofactor matrix) of rank-2 tensor `t`. Only defined in dimensions 1–3.
"""
function adj(t::TensorComponents)
    d = t.dims
    1 <= d <= 3 || throw(ArgumentError("adjugate is supported only in dimensions 1–3"))
    d == 1 && return literal_components(ONE, d, 2)
    kind = d == 3 && t.kind == Kind.Alt() ? Kind.Sym() : t.kind
    return TensorComponents(kind, 2, d) do (i, j)
        if t.kind == Kind.Diag()
            comps = DTerm[t[k, k] for k in 1:d if k != i]
            length(comps) == 1 && return only(comps)
            return makecall_unchecked(Fun(*), Tuple(comps))
        elseif d == 2
            return i == j ? t[3-i, 3-j] : -t[i, j]
        end
        # The transpose is essential: adj(t)[i,j] is cofactor(t)[j,i].
        return cofactor3(t, j, i)
    end
end

"""
    inv(t::TensorComponents)

Return the inverse of rank-2 tensor `t`. Only defined in dimensions 1-3.
"""
Base.inv(t::TensorComponents) = adj(t) / det(t)

"""
    sym(t)

Return the symmetric part `(t + t') / 2` of rank-2 tensor `t`.
"""
function sym(t::TensorComponents)
    t.kind isa Union{Kind.Sym,Kind.Diag} && return t
    t.kind == Kind.Alt() && return literal_components(ZERO, t.dims, 2)
    return TensorComponents(Kind.Sym(), 2, t.dims) do (i, j)
        i == j ? t[i, i] : (1 // 2) * (t[i, j] + t[j, i])
    end
end

"""
    asym(t)

Return the antisymmetric part `(t - t') / 2` of rank-2 tensor `t`.
"""
function asym(t::TensorComponents)
    t.kind == Kind.Alt() && return t
    if t.kind isa Union{Kind.Sym,Kind.Diag}
        return literal_components(ZERO, t.dims, 2)
    end
    return TensorComponents(Kind.Alt(), 2, t.dims) do (i, j)
        (1 // 2) * (t[i, j] - t[j, i])
    end
end

"""
    gram(t)

Return the Gramian `t' ⋅ t` of rank-2 tensor `t`.
"""
function gram(t::TensorComponents)
    kind = t.kind == Kind.Diag() ? Kind.Diag() : Kind.Sym()
    return TensorComponents(kind, 2, t.dims) do (i, j)
        t.kind == Kind.Diag() && return t[i, i]^2
        component_sum(DTerm[t[k, i] * t[k, j] for k in 1:t.dims])
    end
end

"""
    cogram(t)

Return the co-Gramian `t ⋅ t'` of rank-2 tensor `t`.
"""
function cogram(t::TensorComponents)
    kind = t.kind == Kind.Diag() ? Kind.Diag() : Kind.Sym()
    return TensorComponents(kind, 2, t.dims) do (i, j)
        t.kind == Kind.Diag() && return t[i, i]^2
        component_sum(DTerm[t[i, k] * t[j, k] for k in 1:t.dims])
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

function literal_components(x::DTerm, dims, rank)
    kind = rank < 2 ? Kind.None() : Kind.Diag()
    data = fill(x, ncomponents(kind, dims, rank))
    return TensorComponents(dims, rank, kind, data)
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
        return state && data[i]==ₛdata[j]
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
                return state && x==ₛy
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

evaluate(op::Fun, args::Tuple, ::Integer) = evaluate(op.f, args)::Union{DTerm,TensorComponents}
evaluate(op::Operator, args::Tuple, ::Integer) = evaluate(op, args)
evaluate(op::Fun, args::Tuple) = evaluate(op.f, args)::Union{DTerm,TensorComponents}
evaluate(op::Operator, args::Tuple) = makecall_unchecked(op, args::Tuple{Vararg{DTerm}})

broadcast_component(t::DTerm, I) = t
broadcast_component(t::TensorComponents, I) = t[I...]

function checkdims(args::Tuple, dims::Integer)
    if any(x -> x isa TensorComponents && x.dims != dims, args)
        throw(DimensionMismatch("broadcast tensor dimensions must match"))
    end
    return
end

function broadcast_kind(args::Tuple, rank::Integer)
    # arbitrary functions preserve permutation symmetry
    sym_or_diag = rank >= 2 && all(args) do x
        !(x isa TensorComponents) || x.kind isa Union{Kind.Sym,Kind.Diag}
    end
    return sym_or_diag ? Kind.Sym() : Kind.None()
end

function evaluate(op::BroadcastedFun, args::Tuple)::Union{DTerm,TensorComponents}
    f = Fun(op)
    k = findfirst(x -> x isa TensorComponents, args)
    isnothing(k) && return makecall_unchecked(f, args::Tuple{Vararg{DTerm}})
    tensor = args[k]::TensorComponents
    check_broadcast_ranks(op, args)
    checkdims(args, tensor.dims)
    kind = broadcast_kind(args, tensor.rank)
    return TensorComponents(kind, tensor.rank, tensor.dims) do I
        scalars = map(x -> broadcast_component(x, I), args)
        makecall_unchecked(f, scalars::Tuple{Vararg{DTerm}})
    end
end

function broadcasted(f, arg::Union{DTerm,TensorComponents}, args::Union{DTerm,TensorComponents}...)
    inputs = (arg, args...)
    # k cannot be nothing, all-DTerm calls will be dispatched to a method from operators.jl
    k = findfirst(x -> x isa TensorComponents, inputs)::Int
    dims = (inputs[k]::TensorComponents).dims
    op = BroadcastedFun(f)
    lowered = map(x -> x isa DTerm ? components(x, dims) : x, inputs)
    return evaluate(op, lowered)
end
broadcasted(f, a::TensorComponents, b::Number) = broadcasted(f, a, Literal(b))
broadcasted(f, a::Number, b::TensorComponents) = broadcasted(f, Literal(a), b)
function broadcasted(::typeof(Base.literal_pow), f, t::TensorComponents, ::Val{N}) where {N}
    return broadcasted(f, t, Literal(N))
end

evaluate(f, args::Tuple) = makecall_unchecked(Fun(f), args::Tuple{Vararg{DTerm}})
for op in (:+, :-)
    @eval function evaluate(::typeof($op), args::Tuple)
        if first(args) isa DTerm
            return makecall_unchecked(Fun($op), args::Tuple{Vararg{DTerm}})
        end
        return $op(args...)
    end
end
function evaluate(::typeof(*), args::Tuple)
    # at most one factor is a tensor
    k = findfirst(Base.Fix2(isa, TensorComponents), args)
    isnothing(k) && return makecall_unchecked(Fun(*), args::Tuple{Vararg{DTerm}})
    tensor = args[k]::TensorComponents
    factors = Tuple(args[j]::DTerm for j in eachindex(args) if j != k)
    isempty(factors) && return tensor
    scalar = length(factors) == 1 ? only(factors) : makecall_unchecked(Fun(*), factors)
    return scalar * tensor
end
for op in (:/, ://, :÷)
    @eval function evaluate(::typeof($op), args::Tuple)
        a, b = args
        if a isa DTerm
            return makecall_unchecked(Fun($op), args::Tuple{Vararg{DTerm}})
        end
        # the denominator is scalar by the expression's construction contract
        return $op(a, b)
    end
end
for op in (:⋅, :⊡, :⊗, :×)
    @eval evaluate(::typeof($op), args::Tuple) = $op(args...)
end
for op in (:transpose, :adjoint, :tr, :diag, :det, :adj, :inv, :sym, :asym, :gram, :cogram)
    @eval function evaluate(::typeof($op), args::Tuple)
        arg = only(args)
        if arg isa DTerm
            return makecall_unchecked(Fun($op), args::Tuple{Vararg{DTerm}})
        end
        return $op(arg)
    end
end

promote_kind(::T, ::T) where {T<:TensorKind} = T()
promote_kind(::Kind.None, ::Kind.None) = Kind.None()

promote_kind(::TensorKind, ::Kind.None) = Kind.None()
promote_kind(::Kind.Sym, ::Kind.Diag) = Kind.Sym()
promote_kind(::Kind.Alt, ::Kind.Diag) = Kind.None()
promote_kind(::Kind.Alt, ::Kind.Sym) = Kind.None()
promote_kind(::Kind.Diag, ::Kind.Sym) = Kind.Sym()

promote_kind(a, b) = promote_kind(b, a)

function component_arithmetic(f::F, a::TensorComponents, b::TensorComponents) where {F}
    if a.kind == b.kind
        data = map(a.data, b.data) do x, y
            makecall_unchecked(Fun(f), (x, y))
        end
        return TensorComponents(a.dims, a.rank, a.kind, data)
    end
    kind = promote_kind(a.kind, b.kind)
    return TensorComponents(kind, a.rank, a.dims) do I
        makecall_unchecked(Fun(f), (a[I...], b[I...]))
    end
end

function component_sum_at!(terms, tensors, index)
    empty!(terms)
    for t in tensors
        # linear data indices are used only when all storage kinds match
        x = index isa Int ? t.data[index] : t[index...]
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
    free = a.rank - n
    left, right = I[1:free], I[(free+1):end]
    terms = DTerm[]
    sizehint!(terms, ncomponents(Kind.None(), a.dims, n))
    foreach_component(Kind.None(), a.dims, n) do K
        push!(terms, a[left..., K...] * b[K..., right...])
    end
    return component_sum(terms)
end

function contract_components(a::TensorComponents, b::TensorComponents, n)
    rank = a.rank + b.rank - 2n
    rank == 0 && return contraction_component(a, b, (), n)
    return TensorComponents(I -> contraction_component(a, b, I, n), Kind.None(), rank, a.dims)
end

function cofactor3(t, row, col)
    # Remaining rows and columns in increasing order.
    r1, r2 = row == 1 ? 2 : 1, row == 3 ? 2 : 3
    c1, c2 = col == 1 ? 2 : 1, col == 3 ? 2 : 3
    minor = t[r1, c1] * t[r2, c2] - t[r1, c2] * t[r2, c1]
    return iseven(row + col) ? minor : -minor
end
