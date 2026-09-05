# tensor components
function Base.getindex(term::DTerm, I::Vararg{Integer,N}) where {N}
    @match term begin
        Literal(_) => return term
        Index(_) => return term
        Tensor(_, _, kind, _) => begin
            if kind == Kind.Sym()
                return makecomp(term, sort(I))
            elseif kind == Kind.Alt()
                allunique(I) || return ZeroTensor(0)
                expr = makecomp(term, sort(I))
                return iseven(inversion_count(I)) ? expr : -expr
            elseif kind == Kind.Diag()
                return allequal(I) ? makecomp(term, I) : ZeroTensor(0)
            elseif kind == Kind.None()
                return makecomp(term, I)
            else
                error("unreachable")
            end
        end
        ZeroTensor(rank) => begin
            rank == N || throw(ArgumentError("number of components not matching the tensor rank"))
            return ZeroTensor(0)
        end
        IdTensor(rank) => begin
            rank == N || throw(ArgumentError("number of components not matching the tensor rank"))
            return allequal(I) ? IdTensor(0) : ZeroTensor(0)
        end
        _ => return makecomp(term, I)
    end
end

# TODO:
# 1. Construct tensors from expressions with automatic kind detection
# 2. Promote tensors to the widest kind for addition
# 3. Implement tensor operations ⊡, ⋅, ×, ⊗, transpose, det, adj, diag, sym, asym, gram, cogram
# 4. Implement broadcasting for tensor components
# 5. Construct tensors from tensor expressions

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

struct TensorComponents
    dims::Int
    rank::Int
    kind::TensorKind
    data::Vector{DTerm}
end

tensorrank(t::TensorComponents) = t.rank
tensorkind(t::TensorComponents) = t.kind

Base.ndims(t::TensorComponents) = t.dims

ncomponents(t::TensorComponents) = length(t.data)

components(t::TensorComponents) = t.data

isuniform(t::TensorComponents) = all(isuniform, t.data)

foreach_component(f, kind, dims, rank) = visit_components((I, _) -> (f(I); true), kind, dims, rank)

visit_components(f, ::Kind.Diag, dims, rank) = diagonal(f, dims, rank)
visit_components(f, ::Kind.Sym, dims, rank) = nondecreasing(f, dims, rank)
visit_components(f, ::Kind.Alt, dims, rank) = increasing(f, dims, rank)
visit_components(f, ::Kind.None, dims, rank) = cartesian(f, dims, rank)

function TensorComponents(t::DTerm, dims::Integer)
    @match t begin
        Tensor(rank, _, kind, _) => begin
            data = DTerm[]
            foreach_component(kind, dims, rank) do idx
                push!(data, makecomp_unchecked(t, idx))
            end
            TensorComponents(dims, rank, kind, data)
        end
        DExpr(head, args) => begin
            @match head begin
                Call(op) => evaluate(op, dims, args)
                _ => TensorComponents(dims, 0, Kind.None(), [t])
            end
        end
        _ => TensorComponents(dims, 0, Kind.None(), [t])
    end
end

evaluate(op::Fun, dims::Integer, args::NTuple{N,DTerm}) where {N} = op.f(map(x -> TensorComponents(x, dims), args)...)::TensorComponents

function Base.getindex(t::TensorComponents, I::Vararg{Integer,N}) where {N}
    if t.kind == Kind.Sym()
        I = sort(I)
        idx = linear_index(t.kind, t.dims, I)
        return t.data[idx]
    elseif t.kind == Kind.Alt()
        allunique(I) || return ZeroTensor(0)
        idx = linear_index(t.kind, t.dims, sort(I))
        v = t.data[idx]
        return iseven(inversion_count(I)) ? v : -v
    elseif t.kind == Kind.Diag()
        return allequal(I) ? t.data[I[1]] : ZeroTensor(0)
    elseif t.kind == Kind.None()
        return t.data[linear_index(t.kind, t.dims, I)]
    else
        error("unreachable")
    end
end

function promote_kind(a::TensorComponents, b::TensorComponents)
    kind = promote_kind_rule(a.kind, b.kind)
    return to_kind(kind, a), to_kind(kind, b)
end

function to_kind(kind::TensorKind, t::TensorComponents)
    kind == t.kind && return t
    data = DTerm[]
    foreach_component(kind, t.dims, t.rank) do I
        push!(data, t[I])
    end
    return TensorComponents(t.dims, t.rank, kind, data)
end

promote_kind_rule(::T, ::T) where {T<:TensorKind} = T()

promote_kind_rule(::TensorKind, ::Kind.None) = Kind.None()
promote_kind_rule(::Kind.Sym, ::Kind.Diag)   = Kind.Sym()
promote_kind_rule(::Kind.Alt, ::Kind.Diag)   = Kind.None()
promote_kind_rule(::Kind.Alt, ::Kind.Sym)    = Kind.None()
promote_kind_rule(::Kind.Diag, ::Kind.Sym)   = Kind.Sym()

promote_kind_rule(a, b) = promote_kind_rule(b, a)

promote_zero_rule(kind::TensorKind) = kind

promote_one_rule(::Kind.None) = Kind.None()
promote_one_rule(::Kind.Sym) = Kind.Sym()
promote_one_rule(::Kind.Alt) = Kind.None()
promote_one_rule(::Kind.Diag) = Kind.Diag()

Base.:+(a::TensorComponents) = a
function Base.:+(a::TensorComponents, b::TensorComponents)
    a.rank == b.rank || throw(ArgumentError("tensors in a sum must have the same rank"))
    a.dims == b.dims || throw(ArgumentError("tensors must have same number of dimensions"))
    a, b = promote_kind(a, b)
    data = map(+, a.data, b.data)
    return TensorComponents(a.dims, a.rank, a.kind, data)
end

function Base.:-(a::TensorComponents, b::TensorComponents)
    tensorrank(a) == tensorrank(b) || throw(ArgumentError("tensors in a sum must have the same rank"))
    ndims(a) == ndims(b) || throw(ArgumentError("tensors must have same number of dimensions"))
    a, b = promote_kind(a, b)
    data = map(-, a.data, b.data)
    return TensorComponents(a.dims, a.rank, a.kind, data)
end

function Base.:*(s, t::TensorComponents)
    data = [s * x for x in t.data]
    return TensorComponents(t.dims, t.rank, t.kind, data)
end

for op in (:*, :/, ://, :÷)
    @eval function Base.$op(t::TensorComponents, s)
        data = [$op(x, s) for x in t.data]
        return TensorComponents(t.dims, t.rank, t.kind, data)
    end
end

function maketensor(kind::Kind.None, dims, rank, data)
    if issym(dims, rank, data)
        symdata = sym_data(dims, rank, data)
        return maketensor(Kind.Sym(), dims, rank, symdata)
    end
    if isalt(dims, rank, data)
        altdata = alt_data(dims, rank, data)
        return maketensor(Kind.Alt(), dims, rank, altdata)
    end
    return TensorComponents(dims, rank, kind, data)
end
function maketensor(kind::Kind.Sym, dims, rank, data)
    if isdiag(dims, rank, data)
        diagdata = diag_data(dims, rank, data)
        return maketensor(Kind.Diag(), dims, rank, diagdata)
    end
    return TensorComponents(dims, rank, kind, data)
end
function maketensor(kind::Kind.Alt, dims, rank, data)
    all(isstaticzero, data) && return ZeroTensor(rank)
    return TensorComponents(dims, rank, kind, data)
end
function maketensor(kind::Kind.Diag, dims, rank, data)
    all(isstaticzero, data) && return ZeroTensor(rank)
    all(isstaticone, data) && return IdTensor(rank)
    return TensorComponents(dims, rank, kind, data)
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
                return state && (x==ₛ-y || -x==ₛy)
            end
        end
    end
end

function isdiag(dims, rank, data)
    rank < 2 && return false
    return nondecreasing(dims, rank) do I, state
        i = linear_index(Kind.Sym(), dims, I)
        return state && (allequal(I) || isstaticzero(data[i]))
    end
end

function sym_data(dims, rank, data)
    comps = DTerm[]
    nondecreasing(dims, rank) do I, _
        i = linear_index(Kind.None(), dims, I)
        push!(comps, data[i])
        return true
    end
    return comps
end

function alt_data(dims, rank, data)
    comps = DTerm[]
    increasing(dims, rank) do I, _
        i = linear_index(Kind.None(), dims, I)
        push!(comps, data[i])
        return true
    end
    return comps
end

function diag_data(dims, rank, data)
    comps = DTerm[]
    diagonal(dims, rank) do I, _
        i = linear_index(Kind.Sym(), dims, I)
        push!(comps, data[i])
        return true
    end
    return comps
end
