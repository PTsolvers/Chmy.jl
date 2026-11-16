"""
    ==ₛ(a, b)

Test whether two Chmy terms are structurally equal. Terms are built in canonical form, so equivalent constructions are
equal, e.g. `x + y ==ₛ y + x`, but the canonical form is not a normal form: `2(x + y) ==ₛ 2x + 2y` is `false`.
"""
function (==ₛ)(a::DTerm, b::DTerm)
    hash(a) == hash(b) || return false
    return @match (a, b) begin
        (Literal(x), Literal(y)) => isequal(x, y)::Bool
        (Tensor(ar, an, ak, au), Tensor(br, bn, bk, bu)) =>
            ar == br &&
            an == bn &&
            ak == bk &&
            au == bu
        (Coordinate(i), Coordinate(j)) => i == j
        (DExpr(ah, ac), DExpr(bh, bc)) => ah == bh && isequal_args(ac, bc)
        (AddMul(af, ac, ad), AddMul(bf, bc, bd)) =>
            af === bf &&
            isequal(ac, bc)::Bool &&
            isequal_args(ad.keys, bd.keys) &&
            isequal_args(ad.vals, bd.vals)
        _ => false
    end
end

function isequal_args(xs, ys)
    length(xs) == length(ys) || return false
    for k in eachindex(xs, ys)
        xs[k] ==ₛ ys[k] || return false
    end
    return true
end

"""
    !=ₛ(a, b)

Return `true` if `a` and `b` are not structurally equal. See [`==ₛ`](@ref).
"""
(!=ₛ)(a, b) = !(a ==ₛ b)

"""
    isequal(a::DTerm, b::DTerm)

Same as [`==ₛ`](@ref).
"""
Base.isequal(a::DTerm, b::DTerm) = a ==ₛ b

"""
    x <ₛ y

Compare dynamic symbolic terms lexicographically for canonical factor ordering.
"""
function (<ₛ)(x::DTerm, y::DTerm)
    tx, ty = tensorrank(x), tensorrank(y)
    tx == ty || return tx < ty

    rx, ry = termrank(x), termrank(y)
    rx == ry || return rx < ry

    return @match (x, y) begin
        (Coordinate(i), Coordinate(j)) => i < j
        (Literal(a), Literal(b)) => isless(a, b)::Bool
        (Tensor(_, nx, _, _), Tensor(_, ny, _, _)) => begin
            nx == ny || return isless(nx, ny)
            x ==ₛ y || throw(ArgumentError("tensors with the same name must have the same rank, kind, and uniformity"))
            false
        end
        (DExpr(_, _), DExpr(_, _)) => isless_expr(x, y)
        (AddMul(_, _, _), AddMul(_, _, _)) => isless_addmul(x, y)
        _ => error("unreachable")
    end
end
(<ₛ)(x, y) = x < y
(<ₛ)(x::Fun, y::Fun) = isless(nameof(x.f), nameof(y.f))
(<ₛ)(::Fun, ::Operator) = true
(<ₛ)(::Operator, ::Fun) = false
(<ₛ)(x::Operator, y::Operator) = isless(objectid(x), objectid(y))

"""
    𝒪(rank)

Construct symbolic zero tensor of rank `rank`, which is the literal zero for rank zero.
"""
𝒪(rank::Integer) = ZeroTensor(rank)()

"""
    ℐ(rank)

Construct symbolic identity tensor of rank `rank`, which is the literal one for rank zero.
"""
ℐ(rank::Integer) = IdTensor(rank)()

function DTermImpl.DExpr(head::Head, args::NTuple{N, DTerm}) where {N}
    rank = @match head begin
        Call(op) => tensorrank(op, args)::Int
        Pack(r, _, _) => r
        _ => 0
    end
    h = hash(head, hash(length(args), hash(:DExpr)))
    for k in eachindex(args)
        h = hash(args[k], h)
    end
    return DExpr(head, args, rank, h)
end

function DTermImpl.AddMul(op::Union{typeof(+), typeof(*)}, coeff::Number, dict::LittleDict, rank::Int)
    h = hash(coeff, hash(op, hash(:AddMul)))
    for k in eachindex(dict.keys)
        h = hash(dict.vals[k], hash(dict.keys[k], h))
    end
    return AddMul(op, coeff, dict, rank, h)
end

"""
    isliteral(term)

Returns `true` if `term` is a Literal.
"""
isliteral(term::DTerm) = isa_variant(term, Literal)

"""
    iscoordinate(term)

Return `true` if `term` is a symbolic coordinate.
"""
iscoordinate(term::DTerm) = isa_variant(term, Coordinate)

"""
    istensor(term)

Return `true` if `term` is a named tensor.
"""
istensor(term::DTerm) = isa_variant(term, Tensor)

"""
    isexpr(term)

Return `true` if `term` is a symbolic expression.
"""
isexpr(term::DTerm) = isa_variant(term, DExpr) || isa_variant(term, AddMul)

"""
    iscall(term)

Return `true` if `term` is a call expression.
"""
iscall(term::DTerm) = isa_variant(term, AddMul) || (isa_variant(term, DExpr) && isa_variant(term.head, Call))

"""
    iscomp(term)

Return `true` if `term` is a tensor component expression.
"""
iscomp(term::DTerm) = isa_variant(term, DExpr) && isa_variant(term.head, Comp)

"""
    isinds(term)

Return `true` if `term` is a sample, see [`getindex`](@ref).
"""
isinds(term::DTerm) = isa_variant(term, DExpr) && isa_variant(term.head, Inds)

"""
    ispack(term)

Return `true` if `term` is a tensor in component form, see [`pack`](@ref).
"""
ispack(term::DTerm) = isa_variant(term, DExpr) && isa_variant(term.head, Pack)

# sums and products are stored as `AddMul`, see `addmul_operation`
issum(term::DTerm) = isa_variant(term, AddMul) && term.op === (+)
isprod(term::DTerm) = isa_variant(term, AddMul) && term.op === (*)

"""
    head(term)

Return the head of an expression.
"""
function head(term::DTerm)
    return @match term begin
        DExpr(head, _) => head
        AddMul(_, _, _) => Call(operation(term))
        _ => throw(ArgumentError("only expressions have a head"))
    end
end

"""
    children(term)

Return the tuple of children of an expression.
"""
function children(term::DTerm)
    return @match term begin
        DExpr(_, args) => args
        AddMul(_, _, _) => arguments(term)
        _ => throw(ArgumentError("only expressions have children"))
    end
end

"""
    operation(term)

Return the operator of a call expression.
"""
function operation(term::DTerm)
    return @match term begin
        DExpr(Call(op), _) => op
        AddMul(_, _, _) => addmul_operation(term)
        _ => throw(ArgumentError("only call expressions have an operation"))
    end
end

"""
    arguments(term)

Return the argument tuple of a call expression.
"""
function arguments(term::DTerm)
    return @match term begin
        DExpr(Call(_), args) => args
        AddMul(_, _, _) => Tuple(addmul_arguments(term))
        _ => throw(ArgumentError("only call expressions have arguments"))
    end
end

"""
    isuniform(term)

Return `true` for literals, uniform tensors, and expressions with uniform children.
"""
function isuniform(term::DTerm)
    return @match term begin
        Literal(_) => true
        Tensor(_, _, _, uniform) => uniform
        DExpr(_, args) => all_isuniform(args)
        AddMul(_, _, dict) => all(isuniform, Iterators.flatten((dict.keys, dict.vals)))
        _ => false
    end
end

# specialize on tuple length to avoid boxing in isuniform recursive call (Julia 1.13)
all_isuniform(args::NTuple{N, DTerm}) where {N} = all(isuniform, args)

# uniform terms without calls are the same at every position, unlike uniform calls, whose lowering can depend on the
# position, e.g. overrides of uniform terms
function isconstant(term::DTerm)
    return @match term begin
        Literal(_) => true
        Tensor(_, _, _, uniform) => uniform
        DExpr(Comp(_), (arg,)) => isconstant(arg)
        _ => false
    end
end

"""
    value(term)

Return the value of a literal or the axis of a coordinate.
"""
function value(term::DTerm)
    return @match term begin
        Literal(val) || Coordinate(val) => val
        _ => throw(ArgumentError("only literals and coordinates can have a value"))
    end
end

"""
    isstaticzero(term)

Return `true` if `term` is a literal whose value is `isequal` to zero or a symbolic zero tensor.
"""
function isstaticzero(term::DTerm)::Bool
    return @match term begin
        Literal(val) => isequal(val, 0)::Bool
        DExpr(Call(op), args) => isstaticzero(op, args)
        _ => false
    end
end

# calls are not static zeros unless specified otherwise
isstaticzero(::Operator, args) = false
isstaticzero(::ZeroTensor, args) = true

"""
    isstaticone(term)

Return `true` if `term` is a literal whose value is `isequal` to one or a symbolic identity tensor.
"""
function isstaticone(term::DTerm)::Bool
    return @match term begin
        Literal(val) => isequal(val, 1)::Bool
        DExpr(Call(op), args) => isstaticone(op, args)
        _ => false
    end
end

# calls are not static ones unless specified otherwise
isstaticone(::Operator, args) = false
isstaticone(::IdTensor, args) = true

"""
    isinteger(term::DTerm)

Return `true` if `term` is a literal with an integer value, and `false` otherwise.
"""
function Base.isinteger(term::DTerm)
    return @match term begin
        Literal(val) => isinteger(val)
        _ => false
    end
end

"""
    isreal(term::DTerm)

Return `true` if `term` is a literal with a real value, and `false` otherwise.
"""
function Base.isreal(term::DTerm)
    return @match term begin
        Literal(val) => isreal(val)
        _ => false
    end
end

"""
    arity(term)

Return the number of arguments of a call expression.
"""
arity(term::DTerm) = length(arguments(term))

"""
    isunary(term)

Return `true` if `term` is a call expression with one argument.
"""
isunary(term::DTerm) = iscall(term) && arity(term) == 1

"""
    isbinary(term)

Return `true` if `term` is a call expression with two arguments.
"""
isbinary(term::DTerm) = iscall(term) && arity(term) == 2

"""
    argument(term)

Return the term of a component or a sample.
"""
function argument(term::DTerm)
    return @match term begin
        DExpr(Comp(_), (arg,)) => arg
        DExpr(Inds(_), (arg,)) => arg
        _ => throw(ArgumentError("only indexing expressions can have a single argument"))
    end
end

"""
    components(term)

Return the component index tuple of a component expression.
"""
function components(term::DTerm)
    return @match term begin
        DExpr(Comp(comps), _) => comps
        _ => throw(ArgumentError("only component expression can have components"))
    end
end

"""
    entries(term)

Return the tuple of stored entries of a tensor in component form.
"""
function entries(term::DTerm)
    return @match term begin
        DExpr(Pack(_, _, _), data) => data
        _ => throw(ArgumentError("only tensors in component form have entries"))
    end
end

# the number of dimensions of a tensor in component form
function packdims(term::DTerm)
    return @match term begin
        DExpr(Pack(_, dims, _), _) => dims
        _ => throw(ArgumentError("only tensors in component form have a number of dimensions"))
    end
end

"""
    indices(term)

Return the position tuple of a sample.
"""
function indices(term::DTerm)
    return @match term begin
        DExpr(Inds(positions), _) => positions
        _ => throw(ArgumentError("only indexing expression can have indices"))
    end
end

"""
    tensorrank(term)

Return the tensor rank of `term`. Scalars have rank zero.
"""
function tensorrank(term::DTerm)::Int
    return @match term begin
        Tensor(rank, _, _, _) => rank
        DExpr(_, _, rank, _) => rank
        AddMul(_, _, _, rank, _) => rank
        _ => 0
    end
end

"""
    tensorname(term)

Return the symbol naming a named tensor.
"""
function tensorname(term::DTerm)
    return @match term begin
        Tensor(_, name, _, _) => name
        _ => throw(ArgumentError("only named tensors have a name"))
    end
end

"""
    tensorkind(term)

Return the symmetry kind of a named tensor or a packed tensor.
"""
function tensorkind(term::DTerm)
    return @match term begin
        Tensor(_, _, kind, _) => kind
        DExpr(Pack(_, _, kind), _) => kind
        _ => throw(ArgumentError("only named tensors and packed tensors have a kind"))
    end
end

"""
    getindex(term::DTerm)
    getindex(term::DTerm, I::Integer...)
    getindex(term::DTerm, positions::Union{Coord, Trace, Tuple, Lower, Upper, Colon}...)

Without indices, return the scalar `term` itself. Otherwise, construct a symbolic tensor component if `term` is not a scalar, respecting symmetric, alternating,
and diagonal tensor structure. Zero and identity tensors return literal component values, and packed
tensors return the stored entry.
If `term` is a scalar, sample `term` at the absolute `positions`, one per axis: coordinates `(frame, offset)`, see
[`Coord`](@ref), or the traces `↓`, `↑` and `:`, see [`Trace`](@ref), e.g. `g[↓, (:, 1//2)]`. The sample is lowered at
once: `term` is expanded in component form in as many dimensions as there are positions, see [`components`](@ref), and
its calls with their [`lower_rule`](@ref), into samples of fields, e.g. `divg(grad(u))[(:, 1//2), (:, 1//2)]` is a
stencil of samples `u[(:, -1//2), (:, 1//2)]`. Literals and uniform tensors are not sampled, and a coordinate is the
coordinate of the origin of the frame plus the offset, e.g. `ξ¹[(:, 0), :] + 1//2`, or the offset itself in the frame
of the lower wall. Samples cannot be sampled again, also inside expressions. Traces along some axes only, read at the
evaluation point along the others, are taken with [`Project`](@ref).
"""
function Base.getindex(term::DTerm)
    tensorrank(term) == 0 || throw(ArgumentError("number of subscripts not matching tensor rank"))
    return term
end
Base.getindex(term::DTerm, I::Vararg{Integer, N}) where {N} = component(term, map(Int, I))

# the component `I` of `term`, respecting the structure of named and packed tensors
function component(term::DTerm, I::NTuple{N, Int}) where {N}
    tensorrank(term) == N || throw(ArgumentError("number of subscripts not matching tensor rank"))
    N == 0 && return term
    return @match term begin
        Tensor(_, _, kind, _) => begin
            if kind == Kind.Sym()
                makecomp(term, sort(I))
            elseif kind == Kind.Alt()
                if !allunique(I)
                    ZERO
                else
                    expr = makecomp(term, sort(I))
                    iseven(inversion_count(I)) ? expr : -expr
                end
            elseif kind == Kind.Diag()
                allequal(I) ? makecomp(term, I) : ZERO
            elseif kind == Kind.None()
                makecomp(term, I)
            else
                error("unreachable")
            end
        end
        DExpr(Pack(_, dims, kind), data) => begin
            all(i -> 1 <= i <= dims, I) || throw(BoundsError(term, I))
            pack_entry(kind, dims, data, I)
        end
        DExpr(Call(op), args) => begin
            result = component(op, args, I)::Union{Nothing, DTerm}
            isnothing(result) ? makecomp(term, I) : result
        end
        # the component of a sum or a product reads the components of its tensor keys
        AddMul(_, _, _) => mapentries(x -> componentof(x, I), term)
        _ => makecomp(term, I)
    end
end
Base.getindex(term::DTerm, positions::Vararg{PositionLike, N}) where {N} = makeinds(term, positions)
# samples at traces only, which would otherwise be ambiguous with the binding keys `term[locations...]`
Base.getindex(term::DTerm, p::Global, ps::Vararg{Global, N}) where {N} = makeinds(term, (p, ps...))

# hashing DTerms
# DExpr caches the hash, other variants need to compute explicitly

"""
    hash(term::DTerm, h::UInt)

Compute a structural hash of `term` with seed `h`, consistent with [`==ₛ`](@ref).
"""
function Base.hash(term::DTerm, h::UInt)::UInt
    return @match term begin
        Literal(x) => begin
            h = hash(:Literal, h)
            if x isa Union{Int, Rational{Int}, Float64}
                return hash(x, h)
            end
            return hash(x, h)::UInt
        end
        Coordinate(i) => hash(i, hash(:Coordinate, h))
        Tensor(rank, name, kind, uniform) => hash(uniform, hash(kind, hash(name, hash(rank, hash(:Tensor, h)))))
        DExpr(_, _, _, cached) => iszero(h) ? cached : hash(cached, h)
        AddMul(_, _, _, _, cached) => iszero(h) ? cached : hash(cached, h)
    end
end

## Lexicographic comparison of Chmy terms

# Ranks preserve the ordering of the static symbolic core.
function termrank(term)
    return @match term begin
        Coordinate(_) => 0
        Tensor(_, _, _, _) => 1
        Literal(_) => 2
        DExpr(_, _) => 3
        AddMul(_, _, _) => 4
    end
end

function headrank(head)
    return @match head begin
        Comp(_) => 0
        Inds(_) => 1
        Call(_) => 2
        Pack(_, _, _) => 3
    end
end

function isless_expr(x, y)::Bool
    hx, hy = head(x), head(y)
    ax, ay = children(x), children(y)
    rx, ry = headrank(hx), headrank(hy)
    rx == ry || return rx < ry

    return @match (hx, hy) begin
        (Call(opx), Call(opy)) => begin
            opx == opy || return (opx <ₛ opy)::Bool
            isless_args(ax, ay)
        end
        (Comp(ix), Comp(iy)) => begin
            ax[1] ==ₛ ay[1] || return ax[1] <ₛ ay[1]
            isless_args(ix, iy)
        end
        (Inds(ix), Inds(iy)) => begin
            ax[1] ==ₛ ay[1] || return ax[1] <ₛ ay[1]
            isless_args(ix, iy, isless_position)
        end
        (Pack(rx, dx, _), Pack(ry, dy, _)) => begin
            (rx, dx) == (ry, dy) || return isless((rx, dx), (ry, dy))
            isless_args(ax, ay)
        end
        _ => error("unreachable")
    end
end

# sums before products, then by their keys, values, and coefficients
function isless_addmul(x, y)::Bool
    x.op === y.op || return x.op === (+)
    dx, dy = x.dict, y.dict
    isequal_args(dx.keys, dy.keys) || return isless_args(dx.keys, dy.keys)
    isequal_args(dx.vals, dy.vals) || return isless_args(dx.vals, dy.vals)
    return isless(x.coeff, y.coeff)::Bool
end

function isless_args(xs, ys, lt = <ₛ)
    for i in 1:min(length(xs), length(ys))
        x, y = xs[i], ys[i]
        isequal(x, y) || return lt(x, y)
    end
    return length(xs) < length(ys)
end

# samples of a field are ordered by the frames of their coordinates, then by descending offsets, so that stencils read
# `f[(:, 1//2)] - f[(:, -1//2)]`, followed by the traces
function isless_position(x::Coord, y::Coord)
    x.frame === y.frame || return globalrank(x.frame) < globalrank(y.frame)
    return x.offset > y.offset
end
isless_position(::Coord, ::Trace) = true
isless_position(::Trace, ::Coord) = false
isless_position(x::Trace, y::Trace) = globalrank(x.element) < globalrank(y.element)

globalrank(::Lower) = 0
globalrank(::Upper) = 1
globalrank(::Colon) = 2

# sums and products are printed and compiled as raw calls of `+`, `-`, `*`, `/`, `^` and `inv`: the terms of a sum in
# their order with the constant last, where negative terms are subtracted, and the bases of a product with negative
# powers in the denominator
function callform(term::DTerm)
    return @match term begin
        AddMul(op, coeff, dict) && if op === (+) end => sum_callform(coeff, dict)
        AddMul(_, coeff, dict) => signed(isnegative(coeff), monomial_callform(magnitude(coeff), dict))
    end
end

rawcall(f, args) = makecall_unchecked(Fun(f), Tuple(args))
chain(f, args) = length(args) == 1 ? only(args) : rawcall(f, args)
signed(negative, x) = negative ? rawcall(-, (x,)) : x

# the magnitude of a real coefficient, other numbers are kept
magnitude(c) = isnegative(c) ? -c : c

function sum_callform(coeff, dict)
    parts = Tuple{Bool, DTerm}[(isnegative(value(v)), scaled(magnitude(value(v)), k)) for (k, v) in dict]
    iszero(coeff) || push!(parts, (isnegative(coeff), canonical_literal(magnitude(coeff))))
    # runs of positive terms are added, and negative terms are subtracted
    (negative, x), rest = Iterators.peel(parts)
    run = DTerm[signed(negative, x)]
    for (negative, x) in rest
        run = negative ? DTerm[rawcall(-, (chain(+, run), x))] : push!(run, x)
    end
    return chain(+, run)
end

function monomial_callform(c, dict)
    num, den = DTerm[], DTerm[]
    for (b, p) in dict
        negative = isliteral(p) && isnegative(value(p))
        q = negative ? canonical_literal(-value(p)) : p
        push!(negative ? den : num, isstaticone(q) ? b : rawcall(^, (b, q)))
    end
    isone(c) || pushfirst!(num, canonical_literal(c))
    isempty(den) && return chain(*, num)
    return isempty(num) ? rawcall(inv, (chain(*, den),)) : rawcall(/, (chain(*, num), chain(*, den)))
end
