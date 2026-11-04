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
        (ZeroTensor(x), ZeroTensor(y)) => x == y
        (IdTensor(x), IdTensor(y)) => x == y
        (Index(i), Index(j)) => i == j
        (Coordinate(i), Coordinate(j)) => i == j
        (DExpr(ah, ac), DExpr(bh, bc)) => begin
            ah == bh || return false
            length(ac) == length(bc) || return false
            for k in eachindex(ac)
                ac[k] ==ₛ bc[k] || return false
            end
            true
        end
        _ => false
    end
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
        (Index(i), Index(j)) => i < j
        (Coordinate(i), Coordinate(j)) => i < j
        (Literal(a), Literal(b)) => isless(a, b)::Bool
        (Tensor(_, nx, _, _), Tensor(_, ny, _, _)) => begin
            nx == ny || return isless(nx, ny)
            x ==ₛ y || throw(ArgumentError("tensors with the same name must have the same rank, kind, and uniformity"))
            false
        end
        (DExpr(_, _), DExpr(_, _)) => isless_expr(x, y)
        _ => false # Equal-rank zero and identity tensors.
    end
end
(<ₛ)(x, y) = x < y
(<ₛ)(x::Fun, y::Fun) = isless(nameof(x.f), nameof(y.f))
(<ₛ)(::Fun, ::Operator) = true
(<ₛ)(::Operator, ::Fun) = false
(<ₛ)(x::Operator, y::Operator) = isless(objectid(x), objectid(y))

"""
    isnegof(x, y)

Check whether canonical scalar expressions have equal bodies and opposite outer signs.
"""
function isnegof(x::DTerm, y::DTerm)::Bool
    if isliteral(x) && isliteral(y)
        return isnegof(value(x), value(y))
    elseif iscall(x) && isunaryminus(x)
        return only(arguments(x)) ==ₛ y
    elseif iscall(y) && isunaryminus(y)
        return x ==ₛ only(arguments(y))
    end
    return false
end
isnegof(x::T, y::S) where {T <: Number, S <: Number} = isequal(x, -y)::Bool

"""
    𝒪(rank)

Construct symbolic zero tensor of rank `rank`, which is the literal zero for rank zero.
"""
𝒪(rank::Integer) = iszero(rank) ? ZERO : ZeroTensor(rank)

"""
    ℐ(rank)

Construct symbolic identity tensor of rank `rank`, which is the literal one for rank zero.
"""
ℐ(rank::Integer) = iszero(rank) ? ONE : IdTensor(rank)

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

"""
    isliteral(term)

Returns `true` if `term` is a Literal.
"""
isliteral(term::DTerm) = isa_variant(term, Literal)

"""
    isindex(term)

Return `true` if `term` is a symbolic grid index.
"""
isindex(term::DTerm) = isa_variant(term, Index)

"""
    iscoordinate(term)

Return `true` if `term` is a symbolic coordinate.
"""
iscoordinate(term::DTerm) = isa_variant(term, Coordinate)

"""
    istensor(term)

Return `true` if `term` is a named tensor, zero tensor, or identity tensor.
"""
istensor(term::DTerm) = isa_variant(term, Tensor) || isa_variant(term, ZeroTensor) || isa_variant(term, IdTensor)

"""
    isexpr(term)

Return `true` if `term` is a symbolic expression.
"""
isexpr(term::DTerm) = isa_variant(term, DExpr)

"""
    iscall(term)

Return `true` if `term` is a call expression.
"""
iscall(term::DTerm) = isa_variant(term, DExpr) && isa_variant(term.head, Call)

"""
    iscomp(term)

Return `true` if `term` is a tensor component expression.
"""
iscomp(term::DTerm) = isa_variant(term, DExpr) && isa_variant(term.head, Comp)

"""
    isinds(term)

Return `true` if `term` is a grid indexing expression.
"""
isinds(term::DTerm) = isa_variant(term, DExpr) && isa_variant(term.head, Inds)

"""
    ispack(term)

Return `true` if `term` is a tensor in component form, see [`pack`](@ref).
"""
ispack(term::DTerm) = isa_variant(term, DExpr) && isa_variant(term.head, Pack)

"""
    head(term)

Return the head of an expression.
"""
function head(term::DTerm)
    return @match term begin
        DExpr(head, _) => head
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
        _ => throw(ArgumentError("only call expressions have arguments"))
    end
end

"""
    isuniform(term)

Return `true` for literals, grid indices, uniform named tensors, zero tensors, and
identity tensors. An expression is uniform if all its children are uniform.
"""
function isuniform(term::DTerm)
    return @match term begin
        Literal(_) => true
        Tensor(_, _, _, uniform) => uniform
        ZeroTensor(_) || IdTensor(_) => true
        DExpr(_, args) => all_isuniform(args)
        _ => false
    end
end

# specialize on tuple length to avoid boxing in isuniform recursive call (Julia 1.13)
all_isuniform(args::NTuple{N, DTerm}) where {N} = all(isuniform, args)

# uniform terms without calls are the same at every position, unlike uniform calls, whose lowering can depend on the
# position, e.g. overrides of uniform terms
function isconstant(term::DTerm)
    return @match term begin
        Literal(_) || ZeroTensor(_) || IdTensor(_) => true
        Tensor(_, _, _, uniform) => uniform
        DExpr(Comp(_), (arg,)) => isconstant(arg)
        _ => false
    end
end

"""
    value(term)

Return the value of a literal or the dimension of a symbolic grid index.
"""
function value(term::DTerm)
    return @match term begin
        Literal(val) || Index(val) || Coordinate(val) => val
        _ => throw(ArgumentError("only literals and grid indices can have a value"))
    end
end

"""
    isstaticzero(term)

Return `true` if `term` is a literal whose value is `isequal` to zero or a symbolic zero tensor.
"""
function isstaticzero(term::DTerm)::Bool
    return @match term begin
        Literal(val) => isequal(val, 0)::Bool
        ZeroTensor(_) => true
        _ => false
    end
end

"""
    isstaticone(term)

Return `true` if `term` is a literal whose value is `isequal` to one or a symbolic identity tensor.
"""
function isstaticone(term::DTerm)::Bool
    return @match term begin
        Literal(val) => isequal(val, 1)::Bool
        IdTensor(_) => true
        _ => false
    end
end

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
function arity(term::DTerm)
    return @match term begin
        DExpr(Call(_), args) => length(args)
        _ => throw(ArgumentError("only function expressions have arity"))
    end
end

"""
    isunary(term)

Return `true` if `term` is a call expression with one argument.
"""
function isunary(term::DTerm)
    return @match term begin
        DExpr(Call(_), args) => length(args) == 1
        _ => false
    end
end

"""
    isbinary(term)

Return `true` if `term` is a call expression with two arguments.
"""
function isbinary(term::DTerm)
    return @match term begin
        DExpr(Call(_), args) => length(args) == 2
        _ => false
    end
end

"""
    argument(term)

Return the term being indexed by a component, location, or grid indexing expression.
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
        ZeroTensor(rank) || IdTensor(rank) => rank
        DExpr(_, _, rank, _) => rank
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
    getindex(term::DTerm, positions::Union{Integer, Rational, Lower, Upper, Colon}...)

Without indices, return the scalar `term` itself. Otherwise, construct a symbolic tensor component if `term` is not a scalar, respecting symmetric, alternating,
and diagonal tensor structure. Zero and identity tensors return literal component values, and packed
tensors return the stored entry.
If `term` is a scalar, construct a sample of `term` at `positions` relative to the evaluation point: offsets, which
must be multiples of `1//2`, or the global positions `↓`, `↑` and `:`. The global positions `↓` and `↑` sample the
traces of `term` on the lower and upper walls, and `:` its value independent of the axis, e.g. `g[↓, 1//2]`. Samples
of samples with as many positions compose, e.g. `g[1//2][1//2]` is `g[1]`, and literals and uniform tensors are not
sampled.
"""
function Base.getindex(term::DTerm)
    tensorrank(term) == 0 || throw(ArgumentError("number of subscripts not matching tensor rank"))
    return term
end
function Base.getindex(term::DTerm, I::Vararg{Integer, N}) where {N}
    tensorrank(term) == 0 && return makeinds(term, I)
    return component(term, map(Int, I))
end

# the component `I` of `term`, respecting the structure of named, zero, identity, and packed tensors
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
        ZeroTensor(_) => ZERO
        IdTensor(_) => allequal(I) ? ONE : ZERO
        DExpr(Pack(_, dims, kind), data) => begin
            all(i -> 1 <= i <= dims, I) || throw(BoundsError(term, I))
            pack_entry(kind, dims, data, I)
        end
        DExpr(Call(op), args) => begin
            result = component(op, args, I)::Union{Nothing, DTerm}
            isnothing(result) ? makecomp(term, I) : result
        end
        _ => makecomp(term, I)
    end
end
Base.getindex(term::DTerm, positions::Vararg{PositionLike, N}) where {N} = makeinds(term, positions)
# samples at global positions only, which are their own binding keys
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
        Index(i) => hash(i, hash(:Index, h))
        Coordinate(i) => hash(i, hash(:Coordinate, h))
        Tensor(rank, name, kind, uniform) => hash(uniform, hash(kind, hash(name, hash(rank, hash(:Tensor, h)))))
        ZeroTensor(rank) => hash(rank, hash(:ZeroTensor, h))
        IdTensor(rank) => hash(rank, hash(:IdTensor, h))
        DExpr(_, _, _, cached) => iszero(h) ? cached : hash(cached, h)
    end
end

## Lexicographic comparison of Chmy terms

# Ranks preserve the ordering of the static symbolic core.
function termrank(term)
    return @match term begin
        Index(_) => 0
        Coordinate(_) => 1
        Tensor(_, _, _, _) => 2
        ZeroTensor(_) => 3
        IdTensor(_) => 4
        Literal(_) => 5
        DExpr(_, _) => 6
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

function isless_args(xs, ys, lt = <ₛ)
    for i in 1:min(length(xs), length(ys))
        x, y = xs[i], ys[i]
        isequal(x, y) || return lt(x, y)
    end
    return length(xs) < length(ys)
end

# samples of a field are ordered by descending offsets, so that stencils read `f[1//2] - f[-1//2]`, followed by the
# global positions
isless_position(x::Rational{Int}, y::Rational{Int}) = x > y
isless_position(::Rational{Int}, ::Global) = true
isless_position(::Global, ::Rational{Int}) = false
isless_position(x::Global, y::Global) = globalrank(x) < globalrank(y)

globalrank(::Lower) = 0
globalrank(::Upper) = 1
globalrank(::Colon) = 2
