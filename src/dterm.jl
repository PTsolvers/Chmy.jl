"""
    ==ₛ(a, b)

Test whether two Chmy terms are structurally equal.
This function **does not** compare canonical forms of `a` and `b`.
For example, `x + y ==ₛ y + x` is `false`.
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
(<ₛ)(x::Location, y::Location) = x isa Point && y isa Segment

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

Construct symbolic zero tensor of rank `rank`.
"""
const 𝒪 = ZeroTensor

"""
    ℐ(rank)

Construct symbolic identity tensor of rank `rank`.
"""
const ℐ = IdTensor

function DTermImpl.DExpr(head::Head, args::NTuple{N, DTerm}) where {N}
    rank = @match head begin
        Call(op) => tensorrank(op, args)::Int
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
    islocs(term)

Return `true` if `term` is an expression evaluated at staggered grid locations.
"""
islocs(term::DTerm) = isa_variant(term, DExpr) && isa_variant(term.head, Locs)

"""
    isinds(term)

Return `true` if `term` is a grid indexing expression.
"""
isinds(term::DTerm) = isa_variant(term, DExpr) && isa_variant(term.head, Inds)

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
        Literal(_) || Index(_) => true
        Tensor(_, _, _, uniform) => uniform
        ZeroTensor(_) || IdTensor(_) => true
        DExpr(_, args) => all_isuniform(args)
        _ => false
    end
end

# specialize on tuple length to avoid boxing in isuniform recursive call (Julia 1.13)
all_isuniform(args::NTuple{N, DTerm}) where {N} = all(isuniform, args)

"""
    value(term)

Return the value of a literal or the dimension of a symbolic grid index.
"""
function value(term::DTerm)
    return @match term begin
        Literal(val) || Index(val) => val
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
        DExpr(Locs(_), (arg,)) => arg
        DExpr(Inds(), (arg, _...)) => arg
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
    locations(term)

Return the location tuple of a location expression.
"""
function locations(term::DTerm)
    return @match term begin
        DExpr(Locs(locs), _) => locs
        _ => throw(ArgumentError("only location expression can have locations"))
    end
end

"""
    indices(term)

Return the grid index tuple of a grid indexing expression.
"""
function indices(term::DTerm)
    return @match term begin
        DExpr(Inds(), (_, inds...)) => inds
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

Return the symmetry kind of a named tensor.
"""
function tensorkind(term::DTerm)
    return @match term begin
        Tensor(_, _, kind, _) => kind
        _ => throw(ArgumentError("only named tensors have a kind"))
    end
end

"""
    getindex(term::DTerm, I::Integer...)

Construct a symbolic tensor component, respecting symmetric, alternating, and diagonal
tensor structure. Literals and grid indices are returned unchanged. Zero and identity
tensors return literal component values.
"""
function Base.getindex(term::DTerm, I::Vararg{Integer, N}) where {N}
    return @match term begin
        Literal(_) => term
        Index(_) => term
        Tensor(_, _, kind, _) => begin
            if kind == Kind.Sym()
                makecomp(term, sort(I))
            elseif kind == Kind.Alt()
                if !allunique(I)
                    Literal(0)
                else
                    expr = makecomp(term, sort(I))
                    iseven(inversion_count(I)) ? expr : -expr
                end
            elseif kind == Kind.Diag()
                allequal(I) ? makecomp(term, I) : Literal(0)
            elseif kind == Kind.None()
                makecomp(term, I)
            else
                error("unreachable")
            end
        end
        ZeroTensor(rank) => begin
            rank == N || throw(ArgumentError("number of subscripts not matching tensor rank"))
            Literal(0)
        end
        IdTensor(rank) => begin
            rank == N || throw(ArgumentError("number of subscripts not matching tensor rank"))
            allequal(I) ? Literal(1) : Literal(0)
        end
        _ => makecomp(term, I)
    end
end

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
        Tensor(_, _, _, _) => 1
        ZeroTensor(_) => 2
        IdTensor(_) => 3
        Literal(_) => 4
        DExpr(_, _) => 5
    end
end

function headrank(head)
    return @match head begin
        Comp(_) => 0
        Locs(_) => 1
        Inds() => 2
        Call(_) => 3
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
        (Comp(ix), Comp(iy)) || (Locs(ix), Locs(iy)) => begin
            ax[1] ==ₛ ay[1] || return ax[1] <ₛ ay[1]
            isless_args(ix, iy)
        end
        _ => isless_args(ax, ay) # Grid indexing stores argument and indices together.
    end
end

function isless_args(xs, ys)
    for i in 1:min(length(xs), length(ys))
        x, y = xs[i], ys[i]
        isequal(x, y) || return x <ₛ y
    end
    return length(xs) < length(ys)
end
