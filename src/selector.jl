# boundary operators carry no data, the data are the arguments of their calls, e.g. `Dirichlet()(g)`,
# and the selected term is prepended to them when the operator is applied
abstract type BoundaryOperator <: Operator end

# the data of boundary operators may be tensors
checkranks(::BoundaryOperator, ::NTuple{N, DTerm}) where {N} = nothing

struct Next <: BoundaryOperator end
struct Add <: BoundaryOperator end
struct Scale <: BoundaryOperator end
struct Dirichlet <: BoundaryOperator end
struct Neumann <: BoundaryOperator end

# an operator without data can be given without a call
asrule(r::DTerm) = r
asrule(op::BoundaryOperator) = op()

"""
    Normal()

Matches the tensor component along the outward normal of a wall in a tensor slice of a boundary clause.
"""
struct Normal end

"""
    Tangential()

Matches every tensor component tangential to a wall in a tensor slice of a boundary clause.
"""
struct Tangential end

"""
    AnyNormal()

The outward unit normal of the wall, a vector that is only known when a boundary clause is resolved at a wall, so it
can only appear in the data of boundary rules. See also `𝐧`.
"""
struct AnyNormal <: Operator end

"""
    AnyTangent(k = 1)

The unit tangent of the wall along the component matched by the `k`-th `Tangential()` slot of a tensor slice, a vector
that is only known when a boundary clause is resolved, so it can only appear in the data of boundary rules. See also `𝐭`.
"""
struct AnyTangent <: Operator
    slot::Int
end
AnyTangent() = AnyTangent(1)

tensorrank(::Union{AnyNormal, AnyTangent}, args) = 1

const 𝐧 = AnyNormal()()
const 𝐭 = AnyTangent()()

printop(io, ::AnyNormal, args) = print(io, '𝐧')
function printop(io, t::AnyTangent, args)
    print(io, '𝐭')
    t.slot == 1 || print_subscript(io, t.slot)
    return
end

unitvector(a, s, dims) = pack([i == a ? s : 0 for i in 1:dims])

# the vectors of the wall with the signed outward axes `signed_axes` in the global frame, where `tangents` are the
# components of the tangential slots, other operators are kept
wallvector(op::Operator, signed_axes, tangents, dims) = nothing
function wallvector(::AnyNormal, signed_axes, tangents, dims)
    length(signed_axes) == 1 ||
        throw(ArgumentError(lazy"the normal is only defined on walls, got a face of codimension $(length(signed_axes))"))
    a = only(signed_axes)
    return unitvector(abs(a), sign(a), dims)
end
function wallvector(t::AnyTangent, signed_axes, tangents, dims)
    t.slot <= length(tangents) ||
        throw(ArgumentError(lazy"the tangent $(t.slot) requires as many tangential slots, got $(length(tangents))"))
    # tangential axes are never mirrored, so the tangent is the same in the global and the local frame
    return unitvector(tangents[t.slot], 1, dims)
end

# replace the normal and tangent vectors in the data `g` with their components at the resolved wall
function wallvectors(g::DTerm, signed_axes, tangents, dims)
    return postwalk(t -> iscall(t) ? wallvector(operation(t), signed_axes, tangents, dims) : nothing, g)
end

# a tensor slice holds one slot per tensor index: `:` matches every component, `i` the i-th component, and `Normal()`
# and `Tangential()` the components normal and tangential to a wall
struct BoundaryClause
    pattern::FacePattern
    region::GridRegion
    slice::Tuple
end

checkslot(::Colon, d, n) = nothing
function checkslot(i::Int, d, n)
    1 <= i <= n || throw(ArgumentError(lazy"component $i of a tensor slice exceeds the number of dimensions $n"))
    return
end
function checkslot(::Union{Normal, Tangential}, d, n)
    d == 1 || throw(ArgumentError(lazy"normal and tangential components are only defined on walls, got a face of codimension $d"))
    return
end

# the near axes `axes` of a face are a single normal axis for the normal and tangential slots, see `checkslot`
matchslot(::Colon, c, axes) = true
matchslot(i::Int, c, axes) = c == i
matchslot(::Normal, c, axes) = c == only(axes)
matchslot(::Tangential, c, axes) = c != only(axes)

matchslice(slice, comps, axes)::Bool = all(map((s, c) -> matchslot(s, c, axes), slice, comps))

# the free slots of a slice index the tensor data of the rule
isfree(::Union{Colon, Tangential}) = true
isfree(::Union{Int, Normal}) = false

istangential(::Tangential) = true
istangential(::Union{Colon, Int, Normal}) = false

freecomps(slice, comps) = Tuple(c for (s, c) in zip(slice, comps) if isfree(s))
tangentcomps(slice, comps) = Tuple(c for (s, c) in zip(slice, comps) if istangential(s))

# rules are written in the frame of the outward normal, where a component changes sign with every index along a
# mirrored axis
orient(x, comps, signed_axes) = isodd(count(c -> -c ∈ signed_axes, comps)) ? -x : x

# scalar data apply to every component of the slice, while tensor data are given in the global frame and their
# components are taken in the frame of the outward normal
function slicedata(g::DTerm, slice, comps, signed_axes)
    tensorrank(g) == 0 && return g
    free = freecomps(slice, comps)
    return orient(g[free...], free, signed_axes)
end

struct Selector <: Operator
    clauses::Tuple{Vararg{BoundaryClause}}
end

struct SiteSelector{N} <: Operator
    selector::Selector
    comps::Tuple{Vararg{Int}}
    offsets::NTuple{N, Rational{Int}}
    faces::Tuple
end

function candidates(pos::Block{N}, comps, clauses::Tuple{Vararg{BoundaryClause}}) where {N}
    # precompute codimensions of rules for known N once
    # this step also throws for malformed rules
    codims = map(clauses) do c
        d = codim(c.pattern, N)
        foreach(s -> checkslot(s, d, N), c.slice)
        return d
    end
    # find all near axes of the position
    near = findall(x -> x isa Near, pos)
    # each face stores its near axes, and a first matching rule
    faces = Pair{Vector{Int}, Int}[]
    # bit k of the mask selects the k-th near axis
    for mask in 1:(2^length(near) - 1)
        axes = [near[k] for k in eachindex(near) if !iszero(mask & 1 << (k - 1))]
        # make a Face from the near axes
        face = Face(ntuple(a -> a ∈ axes ? pos[a].side : Span(), Val(N)))
        # find the index of the first rule that fully matches the axes and the component
        i = findfirst(eachindex(clauses)) do k
            c = clauses[k]
            codims[k] == length(axes) &&
                matches(c.pattern, face) &&
                all(pos[a].ξ ∈ c.region for a in axes) &&
                matchslice(c.slice, comps, axes)
        end
        isnothing(i) || push!(faces, axes => i)
    end
    # higher codimension first, then ghost before on-boundary before near-boundary axes, then the declaration
    # order of the clause, then the lower axes on a tie
    priority((axes, i)) = (-length(axes), sort!([severity(pos[a].ξ) for a in axes]), i, axes)
    return sort!(faces; by = priority)
end

# the part of a clause after the region is a rule, or a tensor slice and a rule, the slice of scalars is `()`
splitrule(r::Pair) = (asslice(first(r)), asrule(last(r)))
splitrule(r) = ((), asrule(r))

function splitclause((pattern, (region, r)))
    slice, rule = splitrule(r)
    return BoundaryClause(pattern, region, slice) => rule
end

asslice(s::Tuple) = s
asslice(s) = (s,)

# the selector call holds the selected term followed by the boundary operator calls of the clauses
function select(term::DTerm, rules::Pair...)
    parts = map(splitclause, rules)
    return Selector(map(first, parts))(term, map(last, parts)...)
end

tensorrank(::Selector, args::Tuple{Vararg{DTerm}}) = tensorrank(first(args))
function checkranks(sel::Selector, args::Tuple{Vararg{DTerm}})
    rank = tensorrank(first(args))
    foreach(sel.clauses, Base.tail(args)) do c, rule
        length(c.slice) == rank ||
            throw(ArgumentError(lazy"tensor slice $(c.slice) does not match the rank $rank of the selected term"))
        nfree = count(isfree, c.slice)
        all(g -> tensorrank(g) ∈ (0, nfree), arguments(rule)) ||
            throw(ArgumentError(lazy"the data of a rule with tensor slice $(c.slice) must be scalars or tensors of rank $nfree"))
    end
    return
end

tensorrank(site::SiteSelector, args::Tuple{Vararg{DTerm}}) = tensorrank(site.selector, args)
# the arguments of a site selector are the ones of its selector, checked when the selector call was built
checkranks(::SiteSelector, ::Tuple{Vararg{DTerm}}) = nothing

# a selector prints as the selected term, highlighted in red
printop(io, ::Union{Selector, SiteSelector}, args) = printstyled(io, first(args); color = :red)
printprec(::Union{Selector, SiteSelector}, args) = printprec(first(args))

function lower_rule(sel::Selector, args, comps, offs, block)
    pos = map(position, block, offs)
    # the number of faces and their codimensions are only known at runtime
    faces = Tuple(Tuple(axes) => i for (axes, i) in candidates(pos, comps, sel.clauses))
    site = SiteSelector(sel, comps, offs, faces)
    return lower_rule(site, args, comps, offs, block)
end

function lower_rule(site::SiteSelector, args, comps, offs, block)
    if comps != site.comps || offs != site.offsets
        return lower_rule(site.selector, args, comps, offs, block)
    end
    isempty(site.faces) && return here(first(args)[comps...], offs)
    axes, i = first(site.faces)
    rest = SiteSelector(
        site.selector,
        site.comps,
        site.offsets,
        Base.tail(site.faces)
    )(args...)
    pos = map(position, block, offs)
    # rules are written in the frame of the outward normal, i.e. for the upper wall
    signed_axes = ntuple(k -> outward(pos[axes[k]].side) * axes[k], length(axes))
    rule = args[1 + i]
    slice = site.selector.clauses[i].slice
    tangents = tangentcomps(slice, comps)
    data = map(arguments(rule)) do g
        slicedata(wallvectors(g, signed_axes, tangents, length(offs)), slice, comps, signed_axes)
    end
    lifted = Lifted(operation(rule), signed_axes)
    flip(x) = orient(x, comps, signed_axes)
    return flip(lower_rule(lifted, (flip(rest[comps...]), data...), (), offs, block))
end

lower_rule(::Next, (u,)::Tuple{DTerm}, offs, block) = here(u, offs)
lower_rule(::Add, (u, e)::Tuple{DTerm, DTerm}, offs, block) = here(u, offs) + here(e, offs)
lower_rule(::Scale, (u, w)::Tuple{DTerm, DTerm}, offs, block) = here(w, offs) * here(u, offs)

ξ(offs, block) = position(first(block), first(offs)).ξ

# the value of `x` on the wall
trace(x::DTerm, offs, block) = x[-ξ(offs, block)]

function lower_rule(::Dirichlet, (u, g)::Tuple{DTerm, DTerm}, offs::NTuple{1, Rational{Int}}, block::Block{1})
    ξb = ξ(offs, block)
    if ξb == 0
        return trace(g, offs, block)
    end
    if ξb > 0
        # the mirror sample inside the domain
        return 2trace(g, offs, block) - u[-2ξb]
    end
    throw(ArgumentError(lazy"Dirichlet rule can only be declared with Ghost() or OnBoundary() regions with ξ ≥ 0, got ξ = $ξb"))
end

# `g` is the outward normal derivative
function lower_rule(::Neumann, (u, g)::Tuple{DTerm, DTerm}, offs::NTuple{1, Rational{Int}}, block::Block{1})
    ξb = ξ(offs, block)
    if ξb > 0
        return u[-2ξb] + 2ξb * trace(g, offs, block)
    end
    throw(ArgumentError(lazy"Neumann rule can only be declared with Ghost() region with ξ > 0, got ξ = $ξb"))
end
