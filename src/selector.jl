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

struct BoundaryClause
    pattern::FacePattern
    region::GridRegion
end

struct Selector <: Operator
    clauses::Tuple{Vararg{BoundaryClause}}
end

struct SiteSelector{N} <: Operator
    selector::Selector
    offsets::NTuple{N, Rational{Int}}
    faces::Tuple
end

function candidates(pos::Block{N}, clauses::Tuple{Vararg{BoundaryClause}}) where {N}
    # precompute codimensions of rules for known N once
    # this step also throws for malformed rules
    codims = map(c -> codim(c.pattern, N), clauses)
    # find all near axes of the position
    near = findall(x -> x isa Near, pos)
    # each face stores its near axes, and a first matching rule
    faces = Pair{Vector{Int}, Int}[]
    # bit k of the mask selects the k-th near axis
    for mask in 1:(2^length(near) - 1)
        axes = [near[k] for k in eachindex(near) if !iszero(mask & 1 << (k - 1))]
        # make a Face from the near axes
        face = Face(ntuple(a -> a ∈ axes ? pos[a].side : Span(), Val(N)))
        # find the index of the first rule that fully matches the axes
        i = findfirst(eachindex(clauses)) do k
            c = clauses[k]
            codims[k] == length(axes) && matches(c.pattern, face) && all(pos[a].ξ ∈ c.region for a in axes)
        end
        isnothing(i) || push!(faces, axes => i)
    end
    # higher codimension first, then ghost before on-boundary before near-boundary axes, then the declaration
    # order of the clause, then the lower axes on a tie
    priority((axes, i)) = (-length(axes), sort!([severity(pos[a].ξ) for a in axes]), i, axes)
    return sort!(faces; by = priority)
end

# the selector call holds the selected term followed by the boundary operator calls of the clauses
function select(term::DTerm, rules::Pair...)
    clauses = map(p -> BoundaryClause(p[1], first(p[2])), rules)
    return Selector(clauses)(term, map(p -> asrule(last(p[2])), rules)...)
end

tensorrank(::Selector, args::Tuple{Vararg{DTerm}}) = tensorrank(first(args))
function checkranks(::Selector, args::Tuple{Vararg{DTerm}})
    tensorrank(first(args)) > 0 && throw(ArgumentError("only scalar arguments are supported by selectors for now"))
    return
end

tensorrank(site::SiteSelector, args::Tuple{Vararg{DTerm}}) = tensorrank(site.selector, args)
checkranks(site::SiteSelector, args::Tuple{Vararg{DTerm}}) = checkranks(site.selector, args)

function lower_rule(sel::Selector, args, offs, block)
    pos = map(position, block, offs)
    # the number of faces and their codimensions are only known at runtime
    faces = Tuple(Tuple(axes) => i for (axes, i) in candidates(pos, sel.clauses))
    site = SiteSelector(sel, offs, faces)
    return lower_rule(site, args, offs, block)
end

function lower_rule(site::SiteSelector, args, offs, block)
    if offs != site.offsets
        return lower_rule(site.selector, args, offs, block)
    end
    isempty(site.faces) && return here(first(args), offs)
    axes, i = first(site.faces)
    rest = SiteSelector(
        site.selector,
        site.offsets,
        Base.tail(site.faces)
    )(args...)
    pos = map(position, block, offs)
    # rules are written in the frame of the outward normal, i.e. for the upper wall
    signed_axes = ntuple(k -> outward(pos[axes[k]].side) * axes[k], length(axes))
    rule = args[1 + i]
    lifted = Lifted(operation(rule), signed_axes)
    return lower_rule(lifted, (rest, arguments(rule)...), (), offs, block)
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
