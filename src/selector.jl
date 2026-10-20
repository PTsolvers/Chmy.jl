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

function stencil_rule(sel::Selector, args::Tuple{Vararg{DTerm}}, ctx::LowerContext{N}) where {N}
    pos = map(position, ctx.blck, ctx.offs)
    # the number of faces and their codimensions are only known at runtime
    faces = Tuple(Tuple(axes) => i for (axes, i) in candidates(pos, sel.clauses))
    site = SiteSelector(sel, ctx.offs, faces)
    return stencil_rule(site, args, ctx)
end

function stencil_rule(site::SiteSelector{N}, args::Tuple{Vararg{DTerm}}, ctx::LowerContext{N}) where {N}
    if ctx.offs != site.offsets
        return stencil_rule(site.selector, args, ctx)
    end
    isempty(site.faces) && return here(first(args), ctx)
    axes, i = first(site.faces)
    rest = SiteSelector(
        site.selector,
        site.offsets,
        Base.tail(site.faces)
    )(args...)
    pos = map(position, ctx.blck, ctx.offs)
    # rules are written in the frame of the outward normal, i.e. for the upper wall
    signed_axes = ntuple(k -> outward(pos[axes[k]].side) * axes[k], length(axes))
    rule = args[1 + i]
    lifted = Lifted(operation(rule), signed_axes)
    return stencil_rule(lifted, (rest, arguments(rule)...), ctx)
end

# the value of `x` at the sample being resolved
here(x::DTerm, ::LowerContext{N}) where {N} = x[zero_offsets(Val(N))...]

function stencil_rule(r::BoundaryOperator, args, ::LowerContext{N}) where {N}
    return throw(ArgumentError(lazy"boundary operator of type `$(typeof(r))` has no `stencil_rule` defined for $(length(args)) arguments and codimension $N"))
end

stencil_rule(::Next, (u,)::Tuple{DTerm}, ctx::LowerContext) = here(u, ctx)
stencil_rule(::Add, (u, e)::Tuple{DTerm, DTerm}, ctx::LowerContext) = here(u, ctx) + here(e, ctx)
stencil_rule(::Scale, (u, w)::Tuple{DTerm, DTerm}, ctx::LowerContext) = here(w, ctx) * here(u, ctx)

ξ(ctx::LowerContext) = position(first(ctx.blck), first(ctx.offs)).ξ

trace(x::DTerm, ctx::LowerContext) = isuniform(x) ? x : x[-ξ(ctx)]

function stencil_rule(::Dirichlet, (u, g)::Tuple{DTerm, DTerm}, ctx::LowerContext{1})
    ξb = ξ(ctx)
    if ξb == 0
        return trace(g, ctx)
    end
    if ξb > 0
        # the mirror sample inside the domain
        return 2trace(g, ctx) - u[-2ξb]
    end
    throw(ArgumentError(lazy"Dirichlet rule can only be declared with Ghost() or OnBoundary() regions with ξ ≥ 0, got ξ = $ξb"))
end

# `g` is the outward normal derivative
function stencil_rule(::Neumann, (u, g)::Tuple{DTerm, DTerm}, ctx::LowerContext{1})
    ξb = ξ(ctx)
    if ξb > 0
        return u[-2ξb] + 2ξb * trace(g, ctx)
    end
    throw(ArgumentError(lazy"Neumann rule can only be declared with Ghost() region with ξ > 0, got ξ = $ξb"))
end
