abstract type BoundaryRule <: Operator end

struct Next <: BoundaryRule end
struct Add <: BoundaryRule
    val::DTerm
end
struct Scale <: BoundaryRule
    val::DTerm
end

struct Dirichlet <: BoundaryRule
    g::DTerm
end

struct Neumann <: BoundaryRule
    g::DTerm
end

struct BoundaryClause
    pattern::FacePattern
    region::GridRegion
    resolution::BoundaryRule
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

function select(term::DTerm, rules::Pair...)
    clauses = map(p -> BoundaryClause(p[1], p[2]...), rules)
    return Selector(clauses)(term)
end

tensorrank(::Selector, args::Tuple{DTerm}) = tensorrank(only(args))
function checkranks(::Selector, args::Tuple{DTerm})
    tensorrank(only(args)) > 0 && throw(ArgumentError("only scalar arguments are supported by selectors for now"))
    return
end

tensorrank(site::SiteSelector, args::Tuple{DTerm}) = tensorrank(site.selector, args)
checkranks(site::SiteSelector, args::Tuple{DTerm}) = checkranks(site.selector, args)

function stencil_rule(sel::Selector, args::Tuple{DTerm}, ctx::LowerContext{N}) where {N}
    pos = map(position, ctx.blck, ctx.offs)
    # the number of faces and their codimensions are only known at runtime
    faces = Tuple(Tuple(axes) => i for (axes, i) in candidates(pos, sel.clauses))
    site = SiteSelector(sel, ctx.offs, faces)
    return stencil_rule(site, args, ctx)
end

function stencil_rule(site::SiteSelector{N}, args::Tuple{DTerm}, ctx::LowerContext{N}) where {N}
    if ctx.offs != site.offsets
        return stencil_rule(site.selector, args, ctx)
    end
    arg = only(args)
    isempty(site.faces) && return here(arg, ctx)
    axes, i = first(site.faces)
    rest = SiteSelector(
        site.selector,
        site.offsets,
        Base.tail(site.faces)
    )(args...)
    pos = map(position, ctx.blck, ctx.offs)
    signed_axes = ntuple(length(axes)) do k
        a = axes[k]
        pos[a].side isa Upper ? -a : a
    end
    lifted = Lifted(site.selector.clauses[i].resolution, signed_axes)
    return stencil_rule(lifted, (rest,), ctx)
end

# the value of `x` at the sample being resolved
here(x::DTerm, ::LowerContext{N}) where {N} = x[zero_offsets(Val(N))...]

function stencil_rule(r::BoundaryRule, args, ::LowerContext{N}) where {N}
    return throw(ArgumentError(lazy"rule of type `$(typeof(r))` has no `stencil_rule` defined for codimension $N"))
end

stencil_rule(::Next, args::Tuple{DTerm}, ctx::LowerContext) = here(only(args), ctx)
stencil_rule(a::Add, args::Tuple{DTerm}, ctx::LowerContext) = here(only(args), ctx) + here(a.val, ctx)
stencil_rule(s::Scale, args::Tuple{DTerm}, ctx::LowerContext) = here(s.val, ctx) * here(only(args), ctx)
