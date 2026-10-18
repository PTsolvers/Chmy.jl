abstract type BoundaryRule end

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

struct BoundaryClause
    pattern::FacePattern
    region::GridRegion
    resolution::BoundaryRule
end

# check all combinations of Near axes
# create faces for them
# check which rules match
# sort the groups
function candidates(pos::Block{N}, clauses::Tuple{Vararg{BoundaryClause}}) where {N}
    # precompute codimensions of rules for known N once
    # this step also throws for malformed rules
    codims = map(c -> codim(c.pattern, N), clauses)
    # find all near axes of the position
    near = findall(x -> x isa Near, pos)
    # each group corresponds to a single face that could potentially contribute a rule to resolve the selector
    # we can have multiple rules in a group because some rules might return nothing, then the next one will be tried
    groups = Pair{Vector{Int}, Vector{Int}}[]
    # iterate over all faces spanning near axes of position
    # for p near axes, there are 2^p combinations of these axes
    # but the first one would exclude all near axes
    for mask in 1:(2^length(near) - 1)
        axes = [near[i] for i in eachindex(near) if !iszero(mask & 1 << (i - 1))]
        # create Face from the near axes
        face = ntuple(Val(N)) do i
            i ∈ axes ? pos[i].side : Span()
        end |> Face
        group = Int[]
        for (i, c) in enumerate(clauses)
            # check codim
            codims[i] == length(axes) || continue
            # check if face matches
            matches(c.pattern, face) || continue
            # check if region matches
            all(pos[k].ξ ∈ c.region for k in axes) || continue
            # all checks passed, add the candidate rule's index to the group
            push!(group, i)
        end
        !isempty(group) && push!(groups, axes => group)
    end
    # we sort groups by codim first
    # then by "severity" (ghost axes come first, then boundary, then near-boundary)
    # then by the index of the first matching rule
    # tie-break by the axis index (lower first)
    function priority(kv)
        s = sort!([severity(pos[i].ξ) for i in kv[1]])
        return (-length(kv[1]), s, first(kv[2]), kv[1])
    end
    sort!(groups; by = priority)
    return groups
end

struct Selector <: Operator
    clauses::Tuple{Vararg{BoundaryClause}}
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

struct FaceSite{N}
    selector::Selector
    arg::DTerm
    pos::Block{N}
    groups::Vector{Pair{Vector{Int}, Vector{Int}}}
    m::Int
end

function stencil_rule(sel::Selector, args::Tuple{DTerm}, ctx::LowerContext{N}) where {N}
    arg = only(args)
    pos = map(position, ctx.blck, ctx.offs)
    groups = candidates(pos, sel.clauses)
    return visit(sel, arg, pos, groups, 1)
end

function visit(sel::Selector, arg, pos, groups, m)
    m > length(groups) && return arg
    fs = FaceSite(sel, arg, pos, groups, m)
    _, group = groups[m]
    for i in group
        clause = sel.clauses[i]
        r = resolve(clause.resolution, fs)
        isnothing(r) || return r
    end
    return visit(sel, arg, pos, groups, m + 1)
end

next(fs) = visit(fs.selector, fs.arg, fs.pos, fs.groups, fs.m + 1)

resolve(::Next, fs::FaceSite) = next(fs)
resolve(a::Add, fs::FaceSite) = a.val + next(fs)
resolve(s::Scale, fs::FaceSite) = s.val * next(fs)
