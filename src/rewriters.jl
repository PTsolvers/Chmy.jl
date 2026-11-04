# symbolic expression rewriters inspired by SymbolicUtils.jl

"""
    AbstractRule

Base type for symbolic rewrite rules.

Rules are callable objects that take a `DTerm` and return either a replacement
`DTerm` or `nothing` to indicate "no match".
"""
abstract type AbstractRule end

# no match by default
(::AbstractRule)(::DTerm) = nothing

"""
    Passthrough(rule)

Wrap a rule so unmatched terms are returned unchanged instead of `nothing`.
"""
struct Passthrough{R} <: AbstractRule
    rule::R
end
Passthrough(rule::Passthrough) = rule

function (p::Passthrough)(term::DTerm)
    new_term = p.rule(term)::Union{Nothing, DTerm}
    isnothing(new_term) && return term
    return new_term
end

"""
    Chain(rules...)
    Chain(rules::Tuple)

Try `rules` in order and return the first successful rewrite (the first result
that is not `nothing`). If no rule matches, return `nothing`.
"""
struct Chain{Rs <: Tuple} <: AbstractRule
    rules::Rs
end
Chain(chain::Chain) = chain
Chain(rules...) = Chain(rules)

chainfirst(::Tuple{}, ::DTerm) = nothing
function chainfirst(rules::Tuple, term::DTerm)
    new_term = first(rules)(term)::Union{Nothing, DTerm}
    isnothing(new_term) || return new_term
    return chainfirst(Base.tail(rules), term)
end

(c::Chain)(term::DTerm) = chainfirst(c.rules, term)

"""
    Prewalk(rule)

Apply `rule` in a top-down traversal (parent before children), visiting the
replacement's arguments from left to right. Leaves and expression heads are atomic.
"""
struct Prewalk{R} <: AbstractRule
    rule::R
end

(p::Prewalk)(term::DTerm) = walkargs(p, Passthrough(p.rule)(term))

"""
    Postwalk(rule)

Apply `rule` in a bottom-up traversal (children before parent), visiting arguments
from left to right. Children introduced by the final rewrite are not visited.
Expressions with rewritten children are rebuilt in canonical form.
"""
struct Postwalk{R} <: AbstractRule
    rule::R
end

(p::Postwalk)(term::DTerm) = Passthrough(p.rule)(walkargs(p, term))

function walkargs(walk, term::DTerm)
    return @match term begin
        DExpr(head, args) => begin
            for k in eachindex(args)
                arg = walk(args[k])
                arg === args[k] && continue
                return maketerm(head, walkargs(walk, args, arg, k))
            end
            term
        end
        AddMul(_, _, _) => walkaddmul(walk, term)
        _ => term
    end
end

# the arguments of a sum or a product are walked, and it is rebuilt from a vector if one of them changes
function walkaddmul(walk, term::DTerm)
    args = addmul_arguments(term)
    walked = DTerm[walk(a) for a in args]
    all(i -> walked[i] === args[i], eachindex(args)) && return term
    return evaluate(addmul_operation(term), walked)::DTerm
end

# specialize on tuple length only when walk changes one of the children
function walkargs(walk, args::NTuple{N, DTerm}, arg::DTerm, k::Int) where {N}
    return ntuple(Val(N)) do j
        j < k ? args[j] : j == k ? arg : walk(args[j])
    end
end

"""
    Fixpoint(rule)

Repeatedly apply `rule` until it returns `nothing` or a structurally equal term
"""
struct Fixpoint{R} <: AbstractRule
    rule::R
end

function (f::Fixpoint)(term::DTerm)
    while true
        new_term = f.rule(term)::Union{Nothing, DTerm}
        (isnothing(new_term) || new_term ==ₛ term) && return term
        term = new_term
    end
    return
end
