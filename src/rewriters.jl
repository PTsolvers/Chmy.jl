# Symbolic expression rewriters inspired by SymbolicUtils.jl

"""
    AbstractRule

Base type for symbolic rewrite rules.

Rules are callable objects that take an `STerm` and return either a replacement
`STerm` or `nothing` to indicate "no match".
"""
abstract type AbstractRule end

# no match by default
(::AbstractRule)(::STerm) = nothing

"""
    Passthrough(rule)

Wrap a rule so unmatched terms are returned unchanged instead of `nothing`.
"""
struct Passthrough{R} <: AbstractRule
    rule::R
end

Passthrough(rule::Passthrough) = rule

function (p::Passthrough)(term::STerm)
    new_term = p.rule(term)
    isnothing(new_term) && return term
    return new_term
end

"""
    Chain(rules...)
    Chain(rules::Tuple)

Try `rules` in order and return the first successful rewrite (the first result
that is not `nothing`). If no rule matches, return `nothing`.
"""
struct Chain{Rs<:Tuple} <: AbstractRule
    rules::Rs
end

Chain(chain::Chain) = chain
Chain(rules...) = Chain(rules)

_chainfirst(::Tuple{}, ::STerm) = nothing

function _chainfirst(rules::Tuple, term::STerm)
    new_term = first(rules)(term)
    isnothing(new_term) || return new_term
    return _chainfirst(Base.tail(rules), term)
end

Base.@assume_effects :foldable function (c::Chain)(term::STerm)
    return _chainfirst(c.rules, term)
end

"""
    Prewalk(rule)

Apply `rule` in a top-down traversal (parent before children).
"""
struct Prewalk{R} <: AbstractRule
    rule::R
end

Base.@assume_effects :foldable function (p::Prewalk)(term::STerm)
    # Treat "no match" as identity so traversal only needs to handle `STerm`s.
    rule = Passthrough(p.rule)
    new_term = rule(term)
    if isexpr(new_term)
        return SExpr(head(new_term), map(p, children(new_term)))
    else
        return new_term
    end
end

"""
    Postwalk(rule)

Apply `rule` in a bottom-up traversal (children before parent).
"""
struct Postwalk{R} <: AbstractRule
    rule::R
end

Base.@assume_effects :foldable function (p::Postwalk)(term::STerm)
    # Postwalk rewrites descendants first, then gives the rebuilt node to `rule`.
    rule = Passthrough(p.rule)
    if isexpr(term)
        new_term = SExpr(head(term), map(p, children(term)))
        return rule(new_term)
    else
        return rule(term)
    end
end

"""
    Fixpoint(rule)

Repeatedly apply `rule` until it returns `nothing`.
"""
struct Fixpoint{R} <: AbstractRule
    rule::R
end

Base.@assume_effects :foldable function (f::Fixpoint)(term::STerm)
    new_term = f.rule(term)
    isnothing(new_term) && return term
    return f(new_term)
end
