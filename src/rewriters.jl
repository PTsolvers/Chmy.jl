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

struct LowerStencil <: AbstractRule end

Base.@assume_effects :foldable function (::LowerStencil)(t::SExpr{Ind})
    arg = argument(t)
    inds = indices(t)

    isa(arg, SUniform) && return arg

    isexpr(arg) || return t

    # `arg[loc...][inds...]` is represented as `Ind(Loc(arg, loc...), inds...)`,
    # so lower the location and index parts together when present.
    if isloc(arg)
        return _lower_loc(argument(arg), location(arg), inds)
    else
        return _lower_ind(arg, inds)
    end
end

function _lower_loc(t::STerm, loc::NTuple{N,Space}, inds::NTuple{N,STerm}) where {N}
    # Leaf terms and components accept direct indexing; calls must be distributed.
    (!isexpr(t) || iscomp(t)) && return t[loc...][inds...]
    if iscall(t)
        return stencil_rule(operation(t), arguments(t), loc, inds)
    else
        error("malformed static expression")
    end
end

function _lower_ind(t::STerm, inds::NTuple{N,STerm}) where {N}
    # Leaf terms and components accept direct indexing; calls must be distributed.
    (!isexpr(t) || iscomp(t)) && return t[inds...]
    if iscall(t)
        return stencil_rule(operation(t), arguments(t), inds)
    else
        error("malformed static expression")
    end
end

"""
    stencil_rule(op, args, inds)
    stencil_rule(op, args, loc, inds)

Distribute a stencil operation over `args`, indexing each argument with the
provided symbolic indices (and optional staggered locations).
"""
function stencil_rule(op::Union{SRef,SFun}, args::Tuple{Vararg{STerm}}, loc::NTuple{N,Space}, inds::NTuple{N,STerm}) where {N}
    return SExpr(Call(), op, map(x -> x[loc...][inds...], args)...)
end

function stencil_rule(op::Union{SRef,SFun}, args::Tuple{Vararg{STerm}}, inds::NTuple{N,STerm}) where {N}
    return SExpr(Call(), op, map(x -> x[inds...], args)...)
end

struct InsertRule{I,N,Inds} <: AbstractRule
    inds::Inds
end

function InsertRule(inds::NTuple{N,STerm}, ::Val{I}, ::Val{N}) where {I,N}
    InsertRule{I,N,typeof(inds)}(inds)
end

function (rule::InsertRule{I,N})(term::SExpr{Ind}) where {I,N}
    ind = only(indices(term))
    # `lift` builds a 1D stencil in axis `I`; reinsert the fixed indices around it.
    new_inds = ntuple(j -> j == I ? ind : rule.inds[j], Val(N))
    return argument(term)[new_inds...]
end

"""
    lift(op, args, inds, Val(I))
    lift(op, args, loc, inds, Val(I))

Build an `N`-dimensional stencil expression by lifting a one-axis stencil along
dimension `I` and reinserting the fixed indices (and locations) in other axes.
"""
function lift(op::STerm, args, inds::NTuple{N,STerm}, ::Val{I}) where {I,N}
    expr = stencil_rule(op, args, (inds[I],))
    rule = InsertRule(inds, Val(I), Val(N))
    return Prewalk(rule)(expr)
end

struct InsertRuleLoc{I,N,Loc,Inds} <: AbstractRule
    loc::Loc
    inds::Inds
end

function InsertRuleLoc(loc::NTuple{N,Space}, inds::NTuple{N,STerm}, ::Val{I}, ::Val{N}) where {I,N}
    InsertRuleLoc{I,N,typeof(loc),typeof(inds)}(loc, inds)
end

function (rule::InsertRuleLoc{I,N})(term::SExpr{Loc}) where {I,N}
    loc = only(location(term))
    # Reinsert the lifted location into the full location tuple.
    new_loc = ntuple(j -> j == I ? loc : rule.loc[j], Val(N))
    return argument(term)[new_loc...]
end

function (rule::InsertRuleLoc{I,N})(term::SExpr{Ind}) where {I,N}
    ind = only(indices(term))
    # Reinsert the lifted index into the full index tuple.
    new_inds = ntuple(j -> j == I ? ind : rule.inds[j], Val(N))
    return argument(term)[new_inds...]
end

function lift(op::STerm, args, loc::NTuple{N,Space}, inds::NTuple{N,STerm}, ::Val{I}) where {I,N}
    expr = stencil_rule(op, args, (loc[I],), (inds[I],))
    rule = InsertRuleLoc(loc, inds, Val(I), Val(N))
    return Prewalk(rule)(expr)
end

"""
    lower_stencil(expr)

Lower indexed stencil expressions by distributing indexing into call arguments.
"""
lower_stencil(expr::STerm) = Prewalk(LowerStencil())(expr)

struct SubsRule{Lhs,Rhs} <: AbstractRule
    lhs::Lhs
    rhs::Rhs
end

SubsRule(kv::Pair) = SubsRule(kv.first, kv.second)

(rule::SubsRule{Lhs})(::Lhs) where {Lhs<:STerm} = rule.rhs

"""
    subs(expr, lhs => rhs)

Replace occurrences of `lhs` with `rhs` in `expr` using a post-order traversal.
"""
subs(expr::STerm, kv::Pair) = simplify(Postwalk(SubsRule(kv))(expr))
