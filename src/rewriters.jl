abstract type AbstractRule end

# no match by default
(::AbstractRule)(::STerm) = nothing

struct Passthrough{R} <: AbstractRule
    rule::R
end

Passthrough(rule::Passthrough) = rule

function (p::Passthrough)(term::STerm)
    new_term = p.rule(term)
    isnothing(new_term) && return term
    return new_term
end

struct Prewalk{R} <: AbstractRule
    rule::R
end

Base.@assume_effects :foldable function (p::Prewalk)(term::STerm)
    rule = Passthrough(p.rule)
    new_term = rule(term)
    if isexpr(new_term)
        return SExpr(head(new_term), map(p, children(new_term)))
    else
        return new_term
    end
end

struct Postwalk{R} <: AbstractRule
    rule::R
end

Base.@assume_effects :foldable function (p::Postwalk)(term::STerm)
    rule = Passthrough(p.rule)
    if isexpr(term)
        new_term = SExpr(head(term), map(p, children(term)))
        return rule(new_term)
    else
        return rule(term)
    end
end

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

    if isloc(arg)
        return _lower_loc(argument(arg), location(arg), inds)
    else
        return _lower_ind(arg, inds)
    end
end

function _lower_loc(t::STerm, loc::NTuple{N,Space}, inds::NTuple{N,STerm}) where {N}
    (!isexpr(t) || iscomp(t)) && return t[loc...][inds...]
    if iscall(t)
        return stencil_rule(operation(t), arguments(t), loc, inds)
    else
        error("malformed static expression")
    end
end

function _lower_ind(t::STerm, inds::NTuple{N,STerm}) where {N}
    (!isexpr(t) || iscomp(t)) && return t[inds...]
    if iscall(t)
        return stencil_rule(operation(t), arguments(t), inds)
    else
        error("malformed static expression")
    end
end

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
    new_inds = ntuple(j -> j == I ? ind : rule.inds[j], Val(N))
    return argument(term)[new_inds...]
end

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
    new_loc = ntuple(j -> j == I ? loc : rule.loc[j], Val(N))
    return argument(term)[new_loc...]
end

function (rule::InsertRuleLoc{I,N})(term::SExpr{Ind}) where {I,N}
    ind = only(indices(term))
    new_inds = ntuple(j -> j == I ? ind : rule.inds[j], Val(N))
    return argument(term)[new_inds...]
end

function lift(op::STerm, args, loc::NTuple{N,Space}, inds::NTuple{N,STerm}, ::Val{I}) where {I,N}
    expr = stencil_rule(op, args, (loc[I],), (inds[I],))
    rule = InsertRuleLoc(loc, inds, Val(I), Val(N))
    return Prewalk(rule)(expr)
end

lower_stencil(expr::STerm) = Prewalk(LowerStencil())(expr)

struct SubsRule{Lhs,Rhs} <: AbstractRule
    lhs::Lhs
    rhs::Rhs
end

SubsRule(kv::Pair) = SubsRule(kv.first, kv.second)

(rule::SubsRule{Lhs})(::Lhs) where {Lhs<:STerm} = rule.rhs

subs(expr::STerm, kv::Pair) = Postwalk(SubsRule(kv))(expr)

struct LowerTensor{N} <: AbstractRule end

Base.@assume_effects :foldable function (r::LowerTensor{N})(term::STensor{R}) where {N,R}
    
end

Base.@assume_effects :foldable function (r::LowerTensor{N})(term::STerm) where {N}
    
end
