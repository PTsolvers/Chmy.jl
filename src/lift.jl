"""
    lift(op, args, inds, Val(I))
    lift(op, args, loc, inds, Val(I))

Build an `N`-dimensional stencil expression by lifting a one-axis stencil along
dimension `I` and reinserting the fixed indices (and locations) in other axes.
"""
Base.@assume_effects :foldable function lift(op::STerm, args, inds::NTuple{N,STerm}, ::Val{I}) where {I,N}
    expr = stencil_rule(op, tuplemap(Stub, args), (inds[I],))
    rule = InsertRule(inds, Val(I), Val(N))
    return evaluate(unwrap(Prewalk(rule)(expr)))
end
Base.@assume_effects :foldable function lift(op::STerm, args, loc::NTuple{N,Space}, inds::NTuple{N,STerm}, ::Val{I}) where {I,N}
    expr = stencil_rule(op, tuplemap(Stub, args), (loc[I],), (inds[I],))
    rule = InsertRuleLoc(loc, inds, Val(I), Val(N))
    return evaluate(unwrap(Prewalk(rule)(expr)))
end

# Lifting embeds a 1D stencil rule into an N-dimensional indexing/location
# context. The rule itself is still written using ordinary Chmy indexing syntax,
# so we need a way to postpone the eager lowering until the full N-dimensional
# indices and locations have been reconstructed.

# `Stub` is a wrapper for scalar arguments while a 1D rule is being built.
# If lifting used the original arguments directly, indexing inside a 1D stencil
# rule would immediately trigger the normal expression indexing logic. That would
# lower or propagate the partially lifted expression before we had a chance to
# reinsert the untouched axes.
struct Stub{T<:STerm} <: STerm
    arg::T
    function Stub(arg::T) where {T<:STerm}
        tensorrank(arg) == 0 || throw(ArgumentError("Stub can only wrap scalar terms"))
        return new{T}(arg)
    end
end

tensorrank(::Stub) = 0

# Recursively remove lift-local placeholders before the expression re-enters the
# ordinary evaluation/indexing machinery.
unwrap(term::STerm) = term
unwrap(stub::Stub) = stub.arg
unwrap(expr::SExpr) = SExpr(head(expr), tuplemap(unwrap, children(expr))...)

# Stub indexing intentionally preserves the user-written indexing
# structure from the 1D stencil rule without triggering generic lowering.
Base.getindex(stub::Stub, inds::Vararg{STerm}) = SExpr(Ind(), stub, inds...)

function Base.getindex(stub::Stub, loc::Vararg{Space})
    length(loc) == 0 && return stub
    return SExpr(Loc(), stub, loc...)
end

# Keep nested `arg[loc...][inds...]` intact when the argument is a `Stub`.
function Base.getindex(expr::SExpr{Loc,<:Tuple{Stub,Vararg}}, inds::Vararg{STerm,N}) where {N}
    N == 0 && return expr
    return SExpr(Ind(), expr, inds...)
end

# Reinsert the lifted index into the full N-dimensional index tuple, keeping the
# non-lifted axes fixed to the values originally passed to `lift`.
struct InsertRule{I,N,Inds} <: AbstractRule
    inds::Inds
end
function InsertRule(inds::NTuple{N,STerm}, ::Val{I}, ::Val{N}) where {I,N}
    return InsertRule{I,N,typeof(inds)}(inds)
end

function (rule::InsertRule{I,N})(term::SExpr{Ind}) where {I,N}
    ind = only(indices(term))
    new_inds = ntuple(j -> j == I ? ind : rule.inds[j], Val(N))
    return SExpr(Ind(), argument(term), new_inds...)
end

# Variant of `InsertRule` for staggered operators that also rebuilds the full
# location tuple around the lifted axis.
struct InsertRuleLoc{I,N,Loc,Inds} <: AbstractRule
    loc::Loc
    inds::Inds
end
function InsertRuleLoc(loc::NTuple{N,Space}, inds::NTuple{N,STerm}, ::Val{I}, ::Val{N}) where {I,N}
    return InsertRuleLoc{I,N,typeof(loc),typeof(inds)}(loc, inds)
end

function (rule::InsertRuleLoc{I,N})(term::SExpr{Loc}) where {I,N}
    loc = only(location(term))
    new_loc = ntuple(j -> j == I ? loc : rule.loc[j], Val(N))
    return SExpr(Loc(), argument(term), new_loc...)
end
function (rule::InsertRuleLoc{I,N})(term::SExpr{Ind}) where {I,N}
    ind = only(indices(term))
    new_inds = ntuple(j -> j == I ? ind : rule.inds[j], Val(N))
    return SExpr(Ind(), argument(term), new_inds...)
end
