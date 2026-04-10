"""
    SNode(arg)

Protect `arg` from symbolic tree walks.

`SNode` is an inert wrapper aroung a symbolic term: rewriters and traversals treat it as a black box.
When a protected tensor is expanded into components, each scalar component is wrapped back in `SNode`
so parent expressions can still lower around it. Use [`unwrap`](@ref) to
remove the wrapper and re-enter ordinary symbolic evaluation.
"""
struct SNode{T<:STerm} <: STerm
    arg::T
end

argument(node::SNode) = node.arg
tensorrank(node::SNode) = tensorrank(argument(node))

"""
    node(term)

Wrap `term` in an inert [`SNode`](@ref).
"""
node(term::SNode) = term
node(term::SLiteral) = term
node(term::STerm) = SNode(term)
function node(tensor::Tensor{D,R,K}) where {D,R,K}
    comps = tuplemap(node, tensor.components)
    return construct_tensor(Tensor{D,R,K}, comps)
end
node(term) = node(STerm(term))

# `SNode` stays opaque to tree walks, but tensor expansion distributes the
# protection to each scalar component so parent expressions can lower normally.
Tensor{D}(term::SNode) where {D} = node(Tensor{D}(argument(term)))

"""
    unwrap(term)

Recursively remove all [`SNode`](@ref) wrappers from `term` and symbolically
evaluate the rebuilt result.

This is the inverse of [`node`](@ref): once protected subtrees are unwrapped,
ordinary Chmy indexing, lowering, and simplification rules are applied again.
"""
unwrap(term) = term
unwrap(node::SNode) = unwrap(argument(node))
unwrap(expr::SExpr) = evaluate(SExpr(head(expr), tuplemap(unwrap, children(expr))...))

Base.getindex(n::SNode, I::Vararg{Integer,N}) where {N} = n[tuplemap(STerm, I)...]

Base.getindex(n::SNode, inds::Vararg{SLiteral,N}) where {N} = node(argument(n)[inds...])

# Scalar `SNode`s remain opaque by default. If the wrapped scalar still carries
# deferred tensor-component information, materialize just that component first
# and keep the spatial indexing outside the wrapper.
function Base.getindex(n::SNode, inds::Vararg{STerm,N}) where {N}
    tensorrank(n) == 0 || throw(ArgumentError("grid indexing requires a scalar term; take tensor components of $n first"))
    iscomp(argument(n)) && return SExpr(Ind(), Tensor{N}(n), inds...)
    return SExpr(Ind(), n, inds...)
end

function Base.getindex(n::SNode, loc::Vararg{Space,N}) where {N}
    tensorrank(n) == 0 || throw(ArgumentError("location indexing requires a scalar term; take tensor components of $n first"))
    iscomp(argument(n)) && return SExpr(Loc(), Tensor{N}(n), loc...)
    return SExpr(Loc(), n, loc...)
end
