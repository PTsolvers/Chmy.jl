"""
    evaluate(term)

Symbolically evaluate a Chmy term.

`evaluate` rebuilds expressions by recursively evaluating their children and then
reapplying the symbolic operation. In contrast to [`compute`](@ref), it preserves
symbolic objects such as `SUniform`, `SIndex`, and symbolic tensor/field terms.
"""
evaluate(term::Tensor) = term
evaluate(term::STerm) = term

# For built-in symbolic references, emit a direct call to the referenced Julia
# function so the compiler can inline/fold the rebuilt symbolic expression.
@generated function evaluate(::SRef{F}, args::Vararg{Any,N}) where {F,N}
    ex = Expr(:call, F)
    for i in 1:N
        push!(ex.args, :(args[$i]))
    end
    return quote
        @inline
        return $ex
    end
end

# Broadcasting is represented symbolically, so evaluation just reconstructs the
# broadcasted call on already-evaluated arguments.
function evaluate(::SRef{:broadcasted}, op::SFun, args::Vararg{Any,N}) where {N}
    return Base.Broadcast.broadcasted(op.f, args...)
end
@generated function evaluate(::SRef{:broadcasted}, ::SRef{F}, args::Vararg{Any,N}) where {F,N}
    ex = Expr(:call, :(Base.Broadcast.broadcasted), F)
    for i in 1:N
        push!(ex.args, :(args[$i]))
    end
    return quote
        @inline
        return $ex
    end
end
evaluate(sf::SFun, args::Vararg{Any,N}) where {N} = sf.f(args...)
evaluate(op::STerm, args::Vararg{Any,N}) where {N} = op(args...)

# `Call` nodes are evaluated structurally: evaluate the children first, then
# reapply the operation on the rebuilt arguments.
Base.@assume_effects :foldable evaluate(expr::SExpr{Call}) = evaluate(operation(expr), tuplemap(evaluate, arguments(expr))...)

# Component, location, and grid indexing are evaluated by recursively evaluating
# the base term and the index/location arguments, then reusing ordinary `getindex`
# dispatch. This keeps all indexing semantics centralized in the main indexing
# methods instead of duplicating them here.
Base.@assume_effects :foldable evaluate(expr::SExpr{Loc}) = evaluate(argument(expr))[tuplemap(evaluate, location(expr))...]
Base.@assume_effects :foldable evaluate(expr::SExpr{Comp}) = evaluate(argument(expr))[tuplemap(evaluate, indices(expr))...]
Base.@assume_effects :foldable evaluate(expr::SExpr{Ind}) = evaluate(argument(expr))[tuplemap(evaluate, indices(expr))...]
Base.@assume_effects :foldable evaluate(expr::SExpr{Node}) = node(evaluate(argument(expr)))
