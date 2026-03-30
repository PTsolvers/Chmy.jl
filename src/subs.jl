"""
    subs(expr, kvs::Pair...)

Replace occurrences in `expr` using a post-order traversal. Substitutions are
tried in the given order, and the first matching pair is applied.
"""
function subs(expr::STerm, kvs::Pair...)
    rules = map(SubsRule, kvs)
    return simplify(Postwalk(Chain(rules))(expr))
end

struct SubsRule{Lhs,Rhs} <: AbstractRule
    lhs::Lhs
    rhs::Rhs
end

SubsRule(kv::Pair) = SubsRule(kv.first, kv.second)

(rule::SubsRule{Lhs})(::Lhs) where {Lhs<:STerm} = rule.rhs
