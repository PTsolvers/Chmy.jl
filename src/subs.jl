"""
    subs(expr, kvs::Pair...)

Replace occurrences in `expr` using a post-order traversal. Substitutions are
tried in the given order, and the first matching pair is applied.
"""
function subs(expr::DTerm, kvs::Pair...)
    rules = map(SubsRule, kvs)
    return Postwalk(Chain(rules))(expr)
end

struct SubsRule <: AbstractRule
    lhs::DTerm
    rhs::DTerm
end

SubsRule(kv::Pair) = SubsRule(kv.first, kv.second)

(rule::SubsRule)(lhs::DTerm) = lhs==ₛrule.lhs ? rule.rhs : nothing
