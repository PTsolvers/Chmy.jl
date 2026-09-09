"""
    stencil_rule(op, args, inds)
    stencil_rule(op, args, locs, inds)

Distribute a stencil operation over `args`, indexing each argument with the
provided symbolic indices (and optional staggered locations).
"""
function stencil_rule end

stencil_rule(op::Operator, args, locs, inds) = _stencil_rule(op, args, locs, inds)
stencil_rule(op::Operator, args, inds) = _stencil_rule(op, args, inds)

function _stencil_rule(op, args::Tuple{DTerm}, locs::Tuple{Location}, inds::Tuple{DTerm})
    return stencil_rule(op, only(args), only(locs), only(inds))
end
function _stencil_rule(op, args::Tuple{DTerm}, inds::Tuple{DTerm})
    return stencil_rule(op, only(args), only(inds))
end

function stencil_rule(op::Fun, args::Tuple{Vararg{DTerm}}, locs::NTuple{N,Location}, inds::NTuple{N,DTerm}) where {N}
    return makecall_unchecked(op, map(x -> x[locs...][inds...], args))
end
function stencil_rule(op::Fun, args::Tuple{Vararg{DTerm}}, inds::NTuple{N,DTerm}) where {N}
    return makecall_unchecked(op, map(x -> x[inds...], args))
end

"""
    lower(expr)
    lower(expr, inds)
    lower(expr, locs)
    lower(expr, locs, inds)

Lower `expr` into a stencil form.
"""
function lower(expr::DTerm)::DTerm
    check_lowerable(expr)
    return lower_expr(expr)
end

function lower(expr::DTerm, locs::NTuple{N,Location}, inds::NTuple{N,DTerm}) where {N}
    check_lowerable(expr)
    return lower_at(expr, locs, inds)
end

function lower(expr::DTerm, inds::NTuple{N,DTerm}) where {N}
    check_lowerable(expr)
    return lower_at(expr, nothing, inds)
end

function lower(expr::DTerm, locs::NTuple{N,Location}) where {N}
    inds = ntuple(Index, Val(N))
    return lower(expr, locs, inds)
end

@noinline lower_component_error() = throw(ArgumentError("lower requires scalar component form; call components(expr, dims) before lowering and select a scalar entry if needed"))

# validate before expanding
function check_lowerable(expr::DTerm)
    @match expr begin
        DExpr(Comp(I), (arg,)) => begin
            istensor(arg) && tensorrank(arg) == length(I) || lower_component_error()
        end
        DExpr(Call(op), children) => begin
            foreach(check_lowerable, children)
            # checked scalar arguments imply scalar arithmetic
            if !(op isa Union{Fun{typeof(+)},Fun{typeof(-)},Fun{typeof(*)},
                              Fun{typeof(/)},Fun{typeof(//)},Fun{typeof(÷)}})
                tensorrank(expr) == 0 || lower_component_error()
            end
        end
        DExpr(_, children) => foreach(check_lowerable, children)
        _ => tensorrank(expr) == 0 || lower_component_error()
    end
    return nothing
end

lower_expr(expr::DTerm)::DTerm = something(lower_changed(expr), expr)

function lower_changed(expr::DTerm)::Union{Nothing,DTerm}
    @match expr begin
        DExpr(Inds(), children) => begin
            arg = first(children)
            field = if islocs(arg)
                if length(locations(arg)) != length(children)-1
                    throw(ArgumentError("locations and grid indices must have the same length"))
                end
                argument(arg)
            else
                arg
            end
            if iscall(field) || islocs(field) || isinds(field)
                return lower_sample(children)
            end
            return isuniform(field) ? field : lower_args(expr)
        end
        DExpr(Comp(_), _) => nothing
        DExpr(Locs(_), (arg,)) => begin
            if !isexpr(arg) || iscomp(arg)
                return isuniform(arg) ? arg : nothing
            end
            return lower_args(expr)
        end
        DExpr(_, _) => lower_args(expr)
        _ => nothing
    end
end

# Specialize on arity only when a sample actually needs expansion. Base.tail
# then extracts the grid indices without the dynamic tuple slice of a rest match.
function lower_sample(children::NTuple{N,DTerm})::DTerm where {N}
    arg, inds = first(children), Base.tail(children)
    @match arg begin
        DExpr(Locs(locs), (field,)) => lower_at(field, locs, inds)
        _ => lower_at(arg, nothing, inds)
    end
end

function lower_args(expr::DTerm)::Union{Nothing,DTerm}
    children = args(expr)
    for k in eachindex(children)
        child = lower_changed(children[k])
        isnothing(child) && continue
        return DExpr(head(expr), walkargs(lower_expr, children, child, k))
    end
    return nothing
end

function lower_at(expr::DTerm, locs, inds::NTuple{N,DTerm})::DTerm where {N}
    isnothing(locs) || length(locs) == N || throw(ArgumentError("locations and grid indices must have the same length"))
    @match expr begin
        DExpr(Call(op), children) => begin
            result = if isnothing(locs)
                stencil_rule(op, children, inds)::DTerm
            else
                stencil_rule(op, children, locs, inds)::DTerm
            end
            return lower_expr(result)
        end
        DExpr(Locs(explicit), (arg,)) => return lower_at(arg, explicit, inds)
        DExpr(Inds(), _) => return lower_expr(expr)
        _ => begin
            isuniform(expr) && return expr
            field = isnothing(locs) ? expr : makelocs_unchecked(expr, locs)
            return makeinds_unchecked(field, inds)
        end
    end
end
