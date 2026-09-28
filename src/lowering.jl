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

function stencil_rule(op::Fun, args::Tuple{Vararg{DTerm}}, locs::NTuple{N, Location}, inds::NTuple{N, DTerm}) where {N}
    return makecall_unchecked(op, map(x -> x[locs...][inds...], args))
end
function stencil_rule(op::Fun, args::Tuple{Vararg{DTerm}}, inds::NTuple{N, DTerm}) where {N}
    return makecall_unchecked(op, map(x -> x[inds...], args))
end

"""
    lower(expr)
    lower(expr, inds)
    lower(expr, locs)
    lower(expr, locs...)
    lower(expr, locs, inds)

Lower `expr` into a stencil form. Tensor operations in `expr` are first expanded with
[`components`](@ref) in `length(inds)` dimensions. Without explicit locations and grid
indices, `expr` must be a grid sample `f[inds...]` or `f[locs...][inds...]`.
"""
function lower(expr::DTerm)::DTerm
    isinds(expr) || throw(ArgumentError("lower without grid indices requires a grid sample 'f[inds...]' or 'f[locs...][inds...]'"))
    return lower_sampled(children(expr))
end
function lower(expr::DTerm, locs::NTuple{N, Location}, inds::NTuple{N, DTerm}) where {N}
    return lower_at(scalar_components(expr, N), locs, inds)
end
function lower(expr::DTerm, inds::NTuple{N, DTerm}) where {N}
    return lower_at(scalar_components(expr, N), nothing, inds)
end
function lower(expr::DTerm, locs::NTuple{N, Location}) where {N}
    inds = ntuple(Index, Val(N))
    return lower(expr, locs, inds)
end
lower(::DTerm, ::Tuple{Vararg{Location}}, ::Tuple{Vararg{DTerm}}) = locs_length_error()
lower(expr::DTerm, locs::Vararg{Location, N}) where {N} = lower(expr, locs)
lower(expr::DTerm, loc::Location, ind::DTerm) = lower(expr, (loc,), (ind,))
lower(expr::DTerm, ind::DTerm) = lower(expr, (ind,))
lower(expr::DTerm, loc::Location) = lower(expr, (loc,))

# specialize on arity to split the sampled field from the grid indices without a dynamic tuple slice
function lower_sampled(args::NTuple{N, DTerm})::DTerm where {N}
    arg, inds = first(args), Base.tail(args)
    return @match arg begin
        DExpr(Locs(locs), (field,)) => lower(field, locs, inds)
        _ => lower(arg, inds)
    end
end

@noinline locs_length_error() = throw(ArgumentError("locations and grid indices must have the same length"))
@noinline lower_rank_error() = throw(ArgumentError("lower requires a scalar expression; select a scalar entry, e.g. expr[1, 2]"))
@noinline lower_dims_error() = throw(ArgumentError("nested grid samples must have the same number of indices as the enclosing grid sample"))

function scalar_components(expr::DTerm, dims::Integer)
    tensorrank(expr) == 0 || lower_rank_error()
    return components(expr, dims)::DTerm
end

lower_expr(expr::DTerm)::DTerm = something(lower_changed(expr), expr)

function lower_changed(expr::DTerm)::Union{Nothing, DTerm}
    return @match expr begin
        DExpr(Inds(), args) => begin
            arg = first(args)
            field = if islocs(arg)
                if length(locations(arg)) != length(args) - 1
                    locs_length_error()
                end
                argument(arg)
            else
                arg
            end
            if iscall(field) || islocs(field) || isinds(field)
                lower_sample(args)
            else
                isuniform(field) ? field : lower_args(expr)
            end
        end
        DExpr(Comp(_), _) => nothing
        DExpr(Locs(_), (arg,)) => begin
            if !isexpr(arg) || iscomp(arg)
                isuniform(arg) ? arg : nothing
            else
                lower_args(expr)
            end
        end
        DExpr(_, _) => lower_args(expr)
        _ => nothing
    end
end

# specialize on arity only when a sample actually needs expansion. Base.tail
# then extracts the grid indices without the dynamic tuple slice of a rest match.
function lower_sample(args::NTuple{N, DTerm})::DTerm where {N}
    arg, inds = first(args), Base.tail(args)
    return @match arg begin
        DExpr(Locs(locs), (field,)) => lower_at(field, locs, inds)
        _ => lower_at(arg, nothing, inds)
    end
end

function lower_args(expr::DTerm)::Union{Nothing, DTerm}
    args = children(expr)
    for k in eachindex(args)
        arg = lower_changed(args[k])
        isnothing(arg) && continue
        return DExpr(head(expr), walkargs(lower_expr, args, arg, k))
    end
    return nothing
end

function lower_at(expr::DTerm, locs, inds::NTuple{N, DTerm})::DTerm where {N}
    isnothing(locs) || length(locs) == N || locs_length_error()
    return @match expr begin
        DExpr(Call(op), args) => begin
            result = if isnothing(locs)
                stencil_rule(op, args, inds)::DTerm
            else
                stencil_rule(op, args, locs, inds)::DTerm
            end
            lower_expr(result)
        end
        DExpr(Locs(explicit), (arg,)) => lower_at(arg, explicit, inds)
        DExpr(Inds(), args) => begin
            length(args) - 1 == N || lower_dims_error()
            lower_expr(expr)
        end
        _ => begin
            isuniform(expr) && return expr
            field = isnothing(locs) ? expr : makelocs_unchecked(expr, locs)
            makeinds_unchecked(field, inds)
        end
    end
end
