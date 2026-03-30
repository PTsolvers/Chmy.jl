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

Base.getindex(term::STerm, I::Vararg{IntegerOrSLiteral,N}) where {N} = term[tuplemap(STerm, I)...]

function Base.getindex(expr::SExpr{Call}, I::Vararg{SLiteral,N}) where {N}
    N == 0 && return expr
    R = tensorrank(expr)
    R == 0 && return lower_ind(Tensor{N}(expr), I)
    N == R || throw(ArgumentError("expected $R tensor component indices, got $N"))
    return component(expr, I)
end

function Base.getindex(expr::SExpr{Comp}, I::Vararg{SLiteral,N}) where {N}
    N == 0 && return expr
    return lower_ind(Tensor{N}(expr), I)
end

function Base.getindex(expr::SExpr{Loc}, inds::Vararg{STerm,N}) where {N}
    N == 0 && return expr
    return lower_loc(Tensor{N}(argument(expr)), location(expr), inds)
end

function Base.getindex(expr::SExpr{Call}, inds::Vararg{STerm,N}) where {N}
    N == 0 && return expr
    tensorrank(expr) == 0 || throw(ArgumentError("tensor expression '$expr' can only be component-indexed by SLiterals"))
    return lower_ind(Tensor{N}(expr), inds)
end

function Base.getindex(expr::SExpr{Comp}, inds::Vararg{STerm,N}) where {N}
    N == 0 && return expr
    return lower_ind(Tensor{N}(expr), inds)
end

function Base.getindex(expr::SExpr{Call}, loc::Vararg{Space,N}) where {N}
    N == 0 && return expr
    tensorrank(expr) == 0 || throw(ArgumentError("location requires a scalar expression; take tensor components of '$expr' first"))
    return locate_scalar(Tensor{N}(expr), loc)
end

function Base.getindex(expr::SExpr{Comp}, loc::Vararg{Space,N}) where {N}
    N == 0 && return expr
    return locate_scalar(Tensor{N}(expr), loc)
end

component(t::STerm, I::NTuple{N,SLiteral}) where {N} = SExpr(Comp(), t, I...)
Base.@assume_effects :foldable function component(t::SExpr{Call}, I::NTuple{N,SLiteral}) where {N}
    return component(operation(t), arguments(t), I, t)
end
component(::SRef{:+}, args::Tuple{Vararg{STerm}}, I, t) = +(map(arg -> arg[I...], args)...)
component(::SRef{:-}, args::Tuple{STerm}, I, t) = -only(args)[I...]
function component(::SRef{:-}, args::Tuple{STerm,STerm}, I, t)
    a, b = args
    return a[I...] - b[I...]
end

tensor_component_arg(::Tuple{}) = nothing
function tensor_component_arg(args::Tuple)
    tensorrank(first(args)) > 0 && return 1
    tail = tensor_component_arg(Base.tail(args))
    isnothing(tail) && return nothing
    return 1 + tail
end

function component(::SRef{:*}, args::Tuple{Vararg{STerm}}, I, t)
    j = tensor_component_arg(args)
    isnothing(j) && return SExpr(Comp(), t, I...)
    new_args = ntuple(k -> k == j ? args[k][I...] : args[k], Val(length(args)))
    return *(new_args...)
end

# default rule is to take the component of the whole expression
component(::STerm, ::Tuple{Vararg{STerm}}, I, t) = SExpr(Comp(), t, I...)

ispointwise(::STerm) = false
ispointwise(::Union{SRef,SFun}) = true

function locate_scalar(t::STerm, loc::NTuple{N,Space}) where {N}
    isa(t, SLiteral) && return t
    if iscall(t) && ispointwise(operation(t))
        return SExpr(Call(), operation(t), map(x -> x[loc...], arguments(t))...)
    else
        return SExpr(Loc(), t, loc...)
    end
end

function lower_loc(t::STerm, loc::NTuple{N,Space}, inds::NTuple{N,STerm}) where {N}
    isa(t, SLiteral) && return t
    (!isexpr(t) || iscomp(t)) && return SExpr(Ind(), SExpr(Loc(), t, loc...), inds...)
    if iscall(t)
        return evaluate(stencil_rule(operation(t), arguments(t), loc, inds))
    else
        error("malformed static expression $t")
    end
end

function lower_ind(t::STerm, inds::NTuple{N,STerm}) where {N}
    isa(t, SLiteral) && return t
    (!isexpr(t) || iscomp(t)) && return SExpr(Ind(), t, inds...)
    if iscall(t)
        return evaluate(stencil_rule(operation(t), arguments(t), inds))
    else
        error("malformed static expression $t")
    end
end
