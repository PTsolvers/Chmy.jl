"""
    stencil_rule(op, args, inds)
    stencil_rule(op, args, locs, inds)

Distribute a stencil operation over `args`, indexing each argument with the
provided symbolic indices (and optional staggered locations).
"""
function stencil_rule(op::SFun, args::Tuple{Vararg{STerm}}, locs::NTuple{N,Space}, inds::NTuple{N,STerm}) where {N}
    return SExpr(op, map(x -> x[locs...][inds...], args)...)
end
function stencil_rule(op::SFun, args::Tuple{Vararg{STerm}}, inds::NTuple{N,STerm}) where {N}
    return SExpr(op, map(x -> x[inds...], args)...)
end

Base.getindex(term::STerm, I::Vararg{IntegerOrSLiteral,N}) where {N} = term[tuplemap(STerm, I)...]

function Base.getindex(term::STerm, inds::Vararg{STerm,N}) where {N}
    isuniform(term) && return term
    return SExpr(SSub(), term, inds...)
end

function Base.getindex(term::STerm, locs::Vararg{Space,N}) where {N}
    isuniform(term) && return term
    return SExpr(SAt(locs), term)
end

function Base.getindex(expr::SExpr{SSub}, locs::Vararg{Space,N}) where {N}
    inds = indices(expr)
    arg = argument(expr)
    return arg[locs...][inds...]
end

function Base.getindex(t::Chmy.AbstractSTensor{R}, I::Vararg{SLiteral,N}) where {R,N}
    N == R || throw(ArgumentError("expected $R tensor component indices, got $N"))
    return SExpr(SComp(I), t)
end

function Base.getindex(t::AbstractSTensor{R}, loc::Vararg{Space,N}) where {R,N}
    R == 0 || throw(ArgumentError("location indexing requires a scalar term; take tensor components first"))
    return locate_scalar(t, loc)
end

function Base.getindex(t::AbstractSTensor{R}, inds::Vararg{STerm,N}) where {R,N}
    R == 0 && return lower_ind(t, inds)
    throw(ArgumentError("tensor terms with rank > 0 can only be component-indexed by SLiterals"))
end

function Base.getindex(::Tensor{D,R}, ::Vararg{Space,N}) where {D,R,N}
    throw(ArgumentError("location indexing requires a scalar term; take tensor components first"))
end

function Base.getindex(::Tensor{D,R}, ::Vararg{STerm,N}) where {D,R,N}
    throw(ArgumentError("tensors can only be component-indexed by SLiterals"))
end

function Base.getindex(expr::SExpr{SFun}, I::Vararg{SLiteral,N}) where {N}
    R = tensorrank(expr)
    R == 0 && return lower_ind(Tensor{N}(expr), I)
    N == R || throw(ArgumentError("expected $R tensor component indices, got $N"))
    return component(expr, I)
end

function Base.getindex(expr::SExpr{SAt}, inds::Vararg{STerm,N}) where {N}
    return lower_loc(Tensor{N}(argument(expr)), location(expr), inds)
end

function Base.getindex(expr::SExpr{SFun}, inds::Vararg{STerm,N}) where {N}
    tensorrank(expr) == 0 || throw(ArgumentError("grid indexing requires a scalar term; take tensor components of '$expr' first"))
    return lower_ind(Tensor{N}(expr), inds)
end

function Base.getindex(expr::SExpr{SComp}, inds::Vararg{STerm,N}) where {N}
    return lower_ind(Tensor{N}(expr), inds)
end

function Base.getindex(expr::SExpr{SFun}, loc::Vararg{Space,N}) where {N}
    tensorrank(expr) == 0 || throw(ArgumentError("location requires a scalar expression; take tensor components of '$expr' first"))
    return locate_scalar(Tensor{N}(expr), loc)
end

function Base.getindex(expr::SExpr{SComp}, loc::Vararg{Space,N}) where {N}
    return locate_scalar(Tensor{N}(expr), loc)
end

component(t::STerm, I::NTuple{N,SLiteral}) where {N} = SExpr(Comp(), t, I...)
Base.@assume_effects :foldable function component(t::SExpr{SFun}, I::NTuple{N,SLiteral}) where {N}
    return component(operation(t), arguments(t), I, t)
end
component(::typeof(+), args::Tuple{Vararg{STerm}}, I, t) = +(map(arg -> arg[I...], args)...)
component(::typeof(-), args::Tuple{STerm}, I, t) = -only(args)[I...]
function component(::typeof(-), args::Tuple{STerm,STerm}, I, t)
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

function component(::typeof(*), args::Tuple{Vararg{DTerm}}, I, t)
    j = tensor_component_arg(args)
    isnothing(j) && return SExpr(SComp(I), t)
    new_args = ntuple(k -> k == j ? args[k][I...] : args[k], Val(length(args)))
    return *(new_args...)
end
function component(::typeof(/), args::Tuple{DTerm,DTerm}, I, t)
    a, b = args
    tensorrank(a) > 0 && tensorrank(b) == 0 && return a[I...] / b
    return SExpr(SComp(I), t)
end
function component(::typeof(//), args::Tuple{DTerm,DTerm}, I, t)
    a, b = args
    tensorrank(a) > 0 && tensorrank(b) == 0 && return a[I...] // b
    return SExpr(SComp(I), t)
end
function component(::typeof(÷), args::Tuple{DTerm,DTerm}, I, t)
    a, b = args
    tensorrank(a) > 0 && tensorrank(b) == 0 && return a[I...] ÷ b
    return SExpr(SComp(I), t)
end

# default rule is to take the component of the whole expression
component(::DTerm, ::Tuple{Vararg{DTerm}}, I, t) = SExpr(SComp(), t, I...)

ispointwise(::DTerm) = false
ispointwise(::SFun) = true

# Immediate symbolic lowering must stay compile-time foldable, otherwise nested
# indexing of scalar expressions regresses to runtime work and type instability.
Base.@assume_effects :foldable function locate_scalar(t::DTerm, locs::NTuple{N,Space}) where {N}
    isuniform(t) && return t
    if iscall(t) && ispointwise(operation(t))
        return SExpr(operation(t), map(x -> x[locs...], arguments(t))...)
    else
        return SExpr(SAt(locs), t)
    end
end

Base.@assume_effects :foldable function lower_loc(t::DTerm, locs::NTuple{N,Space}, inds::NTuple{N,STerm}) where {N}
    isuniform(t) && return t
    (!isexpr(t) || iscomp(t)) && return SExpr(SSub(), SExpr(SAt(locs), t), inds...)
    if iscall(t)
        return evaluate(stencil_rule(operation(t), arguments(t), loc, inds))
    else
        error("malformed static expression $t")
    end
end

Base.@assume_effects :foldable function lower_ind(t::STerm, inds::NTuple{N,STerm}) where {N}
    # Uniform expressions can be substituted directly during compute, so
    # runtime grid indices are irrelevant and must not be threaded further.
    isuniform(t) && return t
    (!isexpr(t) || iscomp(t)) && return SExpr(SSub(), t, inds...)
    if iscall(t)
        return evaluate(stencil_rule(operation(t), arguments(t), inds))
    else
        error("malformed static expression $t")
    end
end
