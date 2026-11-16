"""
    compile(term, binding; inbounds = false)

Compile `term` into a function `f(data, I, N)`, where `data` are the values of the `binding`, `I` is the
grid point of the kernel and `N` the number of grid points along each axis. A sample at the coordinate
`(frame, offset)` reads the index `o + floor(offset)` along its axis, where the origin `o` of the frame is `I[a]` for
`:`, `1` for `↓`, and `N[a]` for `↑`; traces select the binding, e.g. `g[↓, 𝓈]`, and are not indexed. See
[`bindings`](@ref).
"""
function compile(term::DTerm, b::Binding; inbounds = false)
    bt = binding_types(b)
    body = Expr(:block, Expr(:meta, :inline))
    if inbounds
        push!(body.args, Expr(:inbounds, true))
    end
    push!(body.args, Expr(:return, toexpr(term, bt)))
    expr = Expr(:function, Expr(:tuple, :data, :I, :N), body)
    return drop_expr(@RuntimeGeneratedFunction(expr))
end

function toexpr(expr::DTerm, bnd)
    return @match expr begin
        Literal(val) => val
        Tensor(rank, name, _, uniform) => begin
            rank == 0 || throw(ArgumentError("only scalar components are supported for compilation"))
            uniform || throw(ArgumentError("only uniform tensors can be non-indexed during compilation"))
            lookup_uniform(bnd, expr)
        end
        DExpr(Comp(_), (arg,)) => begin
            isuniform(arg) || throw(ArgumentError("only uniform tensors can be non-indexed during compilation"))
            lookup_uniform(bnd, expr)
        end
        DExpr(Inds(positions), (arg,)) => toexpr_inds(arg, positions, bnd)
        DExpr(Call(op), args) => toexpr_call(op, args, bnd)
        AddMul(_, _, _) => toexpr(callform(expr), bnd)
        DExpr(Pack(_, _, _), _) => throw(ArgumentError("tensors in component form should be indexed before compilation"))
        _ => throw(ArgumentError(LazyString("unsupported expression '", expr, "' for compilation")))
    end
end

function lookup_uniform(b, term)
    haskey(b, term) || throw(ArgumentError(LazyString("binding does not contain key ", term)))
    idx = findkey(b, term)::Int
    typ = b.data[idx]
    if typ <: AbstractArray
        throw(ArgumentError("uniform tensors cannot be bound to arrays"))
    else
        return Expr(:ref, :data, idx)
    end
end

toexpr_call(op, args, bnd) = throw(ArgumentError("only Fun operators can be compiled to Julia code"))
function toexpr_call(op::Fun, args, b)
    args_expr = map(x -> toexpr(x, b), args)
    # resolve standard arithmetics into symbols for better readability
    f = op == Fun(+) ? :+ :
        op == Fun(-) ? :- :
        op == Fun(*) ? :* :
        op == Fun(/) ? :/ :
        op == Fun(^) ? :^ : op.f
    return Expr(:call, f, args_expr...)
end

function toexpr_inds(arg, positions, b)
    # coordinates are the distance from the lower wall, one less than the index of the point
    if iscoordinate(arg)
        a = value(arg)
        return offset_expr(origin_expr(positions[a], a), positions[a].offset - 1)
    end
    key = bindkey(arg, positions)
    idx = findkey(b, key)
    isnothing(idx) && throw(ArgumentError(LazyString("binding does not contain key '", key, "'")))
    data_expr = Expr(:ref, :data, idx)
    b.data[idx] <: AbstractArray || return data_expr
    inds = ntuple(a -> index_expr(positions[a], a), length(positions))
    return Expr(:ref, data_expr, Iterators.flatten(inds)...)
end

# the index of the sample along the axis `a`, traces are not indexed
index_expr(c::Coord, a) = (offset_expr(origin_expr(c, a), floor(Int, c.offset)),)
index_expr(::Trace, a) = ()

# the index of the origin of the frame of a coordinate
origin_expr(c::Coord, a) = origin_expr(c.frame, a)
origin_expr(::Colon, a) = :(I[$a])
origin_expr(::Lower, a) = 1
origin_expr(::Upper, a) = :(N[$a])

offset_expr(o::Integer, k) = stencil_constant(o + k)
function offset_expr(o::Expr, k)
    iszero(k) && return o
    k < 0 && return :($o - $(stencil_constant(-k)))
    return :($o + $(stencil_constant(k)))
end
