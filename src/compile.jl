"""
    compile(term, face, binding; inbounds = false)

Compile the lowered `term` of a kernel next to the `face` into a function `f(data, I, n)`, where `data` are the values
of the `binding`, `I` is the grid point of the kernel and `n` the number of cells along each axis. A sample at the
position `p` along an axis reads the index `I[a] + floor(p)` along the spanning axes `:` of the face, `1 + floor(p)`
along its lower sides `↓`, and `n[a] + 1 + floor(p)` along its upper sides `↑`; the entries of `I` along the sides are
not read. Global positions select the binding, e.g. `g[↓, 𝓈]`, and are not indexed. See [`bindings`](@ref).
"""
function compile(term::DTerm, face::Face{N}, b::Binding; inbounds = false) where {N}
    bt = binding_types(b)
    body = Expr(:block, Expr(:meta, :inline))
    if inbounds
        push!(body.args, Expr(:inbounds, true))
    end
    push!(body.args, Expr(:return, toexpr(term, face, bt)))
    expr = Expr(:function, Expr(:tuple, :data, :I, :n), body)
    return drop_expr(@RuntimeGeneratedFunction(expr))
end

function toexpr(expr::DTerm, face, bnd)
    return @match expr begin
        Literal(val) => val
        Index(a) => anchor(face[a], a)
        Tensor(rank, name, _, uniform) => begin
            rank == 0 || throw(ArgumentError("only scalar components are supported for compilation"))
            uniform || throw(ArgumentError("only uniform tensors can be non-indexed during compilation"))
            lookup_uniform(bnd, expr)
        end
        DExpr(Comp(_), (arg,)) => begin
            isuniform(arg) || throw(ArgumentError("only uniform tensors can be non-indexed during compilation"))
            lookup_uniform(bnd, expr)
        end
        DExpr(Inds(positions), (arg,)) => toexpr_inds(arg, positions, face, bnd)
        DExpr(Call(op), args) => toexpr_call(op, args, face, bnd)
        DExpr(Pack(_, _, _), _) => throw(ArgumentError("tensors in component form should be indexed before compilation"))
        ZeroTensor(_) => throw(ArgumentError("zero tensors should be lowered before compilation"))
        IdTensor(_) => throw(ArgumentError("identity tensors should be lowered before compilation"))
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

toexpr_call(op, args, face, bnd) = throw(ArgumentError("only Fun operators can be compiled to Julia code"))
function toexpr_call(op::Fun, args, face, b)
    args_expr = map(x -> toexpr(x, face, b), args)
    # resolve standard arithmetics into symbols for better readability
    f = op == Fun(+) ? :+ :
        op == Fun(-) ? :- :
        op == Fun(*) ? :* :
        op == Fun(/) ? :/ :
        op == Fun(^) ? :^ : op.f
    return Expr(:call, f, args_expr...)
end

function toexpr_inds(arg, positions, face, b)
    length(positions) == length(face) || throw(ArgumentError("samples must have as many positions as the face has axes"))
    key = bindkey(arg, positions)
    idx = findkey(b, key)
    isnothing(idx) && throw(ArgumentError(LazyString("binding does not contain key '", key, "'")))
    data_expr = Expr(:ref, :data, idx)
    b.data[idx] <: AbstractArray || return data_expr
    inds = ntuple(a -> index_expr(positions[a], a, face, b), length(positions))
    return Expr(:ref, data_expr, Iterators.flatten(inds)...)
end

# the index of the grid point of the kernel along the axis `a`, which is the wall point along the sides of the face
anchor(::Colon, a) = :(I[$a])
anchor(::Lower, a) = 1
anchor(::Upper, a) = :(n[$a] + 1)

# the index of the sample along the axis `a`, global positions are not indexed
index_expr(p::Rational{Int}, a, face, b) = (toexpr(offset_term(Index(a), floor(p)), face, b),)
index_expr(::Global, a, face, b) = ()
