"""
    compile(expr, binding; inbounds=false)

Compile a Chmy expression into a callable Julia function.
"""
function compile(term::DTerm, b::Binding; inbounds = false)
    bt = binding_types(b)
    body = Expr(:block, Expr(:meta, :inline))
    if inbounds
        push!(body.args, Expr(:inbounds, true))
    end
    push!(body.args, Expr(:return, toexpr(term, bt)))
    expr = Expr(:function, Expr(:tuple, :data, :I), body)
    return drop_expr(@RuntimeGeneratedFunction(expr))
end

function toexpr(expr::DTerm, bnd)
    return @match expr begin
        Literal(val) => val
        Index(i) => Expr(:ref, :I, i)
        Tensor(rank, name, _, uniform) => begin
            rank == 0 || throw(ArgumentError("only scalar components are supported for compilation"))
            uniform || throw(ArgumentError("only uniform tensors can be non-indexed during compilation"))
            lookup_uniform(bnd, expr)
        end
        DExpr(Comp(_), (arg,)) => begin
            isuniform(arg) || throw(ArgumentError("only uniform tensors can be non-indexed during compilation"))
            lookup_uniform(bnd, expr)
        end
        DExpr(Inds(offsets), (arg,)) => toexpr_inds(arg, offsets, bnd)
        DExpr(Call(op), args) => toexpr_call(op, args, bnd)
        DExpr(Locs(_), _) => throw(ArgumentError("only uniform tensors can be non-indexed during compilation"))
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

function toexpr_inds(arg, offsets, b)
    haskey(b, arg) || throw(ArgumentError(LazyString("binding does not contain key '", arg, "'")))
    idx = findkey(b, arg)::Int
    typ = b.data[idx]
    data_expr = Expr(:ref, :data, idx)
    if typ <: AbstractArray
        return Expr(:ref, data_expr, ntuple(a -> index_expr(a, offsets[a]), length(offsets))...)
    else
        return data_expr
    end
end

function index_expr(a, offset)
    isinteger(offset) || throw(ArgumentError("only lowered samples with integer offsets can be compiled"))
    k = Int(offset)
    return iszero(k) ? :(I[$a]) : k > 0 ? :(I[$a] + $k) : :(I[$a] - $(-k))
end
