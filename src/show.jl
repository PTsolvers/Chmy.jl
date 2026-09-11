function Base.show(io::IO, ::MIME"text/plain", v::DTerm)
    @match v begin
        DExpr(_, _) => print(io, "Expression:\n ", v)
        _ => print(io, v)
    end
end

function Base.show(io::IO, v::DTerm)
    @match v begin
        Literal(val)                  => print(io, val)
        Index(idx)                    => print_index(io, idx)
        Tensor(rank, name, _, _)      => print_tensor(io, rank, name)
        ZeroTensor(_)                 => print(io, '𝒪')
        IdTensor(_)                   => print(io, 'ℐ')
        DExpr(Call(op), args)         => print_call(io, op, args)
        DExpr(Comp(comp), (arg,))     => print_indexed(io, arg, comp)
        DExpr(Locs(locs), (arg,))     => print_indexed(io, arg, locs)
        DExpr(Inds(), (arg, inds...)) => print_indexed(io, arg, inds)
    end
end

function print_indexed(io, arg, inds)
    parens = !isnothing(printed_operator(arg))
    parens && print(io, '(')
    print(io, arg)
    parens && print(io, ')')
    return print_list(io, '[', inds, ']')
end

function print_subscript(io, i::Integer)
    i < 0 && throw(ArgumentError("subscript must be nonnegative"))
    i >= 10 && print_subscript(io, div(i, 10))
    print(io, '₀' + Int(mod(i, 10)))
    return
end

function print_index(io, i)
    if i == 1
        print(io, '𝑖')
    elseif i == 2
        print(io, '𝑗')
    elseif i == 3
        print(io, '𝑘')
    else
        print(io, "Index(", i, ')')
    end
end

function print_tensor(io, rank, name)
    if rank > 0
        printstyled(io, name; bold=true, underline=true)
    else
        print(io, name)
    end
    return
end

"""
    print_opname(io, op)

Print the name used for `op` in expression calls to `io`. Specialize this method
for custom operators to control their call names. The default uses `show(io, op)`.
"""
print_opname(io, op::Operator) = show(io, op)
print_opname(io, op::Fun) = print(io, nameof(op.f))
print_opname(io, ::AbstractDerivative) = print(io, '∂')
print_opname(io, ::Gradient) = print(io, "grad")
print_opname(io, ::Divergence) = print(io, "divg")
print_opname(io, ::Curl) = print(io, "curl")
function print_opname(io, op::Lifted)
    print_opname(io, op.op)
    print_subscript(io, op.axis)
    return
end

function print_call(io, op::Operator, args)
    print_opname(io, op)
    return print_list(io, '(', args, ')')
end

function print_call(io, op::Fun, args)
    opname = nameof(op.f)

    if !Base.isoperator(opname)
        print_opname(io, op)
        return print_list(io, '(', args, ')')
    end

    if length(args) == 1 && Base.isunaryoperator(opname)
        print_opname(io, op)
        arg = only(args)
        parens = need_parens(arg)
        parens && print(io, '(')
        print(io, arg)
        parens && print(io, ')')
        return
    end

    if length(args) < 2 || !Base.isbinaryoperator(opname)
        print(io, '(')
        print_opname(io, op)
        print(io, ')')
        return print_list(io, '(', args, ')')
    end

    for (i, arg) in enumerate(args)
        omit_operator = i > 1 && omit_mul(args[i - 1], arg, opname)
        if i > 1 && !omit_operator
            print(io, ' ')
            print_opname(io, op)
            print(io, ' ')
        end

        parens = need_parens(arg, opname, i)
        if i < length(args) &&
           omit_mul(arg, args[i + 1], opname) &&
           coef_needs_parens(arg)
            parens = true
        end

        parens && print(io, '(')
        print(io, arg)
        parens && print(io, ')')
    end
    return
end

function omit_mul(left, right, op)
    op === :* || return false
    isreal(left) || return false
    @match right begin
        Tensor(_, name, _, _) => Base.isidentifier(name)
        _ => false
    end
end

function coef_needs_parens(term)
    @match term begin
        Literal(val) => !(val isa Integer && !(val isa Bool)) &&
                        !(val isa AbstractFloat && isfinite(val))
        _ => false
    end
end

function printed_operator(term)
    @match term begin
        DExpr(Call(op), args) => begin
            op isa Fun || return nothing
            opname = nameof(op.f)
            if !Base.isoperator(opname)
                nothing
            elseif length(args) == 1 && Base.isunaryoperator(opname)
                (opname, Base.operator_precedence(:^), true)
            elseif length(args) >= 2 && Base.isbinaryoperator(opname)
                (opname, Base.operator_precedence(opname), false)
            else
                nothing
            end
        end
        Literal(val) => begin
            if val isa Rational
                (Symbol("//"), Base.operator_precedence(Symbol("//")), false)
            elseif val isa Real && signbit(val)
                (:-, Base.operator_precedence(:^), true)
            else
                nothing
            end
        end
        _ => nothing
    end
end

# unary operator needs parentheses
function need_parens(arg)
    child = printed_operator(arg)
    isnothing(child) && return false

    _, precedence, unary = child
    return unary || precedence < Base.operator_precedence(:^)
end

# general operator needs parentheses
function need_parens(arg, op, position)
    child = printed_operator(arg)
    isnothing(child) && return false

    child_op, child_precedence, _ = child
    precedence = Base.operator_precedence(op)
    child_precedence < precedence && return true
    child_precedence > precedence && return false

    if child_op === op && op in (:+, :++, :*)
        return false
    elseif precedence == Base.operator_precedence(:>)
        return true
    elseif Base.operator_associativity(op) === :right
        return position == 1
    else
        return position > 1
    end
end

function print_list(io, b, list, e)
    print(io, b)
    join(io, list, ", ")
    print(io, e)
end

Base.show(io::IO, ::Segment) = print(io, "𝓈")
Base.show(io::IO, ::Point) = print(io, "𝓅")

function Base.show(io::IO, ::MIME"text/plain", t::TensorComponents)
    print(io, "$(t.dims)-D ")
    if t.kind == Kind.Sym()
        print(io, "symmetric ")
    elseif t.kind == Kind.Alt()
        print(io, "alternating ")
    elseif t.kind == Kind.Diag()
        print(io, "diagonal ")
    end
    println(io, "Tensor components:")
    foreach_component(t.kind, t.dims, t.rank) do I
        print(io, ' ')
        print_list(io, '[', I, ']')
        println(io, " => ", t[Tuple(I)...])
    end
    return
end

blank(width::Integer) = repeat(" ", max(0, width))

function Base.show(io::IO, ::MIME"text/plain", b::Binding)
    padlength = maximum(f -> textwidth(string(f)), b.keys; init=0)
    if !haskey(io, :compact)
        io = IOContext(io, :compact => true)
    end
    print(io, "Binding:")
    for (key, value) in zip(b.keys, b.data)
        rendered = string(key)
        print(io, '\n', ' ')
        print(io, rendered)
        print(io, blank(padlength - textwidth(rendered)))
        print(io, " => ")
        show(io, value)
    end
end
