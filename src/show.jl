function Base.show(io::IO, ::MIME"text/plain", v::DTerm)
    return @match v begin
        DExpr(Pack(rank, dims, kind), _) => print_pack_table(io, v, rank, dims, kind)
        DExpr(_, _) => print(io, "Expression:\n ", v)
        _ => print(io, v)
    end
end

function Base.show(io::IO, v::DTerm)
    return @match v begin
        Literal(val) => print(io, val)
        Index(i) => print_index(io, i)
        Coordinate(i) => print_coordinate(io, i)
        Tensor(rank, name, _, _) => print_tensor(io, rank, name)
        ZeroTensor(_) => print(io, '𝒪')
        IdTensor(_) => print(io, 'ℐ')
        DExpr(Call(op), args) => printop(io, op, args)
        DExpr(Comp(comp), (arg,)) => print_indexed(io, arg, comp)
        DExpr(Locs(locs), (arg,)) => print_indexed(io, arg, locs)
        DExpr(Inds(offsets), (arg,)) => print_indexed(io, arg, map(stencil_constant, offsets))
        DExpr(Pack(_, dims, _), _) => print_pack(io, v, dims, ())
    end
end

# nested lists of all components, the outer list runs over the first index
function print_pack(io, t, dims, I)
    length(I) == tensorrank(t) && return print(io, t[I...])
    print(io, '[')
    for i in 1:dims
        i > 1 && print(io, ", ")
        print_pack(io, t, dims, (I..., i))
    end
    return print(io, ']')
end

function print_indexed(io, arg, inds)
    # indexing binds tighter than any operator
    print_operand(io, arg, typemax(Int))
    return print_list(io, '[', inds, ']')
end

function print_subscript(io, i::Integer)
    i < 0 && throw(ArgumentError("subscript must be nonnegative"))
    i >= 10 && print_subscript(io, div(i, 10))
    print(io, '₀' + Int(mod(i, 10)))
    return
end

# in Unicode, superscript numbers have non-consecutive code points
function print_superscript(io, i::Integer)
    i < 0 && throw(ArgumentError("subscript must be nonnegative"))
    i >= 10 && print_superscript(io, div(i, 10))
    i = mod(i, 10)
    if i == 0
        print(io, '⁰')
    elseif i == 1
        print(io, '¹')
    elseif i == 2
        print(io, '²')
    elseif i == 3
        print(io, '³')
    elseif i == 4
        print(io, '⁴')
    elseif i == 5
        print(io, '⁵')
    elseif i == 6
        print(io, '⁶')
    elseif i == 7
        print(io, '⁷')
    elseif i == 8
        print(io, '⁸')
    elseif i == 9
        print(io, '⁹')
    end
    return
end

function print_index(io, i)
    return if i == 1
        print(io, '𝑖')
    elseif i == 2
        print(io, '𝑗')
    elseif i == 3
        print(io, '𝑘')
    else
        print(io, "Index(", i, ')')
    end
end

function print_coordinate(io, i)
    printstyled(io, 'ξ'; italic = true)
    return print_superscript(io, i)
end

function print_tensor(io, rank, name)
    if rank > 0
        printstyled(io, name; bold = true, underline = true)
    else
        print(io, name)
    end
    return
end

print_opname(io, op::Operator) = show(io, op)
print_opname(io, op::Fun) = print(io, nameof(op.f))
function print_opname(io, op::BroadcastedFun)
    name = nameof(op.f)
    return Base.isoperator(name) ? print(io, '.', name) : print(io, name, '.')
end
print_opname(io, ::AbstractDifference) = print(io, '∂')
print_opname(io, ::Gradient) = print(io, "grad")
print_opname(io, ::Divergence) = print(io, "divg")
print_opname(io, ::Curl) = print(io, "curl")
function print_opname(io, op::Lifted)
    print_opname(io, op.op)
    foreach(op.axes) do a
        a < 0 && print(io, '₋')
        print_subscript(io, abs(a))
    end
    return
end

"""
    printop(io, op, args)

Print the call of the operator `op` with the arguments `args` to `io`. Specialize this method together with
[`printprec`](@ref) to control how calls of custom operators are displayed. The default prints the operator name
followed by the parenthesized list of arguments.
"""
function printop(io, op::Operator, args)
    print_opname(io, op)
    return print_list(io, '(', args, ')')
end

"""
    printprec(op, args)

The precedence of the call of `op` with `args` as printed by [`printop`](@ref), or `0` if the call is printed in a
self-delimiting form, e.g. as a function call. An operand is parenthesized if its precedence is positive and not
higher than the precedence of the enclosing operator.
"""
printprec(::Operator, args) = 0

# unary operators bind as tight as `^`, so that `-a * b` needs no parentheses, but `-(a ^ 2)` does
const UNARY_PREC = Base.operator_precedence(:^)

printprec(term::DTerm) = @match term begin
    DExpr(Call(op), args) => printprec(op, args)
    Literal(val) => literalprec(val)
    _ => 0
end

literalprec(val) = 0
literalprec(::Rational) = Base.operator_precedence(Symbol("//"))
literalprec(val::Real) = signbit(val) ? UNARY_PREC : 0

# parenthesize the operand if it doesn't bind tighter than the enclosing operator
function print_operand(io, arg, prec)
    parens = 0 < printprec(arg) <= prec
    parens && print(io, '(')
    print(io, arg)
    parens && print(io, ')')
    return
end

printed_opname(op::Fun) = nameof(op.f)
function printed_opname(op::BroadcastedFun)
    name = nameof(op.f)
    return Base.isoperator(name) ? Symbol('.', name) : name
end

function printprec(op::Union{Fun, BroadcastedFun}, args)
    opname = printed_opname(op)
    Base.isoperator(opname) || return 0
    length(args) == 1 && Base.isunaryoperator(opname) && return UNARY_PREC
    length(args) >= 2 && Base.isbinaryoperator(opname) && return Base.operator_precedence(opname)
    return 0
end

function printop(io, op::Union{Fun, BroadcastedFun}, args)
    prec = printprec(op, args)
    if prec == 0
        # operators that can't be printed in operator form are printed as `(op)(args...)`
        parens = Base.isoperator(printed_opname(op))
        parens && print(io, '(')
        print_opname(io, op)
        parens && print(io, ')')
        return print_list(io, '(', args, ')')
    end
    if length(args) == 1
        print_opname(io, op)
        return print_operand(io, only(args), prec)
    end
    for (i, arg) in enumerate(args)
        if i > 1
            print(io, ' ')
            print_opname(io, op)
            print(io, ' ')
        end
        print_operand(io, arg, prec)
    end
    return
end

function print_list(io, b, list, e)
    print(io, b)
    join(io, list, ", ")
    return print(io, e)
end

Base.show(io::IO, ::Segment) = print(io, "𝓈")
Base.show(io::IO, ::Point) = print(io, "𝓅")

function print_pack_table(io, t, rank, dims, kind)
    print(io, dims, "-D ")
    if kind == Kind.Sym()
        print(io, "symmetric ")
    elseif kind == Kind.Alt()
        print(io, "alternating ")
    elseif kind == Kind.Diag()
        print(io, "diagonal ")
    end
    print(io, "tensor components:")
    foreach_component(kind, dims, rank) do I
        print(io, "\n ")
        print_list(io, '[', I, ']')
        print(io, " => ", t[I...])
    end
    return
end

blank(width::Integer) = repeat(" ", max(0, width))

function Base.show(io::IO, ::MIME"text/plain", b::Binding)
    padlength = maximum(f -> textwidth(string(f)), b.keys; init = 0)
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
    return
end
