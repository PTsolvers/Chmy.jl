function to_subscript(::Val{i}) where {i}
    if i <= 9
        return Symbol('₀' + i)
    else
        # recurse for multi-digit numbers
        return Symbol(to_subscript(Val(div(i, 10))), to_subscript(Val(mod(i, 10))))
    end
end

variablename(::SUniform{Value}) where {Value} = Value
variablename(sf::SFun) = nameof(sf.f)
variablename(::SRef{F}) where {F} = F
variablename(s::Symbol) = s
variablename(::Point) = :ᵖ
variablename(::Segment) = :ˢ
variablename(::AbstractDerivative) = :𝒟

function variablename(::SIndex{I}) where {I}
    if I isa Symbol
        return I
    else
        return Symbol(:i, to_subscript(Val(I)))
    end
end

function variablename(::AbstractPartialDerivative{I}) where {I}
    return Symbol("∂", to_subscript(Val(I)))
end

Base.show(io::IO, term::STerm) = show_static(io, term, 0)

function Base.show(io::IO, ::MIME"text/plain", term::STerm)
    if isexpr(term)
        print(io, "StaticExpression:\n ")
    end
    show_static(io, term, 0)
end

function show_static(io, ::Tag{name,inds}, ::Int) where {name,inds}
    printstyled(io, name; bold=true)
    if length(inds) > 0
        print(io, '[')
        join(io, inds, ", ")
        print(io, ']')
    end
end

show_static(io, term::STerm, ::Int) = print(io, variablename(term))

show_static(io, si::SIndex, ::Int) = printstyled(io, variablename(si); italic=true)

function show_static(io, expr::SExpr, prec::Int)
    if iscall(expr)
        op = operation(expr)
        args = arguments(expr)
        opname = variablename(op)
        op_prec = Base.operator_precedence(opname)
        if Meta.isoperator(opname)
            if length(args) == 1
                parens = needs_parens(expr, prec)
                parens && print(io, '(')
                print(io, opname)
                show_static(io, args[1], op_prec)
                parens && print(io, ')')
            else
                if opname == :* &&
                   length(args) == 2 &&
                   args[1] isa SUniform &&
                   !isexpr(args[2]) &&
                   variablename(args[1]) isa Union{AbstractFloat,Integer}
                    sep = ""
                else
                    sep = " $opname "
                end
                parens = needs_parens(expr, prec)
                parens && print(io, '(')
                show_list(io, args, sep, op_prec)
                parens && print(io, ')')
            end
        else
            print(io, opname)
            print(io, '(')
            show_list(io, args, ", ", op_prec)
            print(io, ')')
        end
    elseif isind(expr)
        arg = argument(expr)
        if !isexpr(arg) || isloc(arg)
            show_static(io, arg, 0)
        else
            print(io, '(')
            show_static(io, arg, 0)
            print(io, ')')
        end
        print(io, '[')
        show_list(io, indices(expr), ", ", 0)
        print(io, ']')
    elseif isloc(expr)
        arg = argument(expr)
        if isexpr(arg)
            print(io, '(')
            show_static(io, arg, 0)
            print(io, ')')
        else
            show_static(io, arg, 0)
        end
        for s in location(expr)
            print(io, variablename(s))
        end
    end
    return
end

function show_list(io, items, sep, prec)
    for (i, item) in enumerate(items)
        i > 1 && print(io, sep)
        show_static(io, item, prec)
    end
    return
end

function needs_parens(expr, prec)
    iscall(expr) || return false
    op = operation(expr)
    op_prec = Base.operator_precedence(variablename(op))
    if op_prec == 0 || prec == 0
        return false
    end
    return op_prec <= prec
end

function Base.show(io::IO, ::MIME"text/plain", v::Vec{N}) where {N}
    print(io, "$N-element Vec:")
    for i in 1:N
        print(io, '\n', ' ')
        show(io, v.components[i])
    end
end

function Base.show(io::IO, ::MIME"text/plain", t::AbstractTensor{O,D}) where {O,D}
    join(io, ntuple(_ -> D, Val(O)), '×')
    sym = symmetry(t)
    if sym <: IdentityGroup
        print(io, " AsymmetricTensor:")
    elseif sym <: SymmetricGroup
        print(io, " SymmetricTensor:")
    else
        error("unknown tensor symmetry")
    end
    for idx in CartesianIndices(ntuple(_ -> D, O))
        inds = Tuple(idx)
        if sym <: SymmetricGroup && !issorted(inds)
            continue
        end
        print(io, '\n', ' ', '[')
        join(io, inds, ", ")
        print(io, "] => ")
        show(io, t[inds...])
    end
end

function Base.show(io::IO, ::MIME"text/plain", b::Binding)
    padlength = maximum(f -> textwidth(string(f)), b.exprs; init=0)
    if !haskey(io, :compact)
        io = IOContext(io, :compact => true)
    end
    print(io, "Binding:")
    for (expr, value) in zip(b.exprs, b.data)
        print(io, '\n', ' ')
        print(io, rpad(string(expr), padlength))
        print(io, " => ")
        show(io, value)
    end
end
