# leaf terms
termrank(::SIndex)      = 0x0
termrank(::STensor)     = 0x1
termrank(::SZeroTensor) = 0x2
termrank(::SIdTensor)   = 0x3
termrank(::SLiteral)    = 0x4
termrank(::SExpr)       = 0x5
termrank(t::STerm)      = 0x6 + objectid(t)

# operator terms
oprank(::SRef)                    = 0x0
oprank(::SFun)                    = 0x1
oprank(::AbstractDerivative)      = 0x2
oprank(::LiftedPartialDerivative) = 0x3
oprank(::Gradient)                = 0x4
oprank(::Divergence)              = 0x5
oprank(::Curl)                    = 0x6
oprank(t::STerm)                  = 0x7 + objectid(t)

# comparing heads of expressions
headrank(::Call)       = 0x0
headrank(::Comp)       = 0x1
headrank(::Ind)        = 0x2
headrank(::Loc)        = 0x3
headrank(::Node)       = 0x4
headrank(h::SExprHead) = 0x5 + objectid(h)
headrank(expr::SExpr)  = headrank(head(expr))

# comparing STerms lexicographically
isless_lex(::SIndex{I}, ::SIndex{J}) where {I,J} = isless(I, J)
isless_lex(::SRef{F1}, ::SRef{F2}) where {F1,F2} = isless(F1, F2)
isless_lex(x::SFun, y::SFun) = isless(nameof(x.f), nameof(y.f))
isless_lex(::Point, ::Segment) = true
isless_lex(::Segment, ::Point) = false

function isless_lex(x::AbstractPartialDerivative{I}, y::AbstractPartialDerivative{J}) where {I,J}
    x === y && return false
    x.op === y.op || return isless_lex(x.op, y.op)
    return isless(I, J)
end

isless_lex(x::AbstractDerivative, y::AbstractDerivative) = isless(objectid(x), objectid(y))

function isless_lex(x::STensor, y::STensor)
    # higher-rank tensors are larger
    rx = tensorrank(x)
    ry = tensorrank(y)
    rx == ry || return isless(rx, ry)
    # compare names lexicographically
    nx = name(x)
    ny = name(y)
    nx == ny || return isless(nx, ny)
    # Chmy requires tensors of the same name and rank to agree in all static metadata.
    if x !== y
        throw(ArgumentError("tensors with the same name must have the same rank, kind, and uniformity"))
    end
    return false
end

function isless_lex(x::STerm, y::STerm)
    x === y && return false
    # compare tensor ranks
    tx = tensorrank(x)
    ty = tensorrank(y)
    tx == ty || return isless(tx, ty)
    # compare static terms at compile time
    if isstatic(x) && isstatic(y)
        return isless(compute(x), compute(y))
    end
    # different kinds of terms are compared by their term rank
    rx = termrank(x)
    ry = termrank(y)
    rx == ry || return rx < ry
    # special logic for comparing expressions
    if isexpr(x) && isexpr(y)
        return isless_expr(x, y)
    end
    # catchall for user-defined objects
    return isless(objectid(x), objectid(y))
end

function isless_expr(x::SExpr, y::SExpr)
    # compare different kinds of expressions by rank of the head
    hx = headrank(x)
    hy = headrank(y)
    hx == hy || return hx < hy
    # compare children lexicographically
    return isless_tuple(children(x), children(y))
end

# special logic for comparing call expressions
function isless_expr(x::SExpr{Call}, y::SExpr{Call})
    # compare operations
    opx = operation(x)
    opy = operation(y)
    if opx !== opy
        # first different kind of operations compare differently
        orx = oprank(opx)
        ory = oprank(opy)
        orx == ory || return isless(orx, ory)
        # otherwise they should be comparable
        return isless_lex(opx, opy)
    end
    # finally lexicographically compare argument list
    return isless_tuple(arguments(x), arguments(y))
end

# compare static tuples by comparing first arguments and if those are equal comparing tails
isless_tuple(::Tuple{}, ::Tuple{}) = false
isless_tuple(::Tuple{}, ::Tuple{Any,Vararg}) = true
isless_tuple(::Tuple{Any,Vararg}, ::Tuple{}) = false

function isless_tuple(xs::Tuple{X,Vararg}, ys::Tuple{Y,Vararg}) where {X,Y}
    xh = first(xs)
    yh = first(ys)
    xh === yh && return isless_tuple(Base.tail(xs), Base.tail(ys))
    return isless_lex(xh, yh)
end

# static sorting of tuples of singleton types
ssort_impl(args::Tuple, lt, by, order) = (sort!(collect(args); lt, by, order)...,)

@generated function ssort(args::Tuple; lt=Base.isless, by=Base.identity, order=Base.Order.Forward)
    sorted = ssort_impl(args.instance, lt.instance, by.instance, order.instance)
    return :($sorted)
end
