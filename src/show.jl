function Base.show(io::IO, ::MIME"text/plain", v::DTerm)
    return @match v begin
        DExpr(Pack(rank, dims, kind), _) => print_pack_table(io, v, rank, dims, kind)
        DExpr(_, _) || AddMul(_, _, _) => print(io, "Expression:\n ", v)
        _ => print(io, v)
    end
end

function Base.show(io::IO, v::DTerm)
    return @match v begin
        Literal(val) => print(io, val)
        Coordinate(i) => print_coordinate(io, i)
        Tensor(rank, name, _, _) => print_tensor(io, rank, name)
        DExpr(Call(op), args) => printop(io, op, args)
        DExpr(Comp(comp), (arg,)) => print_indexed(io, arg, comp)
        DExpr(Inds(positions), (arg,)) => print_indexed(io, arg, positions)
        DExpr(Pack(_, dims, _), _) => print_pack(io, v, dims, ())
        AddMul(_, _, _) => print(io, callform(v))
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
function print_opname(io, d::Interpolate)
    print(io, '∂')
    d.order == 1 || print_superscript(io, d.order)
    return
end
print_opname(io, ::GridInterpolate) = print(io, "interpolate")
print_opname(io, ::Gradient) = print(io, "grad")
print_opname(io, ::Divergence) = print(io, "divg")
print_opname(io, ::Curl) = print(io, "curl")
function print_opname(io, op::Lifted)
    print_opname(io, op.op)
    foreach(a -> print_subscript(io, a), op.axes)
    return
end
print_opname(io, p::Project) = print(io, "Project(", LocationName(p.element), ", ", p.axis, ')')

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

printop(io, ::ZeroTensor, args) = print(io, '𝒪')
printop(io, ::IdTensor, args) = print(io, 'ℐ')

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
    AddMul(_, _, _) => printprec(callform(term))
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
        # arithmetic operators are left-associative, so the first operand of the same precedence needs no parentheses
        print_operand(io, arg, i == 1 && isleftarithmetic(prec) ? prec - 1 : prec)
    end
    return
end

isleftarithmetic(prec) = prec == Base.operator_precedence(:+) || prec == Base.operator_precedence(:*)

function print_list(io, b, list, e)
    print(io, b)
    join(io, list, ", ")
    return print(io, e)
end

Base.show(io::IO, ::Segment) = print(io, "𝓈")
Base.show(io::IO, ::Point) = print(io, "𝓅")
Base.show(io::IO, ::Lower) = print(io, "↓")
Base.show(io::IO, ::Upper) = print(io, "↑")

# locations in a location list, where `:` is printed without piracy of `show(::Colon)`
struct LocationName{L}
    loc::L
end
Base.show(io::IO, l::LocationName) = show(io, l.loc)
Base.show(io::IO, ::LocationName{Colon}) = print(io, ':')

Base.show(io::IO, l::Located) = print_indexed(io, l.term, map(LocationName, l.locations))

# positions print as they are written in samples, `(↓, 1//2)` and `↓`, and coordinates from the origin as offsets, `1//2`
Base.show(io::IO, c::Coord) = print_coord(io, c.frame, c.offset)
print_coord(io, frame, offset) = print(io, '(', LocationName(frame), ", ", stencil_constant(offset), ')')
print_coord(io, ::Colon, offset) = print(io, stencil_constant(offset))
Base.show(io::IO, t::Trace) = show(io, LocationName(t.element))

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

# stencils

function Base.show(io::IO, nu::Nonuniforms)
    print_list(io, "Nonuniforms(", keys(nu.samples), ')')
    print(io, " at ")
    return print_list(io, '[', nu.position, ']')
end

# stencils are drawn in up to three dimensions
Base.show(io::IO, ::MIME"text/plain", nu::Nonuniforms) = show(io, nu)

function Base.show(io::IO, ::MIME"text/plain", nu::Union{Nonuniforms{1}, Nonuniforms{2}, Nonuniforms{3}})
    x = nu.position
    N = length(x)
    o = placed(x, x)
    # the stencil of each field, preceded by the stencil of all fields if there are several
    rows = [(string(var), [(var, p) for p in ps]) for (var, ps) in nu.samples]
    length(rows) == 1 || pushfirst!(rows, ("all", [(var, p) for (var, ps) in nu.samples for p in ps]))
    stencils = [split_placed(samples, x) for (_, samples) in rows]
    # the slices along the third axis with samples of any field or the output position
    zs = sort!(unique!([last(o); [last(p) for (active, _) in stencils for p in active]]))
    data = Matrix{Any}(undef, length(rows), length(zs) + 1)
    for (i, ((label, _), (active, unplaced))) in enumerate(zip(rows, stencils))
        bounds = ntuple(a -> grid_bounds([p[a] for p in [collect(active); o]]), 2)
        cells = [slice_lines(active, o, bounds, map(outward ∘ frame, x), z) for z in zs]
        append!(cells[1], unplaced)
        height = maximum(length, cells)
        top = (height - 1) ÷ 2
        data[i, 1] = join([fill("", top); label; fill("", height - top - 1)], '\n')
        data[i, 2:end] = [join(cell, '\n') for cell in cells]
    end
    labels = N == 3 ? [string(AXIS_LABELS[3], " = ", halfstep_tick(z)) for z in zs] : ["stencil"]
    print(io, "Nonuniforms at ")
    print_list(io, '[', x, ']')
    println(io, ':')
    return pretty_table(
        io, data;
        column_labels = ["variable"; labels],
        alignment = :c,
        line_breaks = true,
        new_line_at_end = false,
        fit_table_in_display_horizontally = false,
        fit_table_in_display_vertically = false,
        table_format = TextTableFormat(; horizontal_lines_at_data_rows = :all),
    )
end

const AXIS_LABELS = ('𝑖', '𝑗', '𝑘')

# the offset in half steps of the sample position `p` in the frame of the output position `x` along an axis, or
# `nothing` if the sample cannot be placed relative to the output position
halfsteps(p::Coord, x::Coord) = p.frame === x.frame ? Int(2 * p.offset) : nothing
halfsteps(p::Trace, x::Coord) = wall_halfsteps(p.element, x.frame)
halfsteps(p::Trace, x::Trace) = p.element === x.element ? 0 : nothing
halfsteps(::Coord, ::Trace) = nothing

# a wall is the origin of its frame
wall_halfsteps(::S, ::S) where {S <: Side} = 0
wall_halfsteps(element, frame) = nothing

# the offsets in half steps of the sample positions `p` along three axes, or `nothing` if the sample cannot be placed
function placed(p::NTuple{N, Position}, x::NTuple{N, Position}) where {N}
    h = map(halfsteps, p, x)
    any(isnothing, h) && return nothing
    return ntuple(a -> a <= N ? h[a]::Int : 0, 3)
end

# the placed offsets of the samples, and the lines listing the samples that cannot be placed
function split_placed(samples, x)
    active = Set{NTuple{3, Int}}()
    unplaced = Base.AnnotatedString{String}[]
    for (var, p) in samples
        h = placed(p, x)
        isnothing(h) ? push!(unplaced, Base.AnnotatedString(sprint(show, makeinds_unchecked(var, p)))) : push!(active, h)
    end
    return active, unplaced
end

# the bounds in half steps of the grid lines encompassing the offsets
grid_bounds(h) = (2 * fld(minimum(h), 2), 2 * cld(maximum(h), 2))

# the label of an offset in half steps, e.g. `-1½`
function halfstep_tick(h)
    iseven(h) && return string(h ÷ 2)
    n = abs(h) ÷ 2
    return string(h < 0 ? "-" : "", n == 0 ? "" : string(n), '½')
end

# the location of the offsets in half steps on the grid of a slice: points, segments along each axis, and cells,
# filled if sampled and empty otherwise
function stencil_glyph(hx, hy, sampled)
    iseven(hx) && iseven(hy) && return sampled ? '•' : '◦'
    iseven(hx) && return sampled ? '▸' : '▹'
    iseven(hy) && return sampled ? '▴' : '▵'
    return sampled ? '▪' : '▫'
end

# the faces of a location: the output position, the samples, and the inactive locations
function stencil_faces(isoutput, sampled)
    isoutput && return sampled ? (:magenta, :bold) : (:magenta,)
    return sampled ? (:cyan, :bold) : STENCIL_GRID
end

const STENCIL_GRID = (:shadow,)

# whether the offset `m` in quarter steps lies beyond the wall with the outward direction `s`
isbeyond(s, m) = s * m > 0

# the lines on a wall are bold, and the lines in the ghost region are dashed
grid_line(solid, dashed, ghost, wall) = (ghost ? dashed : solid, wall ? (:bold,) : STENCIL_GRID)

# a slice is drawn on a canvas of characters with their faces, a half step spans four columns and two rows, and the
# second axis points upwards; `walls` are the outward directions of the walls of the frames of the output position
function slice_lines(active, o, ((xlo, xhi), (ylo, yhi)), walls, z)
    N = length(walls)
    sx, sy = walls[1], get(walls, 2, 0)
    xticks = map(halfstep_tick, xlo:xhi)
    yticks = N >= 2 ? map(halfstep_tick, ylo:yhi) : String[]
    # the tick labels of the second axis and a space, or the overhang of the first tick label of the first axis
    prefix = max(maximum(textwidth, yticks; init = -1) + 1, textwidth(first(xticks)) - 1)
    col(hx) = prefix + 1 + 4 * (hx - xlo)
    row(hy) = 2 + 2 * (yhi - hy)
    canvas = Matrix{Tuple{Char, Tuple{Vararg{Symbol}}}}(undef, row(ylo) + 1, col(xhi) + (N >= 2 ? 2 : 0))
    fill!(canvas, (' ', ()))
    canvas[1, (col(xlo) + col(xhi)) ÷ 2] = (AXIS_LABELS[1], ())
    for hy in ylo:yhi, hx in xlo:xhi
        r, c = row(hy), col(hx)
        # the pieces of grid lines are in the ghost region if their middle is beyond a wall
        if iseven(hy) && hx < xhi
            ghost = isbeyond(sx, 2hx + 1) || isbeyond(sy, 2hy)
            canvas[r, (c + 1):(c + 3)] .= Ref(grid_line('─', '╌', ghost, sy != 0 && hy == 0))
        end
        if iseven(hx) && hy < yhi
            ghost = isbeyond(sx, 2hx) || isbeyond(sy, 2hy + 1)
            canvas[r - 1, c] = grid_line('│', '╎', ghost, sx != 0 && hx == 0)
        end
        sampled = (hx, hy, z) in active
        canvas[r, c] = (stencil_glyph(hx, hy, sampled), stencil_faces((hx, hy, z) == o, sampled))
    end
    for (hx, tick) in zip(xlo:xhi, xticks)
        put_text!(canvas, size(canvas, 1), col(hx) - textwidth(tick) + 1, tick)
    end
    for (hy, tick) in zip(ylo:yhi, yticks)
        put_text!(canvas, row(hy), prefix - textwidth(tick), tick)
    end
    if N >= 2
        canvas[row(yhi - (yhi - ylo) ÷ 2), col(xhi) + 2] = (AXIS_LABELS[2], ())
    end
    return map(canvas_line, eachrow(canvas))
end

put_text!(canvas, r, c, text) = canvas[r, c:(c + length(text) - 1)] = [(char, ()) for char in text]

# runs of characters with the same faces are annotated together
function canvas_line(cells)
    runs = Base.AnnotatedString{String}[]
    i = firstindex(cells)
    while i <= lastindex(cells)
        faces = last(cells[i])
        j = i
        while j < lastindex(cells) && last(cells[j + 1]) === faces
            j += 1
        end
        text = String(map(first, cells[i:j]))
        push!(runs, Base.AnnotatedString(text, [(region = 1:ncodeunits(text), label = :face, value = f) for f in faces]))
        i = j + 1
    end
    return Base.annotatedstring(runs...)
end
