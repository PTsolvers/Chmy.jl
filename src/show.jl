# Styled string primitives and faces

function to_subscript(::Val{i}) where {i}
    if i <= 9
        return Symbol('₀' + i)
    else
        return Symbol(to_subscript(Val(div(i, 10))), to_subscript(Val(mod(i, 10))))
    end
end

const SHOW_NODE_FACE = StyledStrings.Face(; foreground=:red)
const STENCIL_ACTIVE_FACE = StyledStrings.Face(; foreground=:blue, weight=:bold)

show_style(::STerm) = NamedTuple()
show_style(::SIndex) = (; italic=true)
show_style(::AbstractSTensor{R}) where {R} = (; bold=true, underline=(R > 0))
show_style(::BasisVector) = (; bold=true, underline=true, italic=true)

function style_face(style::NamedTuple)
    props = Pair{Symbol,Any}[]
    hasproperty(style, :bold) && style.bold && push!(props, :weight => :bold)
    hasproperty(style, :italic) && style.italic && push!(props, :slant => :italic)
    hasproperty(style, :underline) && style.underline && push!(props, :underline => true)
    hasproperty(style, :color) && push!(props, :foreground => style.color)
    isempty(props) && return nothing
    return StyledStrings.Face(; props...)
end

show_face(term::STerm) = style_face(show_style(term))

function annotated_text(text::AbstractString, face::Union{Nothing,StyledStrings.Face}=nothing)
    annotated = annotatedstring(text)
    if !isnothing(face) && !isempty(text)
        face!(annotated, firstindex(text):lastindex(text), face)
    end
    return annotated
end
annotated_text(text, face::Union{Nothing,StyledStrings.Face}=nothing) = annotated_text(string(text), face)

blank(width::Integer) = repeat(" ", max(0, width))

# These helpers keep layout operations annotation-preserving. Converting an
# AnnotatedString to a plain String would drop all faces, so every padding/join
# operation builds a new AnnotatedString by concatenating existing fragments.
annotated_join(lines::AbstractVector{<:AbstractString}) = isempty(lines) ? annotatedstring("") : reduce((l, r) -> annotatedstring(l, "\n", r), lines)
annotated_pad_right(line::AbstractString, width::Integer) = annotatedstring(line, blank(width - textwidth(line)))

# Fallback for STerm subtypes that have not opted into Chmy pretty printing.
# This deliberately uses Julia's default struct printer instead of calling
# `show`, because `show(::STerm)` delegates back to `styled`.
function default_annotated(x)
    io = IOBuffer()
    Base.show_default(io, x)
    return annotatedstring(String(take!(io)))
end

styled(x) = annotatedstring(sprint(show, x))
styled(term::STerm) = styled_expr(term, 0)

function term_display_string(term; color::Bool=false)
    return sprint(io -> print(IOContext(io, :color => color), styled(term)))
end

# Chmy display names

# `display_name` is the small, known-object classifier used by the expression
# renderer. It returns either the textual head/name that participates in Chmy's
# syntax, or `nothing` for unknown terms. Unknown terms then fall back to
# default Julia display instead of throwing a missing `variablename` MethodError.
display_name(::SLiteral{Value}) where {Value} = Value
display_name(sf::SFun) = nameof(sf.f)
display_name(::SRef{F}) where {F} = F
display_name(::Point) = :ᵖ
display_name(::Segment) = :ˢ
display_name(::AbstractDerivative) = :𝒟
display_name(::AbstractAveraging) = :ℐ
display_name(::Gradient) = :grad
display_name(::Divergence) = :divg
display_name(::Curl) = :curl
display_name(::BoundaryNormal) = :N
display_name(::BoundaryTangent) = :T
display_name(::BasisVector{I}) where {I} = Symbol(:e, to_subscript(Val(I)))
display_name(::SIndex{I}) where {I} = Symbol(:i, to_subscript(Val(I)))
display_name(::AbstractPartialDerivative{I}) where {I} = Symbol("∂", to_subscript(Val(I)))
display_name(::AbstractPartialAveraging{I}) where {I} = Symbol("ℐ", to_subscript(Val(I)))
display_name(::STerm) = nothing

# Chmy expression rendering

# Expression rendering threads a parent precedence value (`prec`) down the
# syntax tree. Each operator compares its own Julia precedence with the parent
# context; when the child would bind too weakly, it wraps itself in
# parentheses. This mirrors ordinary Julia printing while still producing
# AnnotatedStrings, so styling is attached during construction rather than as a
# later string-processing pass.
function styled_expr(expr::SExpr, prec::Int)
    if iscall(expr)
        op = operation(expr)
        args = arguments(expr)
        opname = display_name(op)
        # Calls are split into the compact syntaxes Chmy users expect:
        # broadcasted calls (`a .+ b`), infix/prefix operators (`a + b`, `-a`),
        # adjoints (`a'`), and ordinary function-call syntax (`sin(a)`).
        if op === SRef(:broadcasted)
            return styled_broadcast(args, prec)
        elseif opname isa Symbol && Meta.isoperator(opname)
            return styled_operator(opname, args, prec)
        elseif opname == :adjoint
            return annotatedstring(styled_call_arg(only(args)), "'")
        elseif opname isa Symbol
            op_prec = Base.operator_precedence(opname)
            return annotatedstring(string(opname), "(", styled_list(args, ", ", op_prec), ")")
        end
        return default_annotated(expr)
    elseif iscomp(expr)
        return annotatedstring(styled_expr(argument(expr), 0), "[", styled_list(indices(expr), ", ", 0), "]")
    elseif isind(expr)
        return annotatedstring(styled_call_arg(argument(expr)), "[", styled_list(indices(expr), ", ", 0), "]")
    elseif isloc(expr)
        pieces = Any[styled_call_arg(argument(expr))]
        for loc in location(expr)
            locname = display_name(loc)
            push!(pieces, isnothing(locname) ? styled(loc) : annotated_text(locname))
        end
        return annotatedstring(pieces...)
    end
    return default_annotated(expr)
end

function styled_expr(term::STerm, ::Int)
    name = display_name(term)
    isnothing(name) && return default_annotated(term)
    return annotated_text(name, show_face(term))
end

function styled_expr(::Shift{S}, ::Int) where {S}
    return annotatedstring("δ(", S, ")")
end

function styled_expr(node::SNode, ::Int)
    return annotatedstring(annotated_text("(", SHOW_NODE_FACE),
                           styled_expr(argument(node), 0),
                           annotated_text(")", SHOW_NODE_FACE))
end

function styled_expr(t::STensor{R,<:Any,<:Any,N}, ::Int) where {R,N}
    return annotated_text(N, show_face(t))
end

styled_expr(t::SZeroTensor, ::Int) = annotated_text('𝒪', show_face(t))
styled_expr(t::SIdTensor, ::Int) = annotated_text('ℐ', show_face(t))

function styled_broadcast(args, prec)
    op = first(args)
    bargs = Base.tail(args)
    opname = display_name(op)
    # Broadcast stores the broadcasted operator/function as its first argument.
    # Reuse the normal operator printer for dotted operators, but keep dotted
    # call syntax for non-operator functions.
    if opname isa Symbol && Meta.isoperator(opname)
        return styled_operator(Symbol('.', opname), bargs, prec; parent_opname=opname)
    elseif opname isa Symbol
        op_prec = Base.operator_precedence(opname)
        return annotatedstring(opname, ".(", styled_list(bargs, ", ", op_prec), ")")
    end
    return default_annotated(SExpr(Call(), SRef(:broadcasted), args...))
end

function styled_operator(display_opname::Symbol, args, prec; parent_opname::Symbol=display_opname)
    op_prec = Base.operator_precedence(parent_opname)
    parens = needs_parens(op_prec, prec)
    # `display_opname` is what appears in the string (`.+`), while
    # `parent_opname` is the semantic operator whose precedence Julia knows
    # (`+`). They differ only for dotted broadcast operators.
    body = if length(args) == 1
        annotatedstring(display_opname, styled_expr(only(args), op_prec))
    else
        styled_list(args, operator_separator(display_opname, parent_opname, args), op_prec, parent_opname)
    end
    return parens ? annotatedstring("(", body, ")") : body
end

function styled_call_arg(arg)
    # Function-call heads need a stricter wrapper than ordinary operator
    # precedence: `f(a + b)` is already parenthesized by the call, but
    # indexing/location of a call argument should display `(a + b)[i]`.
    if isexpr(arg) && iscall(arg)
        return annotatedstring("(", styled_expr(arg, 0), ")")
    else
        return styled_expr(arg, 0)
    end
end

function styled_list(items, sep, prec, parent_opname=nothing)
    pieces = Any[]
    for (i, item) in enumerate(items)
        i > 1 && push!(pieces, sep)
        # Some left children of non-associative or same-precedence operators
        # need a slightly lower effective parent precedence. That preserves
        # strings like `a - (b + c)` and `(-a) ^ b` without over-parenthesizing
        # every child expression.
        push!(pieces, styled_expr(item, child_precedence(parent_opname, item, prec, i)))
    end
    return annotatedstring(pieces...)
end

function operator_separator(display_opname::Symbol, parent_opname::Symbol, args)
    # Print numeric scalar multiplication in coefficient form (`2a`) when the
    # left side is a literal and the right side is a non-expression atom. More
    # complex products keep an explicit operator for readability.
    if parent_opname == :* &&
       length(args) == 2 &&
       args[1] isa SLiteral &&
       !(args[2] isa SLiteral) &&
       !isexpr(args[2]) &&
       display_name(args[1]) isa Union{AbstractFloat,Integer}
        return ""
    end
    return " $display_opname "
end

child_precedence(::Nothing, _, prec, _) = prec

function child_precedence(parent_opname::Symbol, item, prec, i)
    # Julia's precedence table alone is not enough for all infix formatting:
    # the first argument of many left-associative operators can bind slightly
    # looser without changing meaning, and unary minus under `+`/`-` needs the
    # same treatment to avoid misleading strings.
    if parent_opname in (:+, :-) && isunaryminus(item)
        return prec - 1
    elseif i == 1 && parent_opname in (:+, :-, :*, :/, ://, :÷, :⋅, :×, :⊡, :⊗, :&, :|, :xor)
        return prec - 1
    end
    return prec
end

needs_parens(op_prec::Int, prec::Int) = op_prec != 0 && prec != 0 && op_prec <= prec

# Tensor, binding, and rule display

function Base.show(io::IO, ::MIME"text/plain", v::Vec{N}) where {N}
    print(io, "$N-element Vec:")
    for i in 1:N
        print(io, '\n', ' ')
        show(io, v.components[i])
    end
end

function Base.show(io::IO, ::MIME"text/plain", t::Tensor{D,R}) where {D,R}
    join(io, ntuple(_ -> D, Val(R)), '×')
    print(io, " Tensor:")
    for idx in CartesianIndices(ntuple(_ -> D, Val(R)))
        inds = Tuple(idx)
        print(io, '\n', ' ', '[')
        join(io, inds, ", ")
        print(io, "] => ")
        show(io, t[inds...])
    end
end

function Base.show(io::IO, ::MIME"text/plain", t::SymTensor{D,R}) where {D,R}
    join(io, ntuple(_ -> D, Val(R)), '×')
    print(io, " SymTensor:")
    foreach_nondecreasing(Val(D), Val(R)) do I
        print(io, '\n', ' ', '[')
        join(io, I, ", ")
        print(io, "] => ")
        show(io, t[I...])
    end
end

function Base.show(io::IO, ::MIME"text/plain", t::AltTensor{D,R}) where {D,R}
    join(io, ntuple(_ -> D, Val(R)), '×')
    print(io, " AltTensor:")
    foreach_increasing(Val(D), Val(R)) do I
        print(io, '\n', ' ', '[')
        join(io, I, ", ")
        print(io, "] => ")
        show(io, t[I...])
    end
end

function Base.show(io::IO, ::MIME"text/plain", t::DiagTensor{D,R}) where {D,R}
    join(io, ntuple(_ -> D, Val(R)), '×')
    print(io, " DiagTensor:")
    for i in 1:D
        I = ntuple(_ -> i, Val(R))
        print(io, '\n', ' ', '[')
        join(io, I, ", ")
        print(io, "] => ")
        show(io, t[I...])
    end
end

function Base.show(io::IO, ::MIME"text/plain", b::Binding)
    padlength = maximum(f -> textwidth(string(styled(f))), b.exprs; init=0)
    if !haskey(io, :compact)
        io = IOContext(io, :compact => true)
    end
    print(io, "Binding:")
    for (expr, value) in zip(b.exprs, b.data)
        rendered = styled(expr)
        print(io, '\n', ' ')
        print(io, rendered)
        print(io, blank(padlength - textwidth(rendered)))
        print(io, " => ")
        show(io, value)
    end
end

Base.show(io::IO, rule::SubsRule) = print(io, styled(rule.lhs), " => ", styled(rule.rhs))

function Base.show(io::IO, ::MIME"text/plain", rule::SubsRule)
    print(io, "SubsRule:\n ")
    show(io, rule)
end

# Stencil geometry and line layout

function stencil_shift_coords(o::CartesianShift)
    return map(value, o.shifts)
end

function stencil_coords(o::CartesianShift, ::Nothing)
    return stencil_shift_coords(o)
end

function stencil_coords(o::CartesianShift, loc::Tuple)
    return tuplemap(stencil_coord, o.shifts, loc)
end

function stencil_coords(s::Stencil)
    return map(s.shifts) do shift
        stencil_coords(shift, s.location)
    end
end

function stencil_coord(shift::Shift, loc::Space)
    shifted_location = SLiteral(shift) + offset(loc)
    return value(shifted_location)
end

function Base.show(io::IO, shift::CartesianShift)
    print(io, "δ(")
    join(io, stencil_shift_coords(shift), ", ")
    print(io, ')')
end

stencil_location_name(::Point) = "Point()"
stencil_location_name(::Segment) = "Segment()"

function show_stencil_location(io::IO, ::Nothing) end
function show_stencil_location(io::IO, loc::Tuple{<:Space})
    print(io, stencil_location_name(only(loc)))
end
function show_stencil_location(io::IO, loc::Tuple)
    print(io, '(')
    names = map(stencil_location_name, loc)
    join(io, names, ", ")
    print(io, ')')
end

function Base.show(io::IO, s::Stencil)
    print(io, "Stencil(")
    if !isnothing(s.location)
        show_stencil_location(io, s.location)
        !isempty(s.shifts) && print(io, ", ")
    end
    join(io, s.shifts, ", ")
    print(io, ')')
end

function stencil_vertex_bounds(coords, nd::Integer=length(first(coords)))
    mins = fill(typemax(Int), nd)
    maxs = fill(typemin(Int), nd)
    for coord in coords
        for d in 1:nd
            mins[d] = min(mins[d], floor(Int, coord[d]))
            maxs[d] = max(maxs[d], ceil(Int, coord[d]))
        end
    end
    return [(mins[d], maxs[d]) for d in 1:nd]
end

# The stencil drawing lives on a half-step grid. Integer grid vertices are even
# half-coordinates, staggered segments are odd in one coordinate, and 2D cells
# are odd in both coordinates. Rendering is therefore a mapping problem: compute
# a layout once, draw each semantic half-grid object into a framebuffer, and
# only then convert the framebuffer to AnnotatedStrings.
stencil_vertex(active::Bool) = active ? '●' : '○'
stencil_xedge(active::Bool) = active ? '▶' : '▷'
stencil_yedge(active::Bool) = active ? '▼' : '▽'
stencil_cell(active::Bool) = active ? '■' : '□'

Base.@kwdef struct StencilRenderOptions
    x_ticks::Bool = true
    y_ticks::Bool = true
    x_axis_label::Bool = true
    y_axis_label::Bool = true
    left_margin::Int = 0
    right_margin::Int = 0
    slice_gap::Int = 1
    textwidth::Union{Nothing,Int} = nothing
end

struct StencilSliceOptions
    fixed::Tuple
    y_ticks::Bool
    reserve_y_tick_space::Bool
end

StencilSliceOptions(opts::StencilRenderOptions) = StencilSliceOptions((), opts.y_ticks, opts.y_ticks)

struct StencilGeometry{N,C,H,B}
    coords::C
    halfcoords::H
    bounds::B
end

stencil_render_ndims(::StencilGeometry{N}) where {N} = min(N, 2)

function StencilGeometry(s::Stencil{N}) where {N}
    coords = stencil_coords(s)
    nd = min(N, 2)
    halfcoords = map(coord -> projected_halfcoord(coord, nd), coords)
    bounds = stencil_vertex_bounds(coords, nd)
    return StencilGeometry{N,typeof(coords),typeof(halfcoords),typeof(bounds)}(coords, halfcoords, bounds)
end

function projected_halfcoord(coord, nd::Integer)
    return ntuple(i -> Int(2coord[i]), nd)
end

function active_in_slice(coord, fixed::Tuple)
    isempty(fixed) && return true
    return all(eachindex(fixed)) do d
        coord[d+2] == fixed[d]
    end
end

function active_halfcoords(geometry::StencilGeometry{N}, fixed::Tuple=()) where {N}
    nd = stencil_render_ndims(geometry)
    active = Set{NTuple{nd,Int}}()
    for (coord, halfcoord) in zip(geometry.coords, geometry.halfcoords)
        active_in_slice(coord, fixed) || continue
        push!(active, ntuple(i -> halfcoord[i], Val(nd)))
    end
    return active
end

struct StencilLayout1D
    xmin::Int
    xmax::Int
    sepwidth::Int
    prefixwidth::Int
    gridwidth::Int
    width::Int
    height::Int
    x_axis_row::Int
    glyph_row::Int
    x_tick_row::Int
end

function tick_label_width(values, visible::Bool)
    visible || return 1
    return maximum(value -> textwidth(string(value)), values)
end

function reserved_y_label_width(values, slice::StencilSliceOptions)
    slice.reserve_y_tick_space || return 0
    return maximum(value -> textwidth(string(value)), values)
end

function grid_center_col(prefixwidth::Integer, gridwidth::Integer)
    centered_grid_col = max(0, (gridwidth - 1) ÷ 2) + 1
    return prefixwidth + centered_grid_col
end

function axis_label_end_col(start_col::Integer, axis::Integer)
    label_width = textwidth(styled(SIndex(axis)))
    return start_col + label_width - 1
end

function StencilLayout1D(bounds, opts::StencilRenderOptions)
    xmin, xmax = bounds[1]
    tickwidth = tick_label_width(xmin:xmax, opts.x_ticks)
    sepwidth = max(9, opts.x_ticks ? tickwidth + 1 : 9)
    prefixwidth = opts.x_ticks ? tickwidth - 1 : 0
    gridwidth = stencil_grid_width(xmax - xmin + 1, sepwidth)
    x_axis_col = grid_center_col(prefixwidth, gridwidth)

    width = prefixwidth + gridwidth
    if opts.x_axis_label
        width = max(width, axis_label_end_col(x_axis_col, 1))
    end

    # Rows are assigned top-down. Optional labels use row 0 when disabled, so
    # drawing code can test the stored row without duplicating option checks.
    row = 1
    x_axis_row = opts.x_axis_label ? row : 0
    row += opts.x_axis_label ? 1 : 0
    glyph_row = row
    row += 1
    x_tick_row = opts.x_ticks ? row : 0
    row += opts.x_ticks ? 1 : 0
    return StencilLayout1D(xmin, xmax, sepwidth, prefixwidth, gridwidth, width, row - 1, x_axis_row, glyph_row, x_tick_row)
end

struct StencilLayout2D
    xmin::Int
    xmax::Int
    ymin::Int
    ymax::Int
    labelwidth::Int
    sepwidth::Int
    prefixwidth::Int
    gridwidth::Int
    gridheight::Int
    width::Int
    height::Int
    x_axis_row::Int
    grid_start_row::Int
    x_tick_row::Int
    y_axis_row::Int
    y_axis_col::Int
end

function StencilLayout2D(bounds, opts::StencilRenderOptions, slice::StencilSliceOptions)
    xmin, xmax = bounds[1]
    ymin, ymax = bounds[2]
    labelwidth = reserved_y_label_width(ymin:ymax, slice)
    tickwidth = tick_label_width(xmin:xmax, opts.x_ticks)
    sepwidth = max(9, opts.x_ticks ? tickwidth + 1 : 9)
    # Prefix columns are owned by visible/reserved y tick labels plus one
    # separator before the grid. There is no anonymous left gutter.
    y_label_prefix = slice.reserve_y_tick_space ? labelwidth + 1 : 0
    x_tick_prefix = opts.x_ticks ? tickwidth - 1 : 0
    prefixwidth = max(y_label_prefix, x_tick_prefix)
    gridwidth = stencil_grid_width(xmax - xmin + 1, sepwidth)
    gridheight = 1 + 4 * (ymax - ymin)
    x_axis_col = grid_center_col(prefixwidth, gridwidth)
    y_axis_col = prefixwidth + gridwidth + 2

    width = prefixwidth + gridwidth
    if opts.x_axis_label
        width = max(width, axis_label_end_col(x_axis_col, 1))
    end
    if opts.y_axis_label
        width = max(width, axis_label_end_col(y_axis_col, 2))
    end

    # The grid itself is the only fixed-height block. Axis labels and ticks are
    # attached above/below it and recorded as absolute rows within the layout.
    row = 1
    x_axis_row = opts.x_axis_label ? row : 0
    row += opts.x_axis_label ? 1 : 0
    grid_start_row = row
    row += gridheight
    x_tick_row = opts.x_ticks ? row : 0
    row += opts.x_ticks ? 1 : 0

    y_axis_row = opts.y_axis_label ? centered_y_axis_row(grid_start_row, ymin, ymax) : 0
    return StencilLayout2D(xmin, xmax, ymin, ymax, labelwidth, sepwidth, prefixwidth,
                           gridwidth, gridheight, width, row - 1, x_axis_row,
                           grid_start_row, x_tick_row, y_axis_row, y_axis_col)
end

function centered_y_axis_row(grid_start_row::Integer, ymin::Integer, ymax::Integer)
    nrows = ymax - ymin + 1
    if isodd(nrows)
        midrow = cld(nrows, 2)
        y = ymax - (midrow - 1)
        return grid_start_row + 4 * (ymax - y)
    else
        midedge = fld(nrows, 2)
        y = ymax - (midedge - 1)
        return grid_start_row + 4 * (ymax - y) + 2
    end
end

stencil_grid_width(nnodes::Integer, sepwidth::Integer) = nnodes + (nnodes - 1) * sepwidth

# Canvas coordinates are one-based row/column positions. The layout keeps grid
# coordinates (`x`, `y`) separate from draw coordinates; these helpers are the
# only place where grid positions become terminal columns/rows.
xnode_col(layout, x::Integer) = layout.prefixwidth + 1 + (x - layout.xmin) * (layout.sepwidth + 1)
xedge_col(layout, x::Integer) = xnode_col(layout, x) + 1 + fld(layout.sepwidth - 1, 2)
xaxis_col(layout) = grid_center_col(layout.prefixwidth, layout.gridwidth)
ynode_row(layout::StencilLayout2D, y::Integer) = layout.grid_start_row + 4 * (layout.ymax - y)

struct StencilCanvas
    cells::Matrix{Base.AnnotatedChar{Char}}
end

function StencilCanvas(height::Integer, width::Integer)
    cells = fill(Base.AnnotatedChar(' '), width + 1, height)
    cells[end, :] .= Base.AnnotatedChar('\n')
    return StencilCanvas(cells)
end

function stencil_char(char::Char, face=nothing)
    isnothing(face) && return Base.AnnotatedChar(char)
    annotations = [(label=:face, value=face)]
    return Base.AnnotatedChar(char, annotations)
end

function put!(canvas::StencilCanvas, row::Integer, col::Integer, char::Base.AnnotatedChar)
    1 <= row <= size(canvas.cells, 2) || throw(BoundsError(canvas.cells, (row, col)))
    1 <= col < size(canvas.cells, 1) || throw(BoundsError(canvas.cells, (row, col)))
    canvas.cells[col, row] = char
    return canvas
end

function put!(canvas::StencilCanvas, row::Integer, col::Integer, char::Char)
    return put!(canvas, row, col, stencil_char(char))
end

function put!(canvas::StencilCanvas, row::Integer, col::Integer, char::Char, face)
    annotated_char = stencil_char(char, face)
    return put!(canvas, row, col, annotated_char)
end

function put!(canvas::StencilCanvas, row::Integer, col::Integer, text::AbstractString)
    cursor = col
    for char in text
        put!(canvas, row, cursor, char)
        cursor += textwidth(char)
    end
    return canvas
end

function put_hline!(canvas::StencilCanvas, row::Integer, firstcol::Integer, lastcol::Integer, char::Char)
    for col in firstcol:lastcol
        put!(canvas, row, col, char)
    end
    return canvas
end

function put_right_aligned!(canvas::StencilCanvas, row::Integer, anchor_col::Integer, text)
    label = annotated_text(text)
    start_col = anchor_col - textwidth(label) + 1
    return put!(canvas, row, start_col, label)
end

struct Stencil3DSlice{O,S,L,T}
    opts::O
    slice::S
    layout::L
    title::T
    width::Int
end

struct StencilLayout3D{S}
    slices::S
    gap::Int
    width::Int
    height::Int
end

function stencil_slice_fixeds(geometry::StencilGeometry{3})
    z_values = sort(unique(coord[3] for coord in geometry.coords))
    return [(z,) for z in z_values]
end

function slice_render_options(opts::StencilRenderOptions, show_y_axis_label::Bool)
    return StencilRenderOptions(; x_ticks=opts.x_ticks,
                                y_ticks=opts.y_ticks,
                                x_axis_label=opts.x_axis_label,
                                y_axis_label=show_y_axis_label,
                                slice_gap=opts.slice_gap)
end

function slice_title(fixed::Tuple)
    title_pieces = Any[]
    for (j, value) in enumerate(fixed)
        j > 1 && push!(title_pieces, ", ")
        push!(title_pieces, styled(SIndex(j + 2)), " = ", string(value))
    end
    return annotatedstring(title_pieces...)
end

function anchored_title_start_col(layout, title)
    title_chars = collect(string(title))
    anchor_idx = something(findfirst(==('='), title_chars), 1)
    return xaxis_col(layout) - anchor_idx + 1
end

function slice_title_width(layout, title)
    start_col = anchored_title_start_col(layout, title)
    left_overflow = max(0, 1 - start_col)
    title_end_col = start_col + textwidth(title) - 1
    return max(layout.width, left_overflow + title_end_col)
end

function StencilLayout3D(geometry::StencilGeometry{3}, opts::StencilRenderOptions)
    fixeds = stencil_slice_fixeds(geometry)
    slices = Stencil3DSlice[]
    gap = opts.slice_gap

    # A 3D stencil is rendered as 2D z-slices. Only the first slice needs y tick
    # labels; only the last slice gets the y-axis label, so adjacent slices can
    # sit close together without repeating the same metadata.
    for (i, fixed) in enumerate(fixeds)
        show_y_ticks = opts.y_ticks && i == 1
        y_label = opts.y_axis_label && i == length(fixeds)
        slice_opts = slice_render_options(opts, y_label)
        slice = StencilSliceOptions(fixed, show_y_ticks, show_y_ticks)
        layout = StencilLayout2D(geometry.bounds, slice_opts, slice)
        title = slice_title(fixed)
        twidth = slice_title_width(layout, title)
        twidth > layout.width && (gap = max(gap, 3))
        push!(slices, Stencil3DSlice(slice_opts, slice, layout, title, twidth))
    end

    width = sum(slice -> slice.width, slices; init=0) + gap * max(0, length(slices) - 1)
    height = maximum(slice -> slice.layout.height + 1, slices; init=0)
    return StencilLayout3D(slices, gap, width, height)
end

stencil_layout(geometry::StencilGeometry{1}, opts::StencilRenderOptions) = StencilLayout1D(geometry.bounds, opts)
stencil_layout(geometry::StencilGeometry{2}, opts::StencilRenderOptions) = StencilLayout2D(geometry.bounds, opts, StencilSliceOptions(opts))
stencil_layout(geometry::StencilGeometry{3}, opts::StencilRenderOptions) = StencilLayout3D(geometry, opts)

function stencil_width(layout, opts::StencilRenderOptions)
    natural_width = layout.width + opts.left_margin + opts.right_margin
    return isnothing(opts.textwidth) ? natural_width : max(opts.textwidth, natural_width)
end

canvas_row(row0::Integer, layout_row::Integer) = row0 + layout_row - 1
canvas_col(col0::Integer, layout_col::Integer) = col0 + layout_col - 1
active_face(active::Bool) = active ? STENCIL_ACTIVE_FACE : nothing

function draw_x_axis_label!(canvas::StencilCanvas, row0::Integer, col0::Integer, layout)
    layout.x_axis_row == 0 && return canvas
    row = canvas_row(row0, layout.x_axis_row)
    col = canvas_col(col0, xaxis_col(layout))
    return put!(canvas, row, col, styled(SIndex(1)))
end

function draw_stencil!(canvas::StencilCanvas, row0::Integer, col0::Integer, geometry::StencilGeometry{1}, layout::StencilLayout1D)
    active = active_halfcoords(geometry)
    draw_x_axis_label!(canvas, row0, col0, layout)

    glyph_row = canvas_row(row0, layout.glyph_row)
    for x in layout.xmin:layout.xmax
        node_active = (2x,) in active
        node_col = canvas_col(col0, xnode_col(layout, x))
        put!(canvas, glyph_row, node_col, stencil_vertex(node_active), active_face(node_active))

        if x < layout.xmax
            first_edge_col = canvas_col(col0, xnode_col(layout, x) + 1)
            last_edge_col = canvas_col(col0, xnode_col(layout, x + 1) - 1)
            put_hline!(canvas, glyph_row, first_edge_col, last_edge_col, '─')

            edge_active = (2x + 1,) in active
            edge_col = canvas_col(col0, xedge_col(layout, x))
            put!(canvas, glyph_row, edge_col, stencil_xedge(edge_active), active_face(edge_active))
        end

        if layout.x_tick_row != 0
            tick_row = canvas_row(row0, layout.x_tick_row)
            put_right_aligned!(canvas, tick_row, node_col, string(x))
        end
    end
    return canvas
end

function draw_stencil!(canvas::StencilCanvas, row0::Integer, col0::Integer, geometry::StencilGeometry{N}, layout::StencilLayout2D,
                       slice::StencilSliceOptions=StencilSliceOptions((), layout.labelwidth > 0, layout.labelwidth > 0)) where {N}
    active = active_halfcoords(geometry, slice.fixed)
    draw_x_axis_label!(canvas, row0, col0, layout)

    # Draw one horizontal grid row at a time, then fill the vertical connectors,
    # y-edges, and cell centers that live between this row and the next one.
    for y in layout.ymax:-1:layout.ymin
        node_row = canvas_row(row0, ynode_row(layout, y))
        if slice.y_ticks
            put!(canvas, node_row, col0, lpad(string(y), layout.labelwidth))
        end

        for x in layout.xmin:layout.xmax
            node_active = (2x, 2y) in active
            node_col = canvas_col(col0, xnode_col(layout, x))
            put!(canvas, node_row, node_col, stencil_vertex(node_active), active_face(node_active))

            if x < layout.xmax
                first_edge_col = canvas_col(col0, xnode_col(layout, x) + 1)
                last_edge_col = canvas_col(col0, xnode_col(layout, x + 1) - 1)
                put_hline!(canvas, node_row, first_edge_col, last_edge_col, '─')

                xedge_active = (2x + 1, 2y) in active
                edge_col = canvas_col(col0, xedge_col(layout, x))
                put!(canvas, node_row, edge_col, stencil_xedge(xedge_active), active_face(xedge_active))
            end
        end

        y == layout.ymin && continue
        hy = 2y - 1
        for x in layout.xmin:layout.xmax
            node_col = canvas_col(col0, xnode_col(layout, x))
            put!(canvas, node_row + 1, node_col, '│')

            yedge_active = (2x, hy) in active
            put!(canvas, node_row + 2, node_col, stencil_yedge(yedge_active), active_face(yedge_active))
            put!(canvas, node_row + 3, node_col, '│')

            if x < layout.xmax
                cell_active = (2x + 1, hy) in active
                cell_col = canvas_col(col0, xedge_col(layout, x))
                put!(canvas, node_row + 2, cell_col, stencil_cell(cell_active), active_face(cell_active))
            end
        end
    end

    if layout.x_tick_row != 0
        tick_row = canvas_row(row0, layout.x_tick_row)
        for x in layout.xmin:layout.xmax
            node_col = canvas_col(col0, xnode_col(layout, x))
            put_right_aligned!(canvas, tick_row, node_col, string(x))
        end
    end
    if layout.y_axis_row != 0
        row = canvas_row(row0, layout.y_axis_row)
        col = canvas_col(col0, layout.y_axis_col)
        put!(canvas, row, col, styled(SIndex(2)))
    end
    return canvas
end

function draw_stencil!(canvas::StencilCanvas, row0::Integer, col0::Integer, geometry::StencilGeometry{3}, layout::StencilLayout3D)
    col = 1
    for slice in layout.slices
        draw_stencil!(canvas, row0, col0 + col - 1, geometry, slice.layout, slice.slice)
        start_col = anchored_title_start_col(slice.layout, slice.title)
        left_padding = max(0, 1 - start_col)
        title_row = row0 + slice.layout.height
        title_col = col0 + col + left_padding + start_col - 2
        put!(canvas, title_row, title_col, slice.title)
        col += slice.width + layout.gap
    end
    return canvas
end

function render_stencil(s::Stencil, opts::StencilRenderOptions)
    N = ndims(s)
    1 <= N <= 3 || throw(ArgumentError("stencil rendering supports only 1D, 2D, and 3D stencils, got $(N)D"))
    isempty(s.shifts) && throw(ArgumentError("cannot render an empty stencil"))
    geometry = StencilGeometry(s)
    return render_stencil(geometry, opts)
end

function render_stencil(geometry::StencilGeometry, opts::StencilRenderOptions)
    layout = stencil_layout(geometry, opts)
    width = stencil_width(layout, opts)
    canvas = StencilCanvas(layout.height, width)
    natural_width = layout.width + opts.left_margin + opts.right_margin
    left_padding = opts.left_margin + fld(width - natural_width, 2)
    draw_stencil!(canvas, 1, left_padding + 1, geometry, layout)
    return join(view(canvas.cells, 1:lastindex(canvas.cells)-1))
end

function render_stencil(s::Stencil; kwargs...)
    # Public stencil renderer: normalize keyword options once, draw the stencil
    # into an annotation-preserving framebuffer, and flatten that framebuffer
    # directly into the AnnotatedString that `show` prints.
    return render_stencil(s, StencilRenderOptions(; kwargs...))
end

function stencil_render_width(s::Stencil, opts::StencilRenderOptions)
    geometry = StencilGeometry(s)
    return stencil_render_width(geometry, opts)
end

function stencil_render_width(geometry::StencilGeometry, opts::StencilRenderOptions)
    layout = stencil_layout(geometry, opts)
    return stencil_width(layout, opts)
end

function stencil_render_height(s::Stencil, opts::StencilRenderOptions)
    geometry = StencilGeometry(s)
    return stencil_render_height(geometry, opts)
end

function stencil_render_height(geometry::StencilGeometry, opts::StencilRenderOptions)
    return stencil_layout(geometry, opts).height
end

function vertically_centered_cell(text::AbstractString, height::Integer)
    top = max(0, fld(height - 1, 2))
    bottom = max(0, height - top - 1)
    leading_blanks = fill(annotatedstring(""), top)
    trailing_blanks = fill(annotatedstring(""), bottom)
    return annotated_join(vcat(leading_blanks, [text], trailing_blanks))
end

# Base.show entrypoints

Base.show(io::IO, term::STerm) = print(io, styled(term))

function Base.show(io::IO, ::MIME"text/plain", term::STerm)
    isexpr(term) && print(io, "StaticExpression:\n ")
    print(io, styled(term))
end

function Base.show(io::IO, ::MIME"text/plain", s::Stencil)
    print(io, "Stencil ($(ndims(s))D, $(length(s.shifts)) shifts")
    if !isnothing(s.location)
        print(io, ", location ")
        show_stencil_location(io, s.location)
    end
    print(io, "):\n")
    print(io, render_stencil(s))
end

function Base.show(io::IO, nu::Nonuniforms)
    print(io, "Nonuniforms(")
    for (i, (term, stencil)) in enumerate(pairstuple(stencils(nu)))
        i > 1 && print(io, ", ")
        print(io, styled(term), " => ", stencil)
    end
    print(io, ')')
end

function nonuniform_dimensions(pairs)
    dimensions = map(pairs) do pair
        ndims(last(pair))
    end
    return sort(unique(dimensions))
end

function combined_stencil_geometry(pairs, N::Integer)
    coords = Any[]
    for pair in pairs
        stencil = last(pair)
        ndims(stencil) == N || continue
        append!(coords, stencil_coords(stencil))
    end

    nd = min(N, 2)
    halfcoords = map(coord -> projected_halfcoord(coord, nd), coords)
    bounds = stencil_vertex_bounds(coords, nd)
    geometry = StencilGeometry{N,typeof(coords),typeof(halfcoords),typeof(bounds)}
    return geometry(coords, halfcoords, bounds)
end

function nonuniform_total_geometries(pairs)
    return map(nonuniform_dimensions(pairs)) do N
        N => combined_stencil_geometry(pairs, N)
    end
end

function Base.show(io::IO, ::MIME"text/plain", nu::Nonuniforms)
    pairs = pairstuple(stencils(nu))
    isempty(pairs) && return show(io, nu)

    stencil_opts = StencilRenderOptions(; left_margin=1, right_margin=1)

    total_geometries = nonuniform_total_geometries(pairs)

    total_width = maximum(total_geometries) do pair
        stencil_render_width(last(pair), stencil_opts)
    end
    cell_width = maximum(values(stencils(nu)); init=total_width) do stencil
        stencil_render_width(stencil, stencil_opts)
    end
    cell_opts = StencilRenderOptions(; left_margin=1, right_margin=1, textwidth=cell_width)

    data = Matrix{Any}(undef, length(pairs) + length(total_geometries), 2)
    for (i, (N, geometry)) in enumerate(total_geometries)
        total_height = stencil_render_height(geometry, cell_opts)
        data[i, 1] = vertically_centered_cell(annotatedstring("Full $(N)D"), total_height)
        data[i, 2] = render_stencil(geometry, cell_opts)
    end

    for (i, (term, stencil)) in enumerate(pairs)
        height = stencil_render_height(stencil, cell_opts)
        row = i + length(total_geometries)
        data[row, 1] = vertically_centered_cell(styled(term), height)
        data[row, 2] = render_stencil(stencil, cell_opts)
    end
    pretty_table(io, data;
                 title="Nonuniforms",
                 column_labels=["Term", "Stencil"],
                 row_group_labels=[length(total_geometries) + 1 => "Fields"],
                 alignment=[:c, :l],
                 column_label_alignment=[:c, :c],
                 line_breaks=true,
                 fit_table_in_display_horizontally=false,
                 fit_table_in_display_vertically=false,
                 table_format=TextTableFormat(; horizontal_lines_at_data_rows=:all),)
end
