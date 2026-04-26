function to_subscript(::Val{i}) where {i}
    if i <= 9
        return Symbol('₀' + i)
    else
        # recurse for multi-digit numbers
        return Symbol(to_subscript(Val(div(i, 10))), to_subscript(Val(mod(i, 10))))
    end
end

variablename(::SLiteral{Value}) where {Value} = Value

variablename(sf::SFun)            = nameof(sf.f)
variablename(::SRef{F}) where {F} = F

variablename(::Point)   = :ᵖ
variablename(::Segment) = :ˢ

variablename(::AbstractDerivative) = :𝒟
variablename(::AbstractAveraging)  = :ℐ
variablename(::Gradient)           = :grad
variablename(::Divergence)         = :divg
variablename(::Curl)               = :curl

variablename(::SIndex{I}) where {I} = Symbol(:i, to_subscript(Val(I)))

function variablename(::AbstractPartialDerivative{I}) where {I}
    return Symbol("∂", to_subscript(Val(I)))
end

function variablename(::AbstractPartialAveraging{I}) where {I}
    return Symbol("ℐ", to_subscript(Val(I)))
end

Base.show(io::IO, term::STerm) = show_static(io, term, 0)

function Base.show(io::IO, ::MIME"text/plain", term::STerm)
    if isexpr(term)
        print(io, "StaticExpression:\n ")
    end
    show_static(io, term, 0)
end

show_static(io, ::Shift{S}, ::Int) where {S} = print(io, "δ(", S, ')')

show_static(io, si::SIndex, ::Int) = printstyled(io, variablename(si); italic=true)

show_static(io, ::STensor{R,<:Any,<:Any,N}, ::Int) where {R,N} = printstyled(io, N; bold=true, underline=(R > 0))

show_static(io, ::SZeroTensor, ::Int) = printstyled(io, '𝒪'; bold=true, underline=true)
show_static(io, ::SIdTensor, ::Int) = printstyled(io, 'ℐ'; bold=true, underline=true)
function show_static(io, node::SNode, ::Int)
    printstyled(io, '('; color=:red)
    show_static(io, argument(node), 0)
    printstyled(io, ')'; color=:red)
    return
end

function show_static(io, expr::SExpr, prec::Int)
    if iscall(expr)
        op = operation(expr)
        args = arguments(expr)
        opname = variablename(op)
        if op === SRef(:broadcasted)
            show_broadcast(io, args, prec)
        elseif Meta.isoperator(opname)
            show_operator(io, opname, args, prec)
        elseif opname == :adjoint
            show_call_arg(io, only(args))
            print(io, ''')
        else
            op_prec = Base.operator_precedence(opname)
            print(io, opname)
            print(io, '(')
            show_list(io, args, ", ", op_prec)
            print(io, ')')
        end
    elseif iscomp(expr)
        arg = argument(expr)
        show_static(io, arg, 0)
        print(io, '[')
        show_list(io, indices(expr), ", ", 0)
        print(io, ']')
    elseif isind(expr)
        show_call_arg(io, argument(expr))
        print(io, '[')
        show_list(io, indices(expr), ", ", 0)
        print(io, ']')
    elseif isloc(expr)
        show_call_arg(io, argument(expr))
        for s in location(expr)
            print(io, variablename(s))
        end
    end
    return
end

show_static(io, term::STerm, _) = print(io, variablename(term))

function show_broadcast(io, args, prec)
    op = first(args)
    bargs = Base.tail(args)
    opname = variablename(op)
    if Meta.isoperator(opname)
        show_operator(io, Symbol('.', opname), bargs, prec; parent_opname=opname)
    else
        op_prec = Base.operator_precedence(opname)
        print(io, opname)
        print(io, ".(")
        show_list(io, bargs, ", ", op_prec)
        print(io, ')')
    end
    return
end

function show_operator(io, display_opname::Symbol, args, prec; parent_opname::Symbol=display_opname)
    op_prec = Base.operator_precedence(parent_opname)
    parens = needs_parens(op_prec, prec)
    parens && print(io, '(')
    if length(args) == 1
        print(io, display_opname)
        show_static(io, only(args), op_prec)
    else
        show_list(io, args, operator_separator(display_opname, parent_opname, args), op_prec, parent_opname)
    end
    parens && print(io, ')')
    return
end

function operator_separator(display_opname::Symbol, parent_opname::Symbol, args)
    if parent_opname == :* &&
       length(args) == 2 &&
       args[1] isa SLiteral &&
       !(args[2] isa SLiteral) &&
       !isexpr(args[2]) &&
       variablename(args[1]) isa Union{AbstractFloat,Integer}
        return ""
    end
    return " $display_opname "
end

function show_call_arg(io, arg)
    if isexpr(arg) && iscall(arg)
        print(io, '(')
        show_static(io, arg, 0)
        print(io, ')')
    else
        show_static(io, arg, 0)
    end
    return
end

function show_list(io, items, sep, prec, parent_opname=nothing)
    for (i, item) in enumerate(items)
        i > 1 && print(io, sep)
        show_static(io, item, child_precedence(parent_opname, item, prec, i))
    end
    return
end

child_precedence(::Nothing, _, prec, _) = prec

function child_precedence(parent_opname::Symbol, item, prec, i)
    if parent_opname in (:+, :-) && isunaryminus(item)
        return prec - 1
    elseif i == 1 && parent_opname in (:+, :-, :*, :/, ://, :÷, :⋅, :×, :⊡, :⊗, :&, :|, :xor)
        return prec - 1
    end
    return prec
end

needs_parens(op_prec::Int, prec::Int) = op_prec != 0 && prec != 0 && op_prec <= prec

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

Base.show(io::IO, rule::SubsRule) = print(io, rule.lhs, " => ", rule.rhs)

function Base.show(io::IO, ::MIME"text/plain", rule::SubsRule)
    print(io, "SubsRule:\n ")
    show(io, rule)
end

stencil_shift_coords(o::CartesianShift) = map(shift_value, o.shifts)
stencil_coords(o::CartesianShift, ::Nothing) = stencil_shift_coords(o)
stencil_coords(o::CartesianShift, loc::Tuple) = tuplemap(stencil_coord, o.shifts, loc)
stencil_coords(s::Stencil) = map(shift -> stencil_coords(shift, s.location), s.shifts)
stencil_coord(shift::Shift, loc::Space) = value(SLiteral(shift) + offset(loc))
stencil_axis_name(dim::Integer) = string(Symbol(:i, to_subscript(Val(dim))))

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
    join(io, map(stencil_location_name, loc), ", ")
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

function stencil_slice_matches(coord, fixed::Tuple)
    length(coord) == 2 && return isempty(fixed)
    return all(coord[d+2] == fixed[d] for d in eachindex(fixed))
end

stencil_halfstep_index(coord::Real) = isinteger(2coord) ? Int(2coord) : nothing

function stencil_halfgrid_coord(coord, nd::Integer)
    halfcoord = ntuple(i -> stencil_halfstep_index(coord[i]), nd)
    any(isnothing, halfcoord) && return nothing
    return ntuple(i -> something(halfcoord[i]), nd)
end

function stencil_halfgrid_coords(coords, nd::Integer)
    halfcoords = map(coord -> stencil_halfgrid_coord(coord, nd), coords)
    any(isnothing, halfcoords) && return nothing
    return map(coord -> something(coord), halfcoords)
end

stencil_vertex(active::Bool) = active ? '●' : '○'
stencil_xedge(active::Bool) = active ? '▶' : '▷'
stencil_yedge(active::Bool) = active ? '▼' : '▽'
stencil_cell(active::Bool) = active ? '■' : '□'

function stencil_symbol_span(symbol::Char, sepwidth::Integer; fill::Char)
    left = fld(sepwidth - 1, 2)
    right = sepwidth - left - 1
    return repeat(string(fill), left) * string(symbol) * repeat(string(fill), right)
end

function stencil_node_row_string(active_nodes, active_segments, sepwidth::Integer=3)
    nodes = collect(active_nodes)
    segments = collect(active_segments)
    chars = String[]
    for (i, active) in enumerate(nodes)
        push!(chars, string(stencil_vertex(active)))
        i == length(nodes) && continue
        push!(chars, stencil_symbol_span(stencil_xedge(segments[i]), sepwidth; fill='─'))
    end
    return join(chars)
end

function stencil_edge_row_string(active_edges, active_cells, sepwidth::Integer=3)
    edges = collect(active_edges)
    cells = collect(active_cells)
    chars = String[]
    for (i, active) in enumerate(edges)
        push!(chars, string(stencil_yedge(active)))
        i == length(edges) && continue
        push!(chars, stencil_symbol_span(stencil_cell(cells[i]), sepwidth; fill=' '))
    end
    return join(chars)
end

stencil_vertical_border_row_string(n::Integer, sepwidth::Integer=3) = join(fill("│", n), repeat(" ", sepwidth))

stencil_fixed_indices_string(fixed::Tuple) = join((string(stencil_axis_name(i + 2), " = ", value) for (i, value) in enumerate(fixed)), ", ")

function stencil_label_line(labels, prefixwidth::Integer, sepwidth::Integer)
    # Tick labels need explicit placement so values like `-1` stay centered under their nodes.
    nodecols = [prefixwidth + 1 + (i - 1) * (sepwidth + 1) for i in eachindex(labels)]
    line = fill(' ', nodecols[end])
    for (label, nodecol) in zip(labels, nodecols)
        startcol = nodecol - textwidth(label) + 1
        for (offset, char) in enumerate(label)
            line[startcol+offset-1] = char
        end
    end
    return String(line)
end

stencil_grid_width(nnodes::Integer, sepwidth::Integer) = nnodes + (nnodes - 1) * sepwidth

function stencil_centered_line(label::AbstractString, prefixwidth::Integer, contentwidth::Integer)
    # Center the leading glyph on the axis and let any trailing modifiers extend to the right.
    chars = collect(label)
    startcol = prefixwidth + max(0, (contentwidth - 1) ÷ 2) + 1
    line = fill(' ', max(prefixwidth + contentwidth, startcol + length(chars) - 1))
    for (offset, char) in enumerate(chars)
        line[startcol+offset-1] = char
    end
    return String(line)
end

stencil_axis_anchor_column(prefixwidth::Integer, gridwidth::Integer) = prefixwidth + max(0, (gridwidth - 1) ÷ 2) + 1

function stencil_anchor_text_line(label::AbstractString, anchor::AbstractChar, anchor_col::Integer, minwidth::Integer)
    chars = collect(label)
    anchor_idx = something(findfirst(==(anchor), chars), 1)
    startcol = anchor_col - anchor_idx + 1
    leftpad = max(0, 1 - startcol)
    width = max(minwidth, leftpad + startcol + length(chars) - 1)
    line = fill(' ', width)
    for (i, char) in enumerate(chars)
        line[leftpad+startcol+i-1] = char
    end
    return String(line)
end

function stencil_xlabels_line(xmin::Integer, xmax::Integer, prefixwidth::Integer, sepwidth::Integer)
    labels = map(string, xmin:xmax)
    return stencil_label_line(labels, prefixwidth, sepwidth)
end

function stencil_slice_layout_2d(bounds;
                                 show_x_ticks::Bool=true,
                                 show_y_ticks::Bool=true,
                                 reserve_y_tick_space::Bool=show_y_ticks)
    xmin, xmax = bounds[1]
    ymin, ymax = bounds[2]
    labelwidth = reserve_y_tick_space ? maximum(y -> textwidth(string(y)), ymin:ymax) : 0
    tickwidth = show_x_ticks ? maximum(x -> textwidth(string(x)), xmin:xmax) : 1
    sepwidth = max(9, show_x_ticks ? tickwidth + 1 : 9)
    prefixwidth = max(reserve_y_tick_space ? labelwidth + 2 : 0, show_x_ticks ? tickwidth - 1 : 0)
    gridwidth = stencil_grid_width(xmax - xmin + 1, sepwidth)
    return (; xmin, xmax, ymin, ymax, labelwidth, tickwidth, sepwidth, prefixwidth, gridwidth)
end

stencil_pad_line(line::AbstractString, width::Integer) = rpad(line, width)
stencil_trim_prefix(line::AbstractString, width::Integer) = textwidth(line) <= width ? "" : line[nextind(line, firstindex(line), width):end]

function stencil_leading_space_width(line::AbstractString)
    width = 0
    for char in line
        char == ' ' || break
        width += 1
    end
    return width
end

function stencil_hcat_blocks(blocks::Vector{<:Vector{String}}; gap::Integer=3)
    widths = [maximum(textwidth, block; init=0) for block in blocks]
    height = maximum(length, blocks; init=0)
    return [join((stencil_pad_line(row <= length(block) ? block[row] : "", width) for (block, width) in zip(blocks, widths)),
                 repeat(" ", gap)) for row in 1:height]
end

stencil_style_line(line::String) = Base.annotatedstring(line)

stencil_style_lines(lines::AbstractVector{String}) = map(stencil_style_line, lines)

function stencil_join_lines(lines::AbstractVector{<:AbstractString})
    isempty(lines) && return Base.annotatedstring("")
    return reduce((left, right) -> Base.annotatedstring(left, "\n", right), lines)
end

function stencil_centered_block(lines::AbstractVector{<:AbstractString}, width::Integer)
    # Center the multiline stencil as one block so narrow slices keep their internal geometry.
    blockwidth = maximum(textwidth, lines; init=0)
    leftpad = max(0, (width - blockwidth) ÷ 2)
    padded = [rpad(repeat(" ", leftpad) * line, width) for line in lines]
    return stencil_join_lines(padded)
end

function stencil_slice_lines_1d(halfcoords, bounds; show_ticks::Bool=true, show_axis_labels::Bool=true)
    xmin, xmax = bounds[1]
    active = Set(first(coord) for coord in halfcoords)
    tickwidth = show_ticks ? maximum(x -> textwidth(string(x)), xmin:xmax) : 1
    sepwidth = max(9, show_ticks ? tickwidth + 1 : 9)
    prefixwidth = show_ticks ? tickwidth - 1 : 0
    gridwidth = stencil_grid_width(xmax - xmin + 1, sepwidth)
    lines = String[]
    if show_axis_labels
        push!(lines, stencil_centered_line(stencil_axis_name(1), prefixwidth, gridwidth))
    end
    push!(lines,
          rpad("", prefixwidth) *
          stencil_node_row_string((2x in active for x in xmin:xmax),
                                  (2x + 1 in active for x in xmin:xmax-1),
                                  sepwidth))
    show_ticks && push!(lines, stencil_xlabels_line(xmin, xmax, prefixwidth, sepwidth))
    return lines
end

function stencil_slice_lines_2d(coords, halfcoords, bounds, fixed::Tuple=();
                                show_x_ticks::Bool=true,
                                show_y_ticks::Bool=true,
                                reserve_y_tick_space::Bool=show_y_ticks,
                                show_x_axis_label::Bool=true,
                                show_y_axis_label::Bool=true)
    layout = stencil_slice_layout_2d(bounds; show_x_ticks, show_y_ticks, reserve_y_tick_space)
    xmin, xmax = layout.xmin, layout.xmax
    ymin, ymax = layout.ymin, layout.ymax
    active = Set((halfcoord[1], halfcoord[2]) for (coord, halfcoord) in zip(coords, halfcoords) if stencil_slice_matches(coord, fixed))
    labelwidth = layout.labelwidth
    sepwidth = layout.sepwidth
    prefixwidth = layout.prefixwidth
    gridwidth = layout.gridwidth
    nrows = ymax - ymin + 1
    midrow = cld(nrows, 2)
    midedge = fld(nrows, 2)
    lines = String[]
    show_x_axis_label && push!(lines, stencil_centered_line(stencil_axis_name(1), prefixwidth, gridwidth))
    blankprefix = repeat(" ", prefixwidth)
    for (row, y) in enumerate(ymax:-1:ymin)
        prefix = show_y_ticks ? " $(lpad(string(y), labelwidth)) " : blankprefix
        suffix = show_y_axis_label && isodd(nrows) && row == midrow ? " " * stencil_axis_name(2) : ""
        push!(lines,
              prefix *
              stencil_node_row_string(((2x, 2y) in active for x in xmin:xmax),
                                      ((2x + 1, 2y) in active for x in xmin:xmax-1),
                                      sepwidth) *
              suffix)
        if row < ymax - ymin + 1
            edgeprefix = show_y_ticks ? " $(repeat(" ", labelwidth)) " : blankprefix
            centersuffix = show_y_axis_label && iseven(nrows) && row == midedge ? " " * stencil_axis_name(2) : ""
            ncols = xmax - xmin + 1
            hy = 2y - 1
            push!(lines,
                  edgeprefix *
                  stencil_vertical_border_row_string(ncols, sepwidth))
            push!(lines,
                  edgeprefix *
                  stencil_edge_row_string(((2x, hy) in active for x in xmin:xmax),
                                          ((2x + 1, hy) in active for x in xmin:xmax-1),
                                          sepwidth) *
                  centersuffix)
            push!(lines,
                  edgeprefix *
                  stencil_vertical_border_row_string(ncols, sepwidth))
        end
    end
    show_x_ticks && push!(lines, stencil_xlabels_line(xmin, xmax, prefixwidth, sepwidth))
    return lines
end

function stencil_render_lines(s::Stencil; show_ticks::Bool=true, show_axis_labels::Bool=true)
    isempty(s.shifts) && return [Base.annotatedstring(sprint(show, s))]
    coords = stencil_coords(s)
    nd = length(first(coords))
    halfcoords = stencil_halfgrid_coords(coords, min(nd, 2))
    isnothing(halfcoords) && return [Base.annotatedstring(sprint(show, s))]
    bounds = stencil_vertex_bounds(coords, min(nd, 2))
    if nd == 1
        return stencil_style_lines(stencil_slice_lines_1d(halfcoords, bounds; show_ticks, show_axis_labels))
    elseif nd == 2
        return stencil_style_lines(stencil_slice_lines_2d(coords, halfcoords, bounds;
                                                          show_x_ticks=show_ticks,
                                                          show_y_ticks=show_ticks,
                                                          reserve_y_tick_space=show_ticks,
                                                          show_x_axis_label=show_axis_labels,
                                                          show_y_axis_label=show_axis_labels))
    elseif nd != 3
        return [Base.annotatedstring(sprint(show, s))]
    end
    ranges = [sort(unique(coord[d] for coord in coords)) for d in 3:nd]
    fixeds = collect(Tuple(values) for values in Iterators.product(ranges...))
    blocks = Vector{Vector{String}}()
    gap = 1
    for (i, fixed) in enumerate(fixeds)
        # Horizontal slice layouts keep x ticks on every slice, but avoid repeated y ticks.
        show_y_ticks = show_ticks && i == 1
        slicelines = stencil_slice_lines_2d(coords, halfcoords, bounds, fixed;
                                            show_x_ticks=show_ticks,
                                            show_y_ticks=show_y_ticks,
                                            reserve_y_tick_space=show_ticks,
                                            show_x_axis_label=show_axis_labels,
                                            show_y_axis_label=show_axis_labels && i == length(fixeds))
        fixedline = stencil_fixed_indices_string(fixed)
        layout = stencil_slice_layout_2d(bounds;
                                         show_x_ticks=show_ticks,
                                         show_y_ticks=show_y_ticks,
                                         reserve_y_tick_space=show_ticks)
        trimwidth = show_y_ticks ? 0 : minimum(stencil_leading_space_width, slicelines; init=0)
        slicelines = trimwidth == 0 ? slicelines : map(line -> stencil_trim_prefix(line, trimwidth), slicelines)
        blockwidth = maximum(textwidth, slicelines; init=0)
        textwidth(fixedline) > blockwidth && (gap = 3)
        # Align the `=` in the slice title under the `i` of the x-axis label.
        titleline = stencil_anchor_text_line(fixedline,
                                             '=',
                                             stencil_axis_anchor_column(layout.prefixwidth, layout.gridwidth) - trimwidth,
                                             blockwidth)
        push!(blocks, [map(line -> stencil_pad_line(line, blockwidth), slicelines)..., titleline])
    end
    return stencil_style_lines(stencil_hcat_blocks(blocks; gap))
end

function vertically_centered_cell(text::AbstractString, height::Integer)
    # Pad only above/below the term so PrettyTables can keep the stencil cell untouched.
    top = max(0, fld(height - 1, 2))
    bottom = max(0, height - top - 1)
    return join(vcat(fill("", top), [text], fill("", bottom)), '\n')
end

function Base.show(io::IO, ::MIME"text/plain", s::Stencil)
    isempty(s.shifts) && return show(io, s)
    ndims(s) > 3 && return show(io, s)
    print(io, "Stencil ($(ndims(s))D, $(length(s.shifts)) shifts")
    if !isnothing(s.location)
        print(io, ", location ")
        show_stencil_location(io, s.location)
    end
    print(io, "):\n")
    print(io, stencil_join_lines(stencil_render_lines(s)))
end

function Base.show(io::IO, nu::Nonuniforms)
    print(io, "Nonuniforms(")
    for (i, (term, stencil)) in enumerate(pairstuple(stencils(nu)))
        i > 1 && print(io, ", ")
        print(io, term, " => ", stencil)
    end
    print(io, ')')
end

function Base.show(io::IO, ::MIME"text/plain", nu::Nonuniforms)
    pairs = pairstuple(stencils(nu))
    isempty(pairs) && return show(io, nu)
    if any(ndims(stencil) > 3 for (_, stencil) in pairs)
        padlength = maximum(term -> textwidth(repr(term)), keys(stencils(nu)))
        println(io, "Nonuniforms:")
        for (term, stencil) in pairs
            println(io, ' ', rpad(repr(term), padlength), " => ", stencil)
        end
        return
    end
    stencil_lines = [stencil_render_lines(stencil) for (_, stencil) in pairs]
    stencil_width = maximum(lines -> maximum(textwidth, lines; init=0), stencil_lines; init=0)
    data = Matrix{Any}(undef, length(pairs), 2)
    for (i, (term, stencil)) in enumerate(pairs)
        lines = stencil_lines[i]
        height = length(lines)
        data[i, 1] = vertically_centered_cell(repr(term), height)
        data[i, 2] = stencil_centered_block(lines, stencil_width)
    end
    pretty_table(io, data;
                 column_labels=["term", "stencil"],
                 alignment=[:c, :l],
                 column_label_alignment=[:c, :c],
                 line_breaks=true,
                 fit_table_in_display_horizontally=false,
                 fit_table_in_display_vertically=false,
                 table_format=TextTableFormat(; horizontal_lines_at_data_rows=:all),)
end
