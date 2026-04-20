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

show_static(io, ::Offset{O}, ::Int) where {O} = print(io, "δ(", O, ')')

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

offset_value(::Offset{O}) where {O} = O
stencil_coords(offset::Tuple) = map(offset_value, offset)
stencil_axis_name(dim::Integer) = string(Symbol(:i, to_subscript(Val(dim))))

function show_stencil_offset(io::IO, offset::Tuple)
    print(io, "δ(")
    join(io, stencil_coords(offset), ", ")
    print(io, ')')
end

function Base.show(io::IO, s::Stencil)
    offsets = map(offset -> sprint(show_stencil_offset, offset), s.offsets)
    print(io, "Stencil(", join(offsets, ", "), ')')
end

function stencil_bounds(coords)
    nd = length(first(coords))
    mins = fill(typemax(Int), nd)
    maxs = fill(typemin(Int), nd)
    for coord in coords
        for d in 1:nd
            mins[d] = min(mins[d], coord[d])
            maxs[d] = max(maxs[d], coord[d])
        end
    end
    return [(mins[d], maxs[d]) for d in 1:nd]
end

function stencil_slice_matches(coord, fixed::Tuple)
    length(coord) == 2 && return isempty(fixed)
    return all(coord[d+2] == fixed[d] for d in eachindex(fixed))
end

stencil_node(active::Bool) = active ? '●' : '·'

const STENCIL_ACTIVE_FACE = StyledStrings.Face(; foreground=:cyan, weight=:bold)
const STENCIL_GRID_FACE = StyledStrings.Face(; foreground=:light_black)

function stencil_face(char::Char)
    char == '●' && return STENCIL_ACTIVE_FACE
    char in ('·', '─', '│') && return STENCIL_GRID_FACE
    return nothing
end

function stencil_node_row_string(active_nodes, sepwidth::Integer=3)
    return join((stencil_node(active) for active in active_nodes), repeat("─", sepwidth))
end

stencil_edge_row_string(n::Integer, sepwidth::Integer=3) = join(fill("│", n), repeat(" ", sepwidth))

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
    sepwidth = max(3, show_x_ticks ? tickwidth + 1 : 3)
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

function stencil_style_line(line::String)
    styledline = Base.annotatedstring(line)
    runstart = firstindex(line)
    runface = nothing
    for idx in eachindex(line)
        face = stencil_face(line[idx])
        if face !== runface
            isnothing(runface) || face!(styledline, runstart:prevind(line, idx), runface)
            runstart = idx
            runface = face
        end
    end
    isnothing(runface) || face!(styledline, runstart:lastindex(line), runface)
    return styledline
end

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

function stencil_slice_lines_1d(coords, bounds; show_ticks::Bool=true, show_axis_labels::Bool=true)
    xmin, xmax = bounds[1]
    active = Set(coords)
    tickwidth = show_ticks ? maximum(x -> textwidth(string(x)), xmin:xmax) : 1
    sepwidth = max(3, show_ticks ? tickwidth + 1 : 3)
    prefixwidth = show_ticks ? tickwidth - 1 : 0
    gridwidth = stencil_grid_width(xmax - xmin + 1, sepwidth)
    lines = String[]
    if show_axis_labels
        push!(lines, stencil_centered_line(stencil_axis_name(1), prefixwidth, gridwidth))
    end
    push!(lines, rpad("", prefixwidth) * stencil_node_row_string(((x,) in active for x in xmin:xmax), sepwidth))
    show_ticks && push!(lines, stencil_xlabels_line(xmin, xmax, prefixwidth, sepwidth))
    return lines
end

function stencil_slice_lines_2d(coords, bounds, fixed::Tuple=();
                                show_x_ticks::Bool=true,
                                show_y_ticks::Bool=true,
                                reserve_y_tick_space::Bool=show_y_ticks,
                                show_x_axis_label::Bool=true,
                                show_y_axis_label::Bool=true)
    layout = stencil_slice_layout_2d(bounds; show_x_ticks, show_y_ticks, reserve_y_tick_space)
    xmin, xmax = layout.xmin, layout.xmax
    ymin, ymax = layout.ymin, layout.ymax
    active = Set((coord[1], coord[2]) for coord in coords if stencil_slice_matches(coord, fixed))
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
        push!(lines, prefix * stencil_node_row_string(((x, y) in active for x in xmin:xmax), sepwidth) * suffix)
        if row < ymax - ymin + 1
            edgeprefix = show_y_ticks ? " $(repeat(" ", labelwidth)) " : blankprefix
            # For even-height stencils, the y-axis sits between the two middle rows.
            edgesuffix = show_y_axis_label && iseven(nrows) && row == midedge ? " " * stencil_axis_name(2) : ""
            push!(lines, edgeprefix * stencil_edge_row_string(xmax - xmin + 1, sepwidth) * edgesuffix)
        end
    end
    show_x_ticks && push!(lines, stencil_xlabels_line(xmin, xmax, prefixwidth, sepwidth))
    return lines
end

function stencil_render_lines(s::Stencil; show_ticks::Bool=true, show_axis_labels::Bool=true)
    isempty(s.offsets) && return [Base.annotatedstring(sprint(show, s))]
    coords = map(stencil_coords, s.offsets)
    bounds = stencil_bounds(coords)
    nd = length(first(coords))
    if nd == 1
        return stencil_style_lines(stencil_slice_lines_1d(coords, bounds; show_ticks, show_axis_labels))
    elseif nd == 2
        return stencil_style_lines(stencil_slice_lines_2d(coords, bounds;
                                                          show_x_ticks=show_ticks,
                                                          show_y_ticks=show_ticks,
                                                          reserve_y_tick_space=show_ticks,
                                                          show_x_axis_label=show_axis_labels,
                                                          show_y_axis_label=show_axis_labels))
    elseif nd != 3
        return [Base.annotatedstring(sprint(show, s))]
    end
    ranges = [bounds[d][1]:bounds[d][2] for d in 3:length(bounds)]
    fixeds = collect(Tuple(values) for values in Iterators.product(ranges...))
    blocks = Vector{Vector{String}}()
    gap = 1
    for (i, fixed) in enumerate(fixeds)
        # Horizontal slice layouts keep x ticks on every slice, but avoid repeated y ticks.
        show_y_ticks = show_ticks && i == 1
        slicelines = stencil_slice_lines_2d(coords, bounds, fixed;
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
    isempty(s.offsets) && return show(io, s)
    ndims(s) > 3 && return show(io, s)
    print(io, "Stencil ($(ndims(s))D, $(length(s.offsets)) offsets):\n")
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
