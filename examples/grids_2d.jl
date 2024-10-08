using Chmy.Grids
using GLMakie

import .Iterators: product, flatten

grid = StructuredGrid{Tuple{Bounded,Bounded}}(UniformAxis(0.0, 1.0, 5),
                                              FunctionAxis(i -> ((i - 1) / 4)^1.5, 4))

fig = Figure(; size=(400, 450))
ax  = Axis(fig[2, 1]; aspect=DataAspect(), xlabel="x", ylabel="y", xticks=0:1, yticks=0:1)

hidespines!(ax)

ax.xticksvisible[] = false
ax.yticksvisible[] = false

ax.xticklabelsvisible[] = false
ax.yticklabelsvisible[] = false

C = Center()
V = Vertex()

xface = map(Point, product(coords(grid, (V, C))...)) |> vec
yface = map(Point, product(coords(grid, (C, V))...)) |> vec
cells = map(Point, product(coords(grid, (C, C))...)) |> vec

xyedge = map(Point, product(coords(grid, (V, V))...)) |> vec

nodes = map(Point, product(coords(grid, (V, V))...))

xedges = collect(flatten(zip(nodes[1:end-1, :], nodes[2:end, :])))
yedges = collect(flatten(zip(nodes[:, 1:end-1], nodes[:, 2:end])))

linesegments!(ax, xedges; linewidth=1.5, color=:gray)
linesegments!(ax, yedges; linewidth=1.5, color=:gray)

scatter!(ax, xface; marker=:hline, markersize=15, label=L"v_x", rotation=0)
scatter!(ax, yface; marker=:vline, markersize=15, label=L"v_y", rotation=0)
scatter!(ax, cells; marker=:xcross, markersize=10, label=L"\sigma_{xx}\,,~\sigma_{yy}")
scatter!(ax, xyedge; marker=:circle, markersize=10, label=L"\tau_{xy}")

Legend(fig[1, 1], ax, "Grid locations"; valign=:top, orientation=:horizontal, colgap=10)

fig
