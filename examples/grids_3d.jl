using Chmy
using KernelAbstractions
using GLMakie

import .Iterators: product, flatten

grid = UniformGrid(Arch(CPU()); origin=(0, 0, 0), extent=(1, 1, 1), dims=(1, 1, 1))

fig = Figure(; size=(400, 450))
ax  = Axis3(fig[2, 1]; aspect=:data, xlabel="x", ylabel="y", zlabel="z")

hidespines!(ax)

ax.xticksvisible[] = false
ax.yticksvisible[] = false
ax.zticksvisible[] = false

ax.xticklabelsvisible[] = false
ax.yticklabelsvisible[] = false
ax.zticklabelsvisible[] = false

ax.xlabeloffset[] = 10
ax.ylabeloffset[] = 10
ax.zlabeloffset[] = 10

C = Center()
V = Vertex()

xface = map(Point, product(coords(grid, (V, C, C))...)) |> vec
yface = map(Point, product(coords(grid, (C, V, C))...)) |> vec
zface = map(Point, product(coords(grid, (C, C, V))...)) |> vec
cells = map(Point, product(coords(grid, (C, C, C))...)) |> vec

xyedge = map(Point, product(coords(grid, (V, V, C))...)) |> vec
xzedge = map(Point, product(coords(grid, (V, C, V))...)) |> vec
yzedge = map(Point, product(coords(grid, (C, V, V))...)) |> vec

nodes = map(Point, product(coords(grid, (V, V, V))...))

xedges = collect(flatten(zip(nodes[1:end-1, :, :], nodes[2:end, :, :])))
yedges = collect(flatten(zip(nodes[:, 1:end-1, :], nodes[:, 2:end, :])))
zedges = collect(flatten(zip(nodes[:, :, 1:end-1], nodes[:, :, 2:end])))

linesegments!(ax, xedges; linewidth=1.5, color=:gray)
linesegments!(ax, yedges; linewidth=1.5, color=:gray)
linesegments!(ax, zedges; linewidth=1.5, color=:gray)

scatter!(ax, xface; marker=:hline, markersize=15, label=L"v_x", rotation=0)
scatter!(ax, yface; marker=:hline, markersize=15, label=L"v_y", rotation=0)
scatter!(ax, zface; marker=:vline, markersize=15, label=L"v_z", rotation=0)
scatter!(ax, cells; marker=:xcross, markersize=10, label=L"\sigma_{xx}\,,~\sigma_{yy}\,,~\sigma_{zz}")
scatter!(ax, xyedge; marker=:circle, markersize=10, label=L"\tau_{xy}")
scatter!(ax, xzedge; marker=:circle, markersize=10, label=L"\tau_{xz}")
scatter!(ax, yzedge; marker=:circle, markersize=10, label=L"\tau_{yz}")

Legend(fig[1, 1], ax, "Grid locations"; valign=:top, orientation=:horizontal, colgap=10)

fig
