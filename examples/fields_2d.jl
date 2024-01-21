using Chmy.Grids, Chmy.Fields, Chmy.GridOperators
using KernelAbstractions

function compute_q!(q, C, D, grid)
    @. q.x = -$lerpx(D, grid) * $∂x(C, grid)
    @. q.y = -$lerpy(D, grid) * $∂y(C, grid)
    return
end

function update_C!(C, q, Δt, grid)
    @. C -= Δt * $divg(q, grid)
    return
end

function update_C!(C, D, Δt, grid)
    @. C -= Δt * $divg_grad(C, D, grid)
    bc!(grid, C, Neumann())
    return
end

@kernel inbounds = true function compute_q!(q, C, D, grid)
    I = @index(Global, Cartesian)
    q.x[I] = -lerpx(D, grid, I) * ∂x(C, grid, I)
    q.y[I] = -lerpy(D, grid, I) * ∂y(C, grid, I)
end

@kernel inbounds = true function update_C!(C, q, Δt, grid)
    I = @index(Global, Cartesian)
    C[I] -= Δt * (∂x(q.x, grid, I) +
                  ∂y(q.y, grid, I))
end

MPI.Init()

dev  = CPU()
arch = Architecture(dev, MPI.COMM_WORLD)

grid = UniformGrid(arch; origin=(0, 0), extent=(1, 1), dims=(4, 4))

C = Field(dev, grid, Center())
q = VectorField(dev, grid)
D = OneField{eltype(grid)}()

Δt = minimum(spacing(grid))^2 / maximum(D) / ndims(grid) / 1.1

set!(C, grid, (x, y) -> exp(-x^2 - y^2))
set!(q, 0.0)
bc!(grid, C, NeumannBoundaryCondition())

launch = KernelLauncher(grid; with_workers=true, outer_width=2)

bc_C = (C => BoundaryConditions(grid; y=(nothing, Dirichlet(1.0))))

bc_q = (qx => BoundaryConditions(grid; x=(Dirichlet(), Dirichlet())),
        qy => BoundaryConditions(grid; y=(Dirichlet(-1.0), nothing)))



for it in 1:100
    launch(compute_q! => (q, C, D, grid); bc=bc_q)
    launch(update_C! => (C, q, Δt, grid); bc=bc_C)

    # compute_q!(dev, 256, size(grid, Vertex()))(q, C, D, grid)
    # update_C!(dev, 256, size(grid, Center()))(C, q, Δt, grid)
    # bc!(grid, C, NeumannBoundaryCondition())
end
