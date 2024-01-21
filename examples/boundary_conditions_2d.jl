using Chmy, Chmy.Grids, Chmy.Fields, Chmy.BoundaryConditions, Chmy.GridOperators
using KernelAbstractions
using CUDA
using CairoMakie

@kernel inbounds = true function compute_q!(q, C, χ, g::StructuredGrid)
    I = @index(Global, NTuple)
    q.x[I...] = -χ * ∂x(C, g, I...)
    q.y[I...] = -χ * ∂y(C, g, I...)
end

@kernel inbounds = true function update_C!(C, q, Δt, g::StructuredGrid)
    I = @index(Global, NTuple)
    C[I...] -= Δt * divg(q, g, I...)
end

@views function main(backend=CPU())
    # geometry
    grid = UniformGrid(; origin=(0, 0), extent=(1, 1), dims=(4096, 4096))
    # physics
    χ = 1.0
    # numerics
    Δt = minimum(spacing(grid, Center(), 1, 1))^2 / χ / ndims(grid) / 2.1
    # allocate fields
    C = Field(backend, grid, Center())
    q = VectorField(backend, grid)
    # initial conditions
    set!(C, grid, (_, _) -> rand())
    bc!(grid, C => Neumann())
    # boundary conditions
    bc = (q.x => (x=Dirichlet(),),
          q.y => (y=Dirichlet(),))

    # visualisation
    fig = Figure(; size=(400, 350))
    ax  = Axis(fig[1, 1][1, 1]; aspect=DataAspect(), xlabel="x", ylabel="y", title="it = 0")
    plt = heatmap!(ax, coords(grid, Center())..., interior(C) |> Array; colormap=:turbo)
    Colorbar(fig[1, 1][1, 2], plt)
    display(fig)
    # action
    @time begin
        for it in 1:1000
            compute_q!(backend, 256, size(grid, Vertex()))(q, C, χ, grid)
            bc!(grid, bc...)
            update_C!(backend, 256, size(grid, Center()))(C, q, Δt, grid)
        end
        KernelAbstractions.synchronize(backend)
    end
    plt[3] = interior(C) |> Array
    ax.title = "it = 1000"
    display(fig)

    return
end

main(CUDABackend())
