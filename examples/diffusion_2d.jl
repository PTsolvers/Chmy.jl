using Chmy, Chmy.Grids, Chmy.GridOperators
using KernelAbstractions
using Printf
using CairoMakie

# using AMDGPU

@kernel inbounds = true function compute_q!(q, C, χ, g::StructuredGrid, O)
    I = @index(Global, NTuple)
    I = I + O
    q.x[I...] = -χ * ∂x(C, g, I...)
    q.y[I...] = -χ * ∂y(C, g, I...)
end

@kernel inbounds = true function update_C!(C, q, Δt, g::StructuredGrid, O)
    I = @index(Global, NTuple)
    I = I + O
    C[I...] -= Δt * divg(q, g, I...)
end

@views function main(backend=CPU(); nxy=(126, 126))
    arch = Arch(backend)
    # geometry
    grid   = UniformGrid(arch; origin=(-1, -1), extent=(2, 2), dims=nxy)
    launch = Launcher(arch, grid; outer_width=(16, 8))
    # physics
    χ = 1.0
    # numerics
    Δt = minimum(spacing(grid))^2 / χ / ndims(grid) / 2.1
    # allocate fields
    C = Field(backend, grid, Center())
    q = VectorField(backend, grid)
    # initial conditions
    set!(C, grid, (_, _) -> rand())
    bc!(arch, grid, C => Neumann(); exchange=C)
    # visualisation
    fig = Figure(; size=(400, 320))
    ax  = Axis(fig[1, 1]; aspect=DataAspect(), xlabel="x", ylabel="y", title="it = 0")
    plt = heatmap!(ax, centers(grid)..., interior(C) |> Array; colormap=:turbo)
    Colorbar(fig[1, 2], plt)
    display(fig)
    # action
    nt = 100
    for it in 1:nt
        @printf("it = %d/%d \n", it, nt)
        launch(arch, grid, compute_q! => (q, C, χ, grid))
        launch(arch, grid, update_C! => (C, q, Δt, grid); bc=batch(grid, C => Neumann(); exchange=C))
    end
    KernelAbstractions.synchronize(backend)
    plt[3] = interior(C) |> Array
    ax.title = "it = $nt"
    display(fig)
    return
end

n = 128
# main(ROCBackend(); nxy=(n, n) .- 2)
main(; nxy=(n, n) .- 2)
