using Chmy, Chmy.Architectures, Chmy.Grids, Chmy.Fields, Chmy.BoundaryConditions, Chmy.GridOperators, Chmy.KernelLaunch
using KernelAbstractions
# using AMDGPU
using CairoMakie

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

@views function main(backend=CPU())
    arch = Arch(backend)
    # geometry
    grid = UniformGrid(arch; origin=(-1, -1), extent=(2, 2), dims=(1022, 1022))
    # physics
    χ = 1.0
    # numerics
    Δt = minimum(spacing(grid, Center(), 1, 1))^2 / χ / ndims(grid) / 2.1
    # allocate fields
    C = Field(backend, grid, Center())
    q = VectorField(backend, grid)
    # initial conditions
    set!(C, grid, (_, _) -> rand())
    bc!(arch, grid, C => Neumann(); exchange=C)
    launch = Launcher(arch, grid)
    @show typeof(launch)
    # visualisation
    fig = Figure(; size=(400, 320))
    ax  = Axis(fig[1, 1]; aspect=DataAspect(), xlabel="x", ylabel="y", title="it = 0")
    plt = heatmap!(ax, centers(grid)..., interior(C) |> Array; colormap=:turbo)
    Colorbar(fig[1, 2], plt)
    display(fig)
    # action
    nt = 100
    for it in 1:nt
        @time launch(arch, grid, compute_q! => (q, C, χ, grid))
        @time launch(arch, grid, update_C! => (C, q, Δt, grid); bc=batch(grid, C => Neumann(); exchange=C))
    end
    KernelAbstractions.synchronize(backend)
    plt[3] = interior(C) |> Array
    ax.title = "it = $nt"
    display(fig)
    return
end

# main(ROCBackend())
main()
