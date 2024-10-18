using Chmy
using KernelAbstractions
using Printf
using CairoMakie

using Metal

@kernel inbounds = true function compute_q!(q, C, χ, g::StructuredGrid, O)
    I = @index(Global, NTuple)
    I = I + O
    q.x[I...] = -χ * ∂x(C, g, I...)
end

@kernel inbounds = true function update_C!(C, q, Δt, g::StructuredGrid, O)
    I = @index(Global, NTuple)
    I = I + O
    C[I...] -= Δt * divg(q, g, I...)
end

@views function main(backend=CPU(); nx=(32, ))
    arch = Arch(backend)
    # geometry
    grid = UniformGrid(arch; origin=(-1f0, ), extent=(2f0, ), dims=nx)
    launch = Launcher(arch, grid; outer_width=(4, ))
    # physics
    χ = 1.0f0
    # numerics
    Δt = minimum(spacing(grid))^2 / χ / ndims(grid) / 2.1f0
    nt = 100
    # allocate fields
    C = Field(backend, grid, Center())
    q = VectorField(backend, grid)
    # initial conditions
    set!(C, rand(Float32, size(C)))
    bc!(arch, grid, C => Neumann())
    # visualisation
    fig = Figure(; size=(400, 320))
    ax  = Axis(fig[1, 1]; xlabel="x", ylabel="y", title="it = 0")
    plt = lines!(ax, centers(grid)..., interior(C) |> Array)
    display(fig)
    # action
    for it in 1:nt
        @printf("it = %d/%d \n", it, nt)
        launch(arch, grid, compute_q! => (q, C, χ, grid))
        launch(arch, grid, update_C! => (C, q, Δt, grid); bc=batch(grid, C => Neumann()))
    end
    KernelAbstractions.synchronize(backend)
    plt[2] = interior(C) |> Array
    ax.title = "it = $nt"
    display(fig)
    return
end

n = 64

main(MetalBackend(); nx=(n, ) .- 2)
