using Chmy
using KernelAbstractions
using Printf
using CairoMakie

# using AMDGPU
# AMDGPU.allowscalar(false)
# using CUDA
# CUDA.allowscalar(false)

using MPI
MPI.Init()

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

@views function main(backend=CPU(); nxy_l=(126, 126))
    arch = Arch(backend, MPI.COMM_WORLD, (0, 0))
    topo = topology(arch)
    me   = global_rank(topo)
    # geometry
    dims_l = nxy_l
    dims_g = dims_l .* dims(topo)
    grid   = UniformGrid(arch; origin=(-2, -2), extent=(4, 4), dims=dims_g)
    launch = Launcher(arch, grid, outer_width=(16, 8))
    nx, ny = dims_g
    # physics
    χ = 1.0
    # numerics
    Δt = minimum(spacing(grid))^2 / χ / ndims(grid) / 2.1
    # allocate fields
    C = Field(backend, grid, Center())
    q = VectorField(backend, grid)
    C_v = (me==0) ? KernelAbstractions.zeros(CPU(), Float64, size(interior(C)) .* dims(topo)) : nothing
    # initial conditions
    set!(C, grid, (x, y) -> exp(-x^2 - y^2))
    bc!(arch, grid, C => Neumann(); exchange=C)
    # visualisation
    fig = Figure(; size=(400, 320))
    ax  = Axis(fig[1, 1]; aspect=DataAspect(), xlabel="x", ylabel="y", title="it = 0")
    plt = heatmap!(ax, centers(grid)..., interior(C) |> Array; colormap=:turbo)
    Colorbar(fig[1, 2], plt)
    # action
    nt = 100
    for it in 1:nt
        (me==0) && @printf("it = %d/%d \n", it, nt)
        launch(arch, grid, compute_q! => (q, C, χ, grid))
        launch(arch, grid, update_C! => (C, q, Δt, grid); bc=batch(grid, C => Neumann(); exchange=C))
    end
    KernelAbstractions.synchronize(backend)
    # local postprocess
    # plt[3] = interior(C) |> Array
    # ax.title = "it = $nt"
    # display(fig)
    # save("out$me.png", fig)
    # global postprocess
    gather!(arch, C_v, C)
    if me == 0
        fig = Figure(; size=(400, 320))
        ax  = Axis(fig[1, 1]; aspect=DataAspect(), xlabel="x", ylabel="y", title="it = 0")
        plt = heatmap!(ax, C_v; colormap=:turbo) # how to get the global grid for axes?
        Colorbar(fig[1, 2], plt)
        save("out_gather_$nx.png", fig)
    end
    return
end

n = 128

# main(ROCBackend(); nxy_l=(n, n) .- 2)
# main(CUDABackend(); nxy_l=(n, n) .- 2)
main(; nxy_l=(n, n) .- 2)

MPI.Finalize()
