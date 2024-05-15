using Chmy, Chmy.Architectures, Chmy.Grids, Chmy.Fields, Chmy.BoundaryConditions, Chmy.GridOperators, Chmy.KernelLaunch
using KernelAbstractions
using Printf
# using CairoMakie

using AMDGPU
AMDGPU.allowscalar(false)
# using CUDA
# CUDA.allowscalar(false)

using Chmy.Distributed
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

function compute(arch, grid, launch, q, C, χ, Δt, nt)
    tic = time_ns()
    for _ in 1:nt
        launch(arch, grid, compute_q! => (q, C, χ, grid))
        launch(arch, grid, update_C! => (C, q, Δt, grid); bc=batch(grid, C => Neumann(); exchange=C))
    end
    wtime = (time_ns() - tic) * 1e-9
    return wtime
end

@views function main(backend=CPU(); nxy_l=(126, 126))
    arch = Arch(backend, MPI.COMM_WORLD, (0, 0))
    topo = topology(arch)
    me   = global_rank(topo)
    # geometry
    dims_l = nxy_l
    dims_g = dims_l .* dims(topo)
    grid   = UniformGrid(arch; origin=(-2, -2), extent=(4, 4), dims=dims_g)
    launch = Launcher(arch, grid, outer_width=(128, 8))
    nx, ny = dims_g
    # physics
    χ = 1.0
    # numerics
    Δt = minimum(spacing(grid))^2 / χ / ndims(grid) / 2.1
    # allocate fields
    C = Field(backend, grid, Center())
    q = VectorField(backend, grid)
    # C_v = (me==0) ? KernelAbstractions.zeros(CPU(), Float64, size(interior(C)) .* dims(topo)) : nothing
    # initial conditions
    set!(C, grid, (x, y) -> exp(-x^2 - y^2))
    bc!(arch, grid, C => Neumann(); exchange=C)
    # visualisation
    # fig = Figure(; size=(400, 320))
    # ax  = Axis(fig[1, 1]; aspect=DataAspect(), xlabel="x", ylabel="y", title="it = 0")
    # plt = heatmap!(ax, centers(grid)..., interior(C) |> Array; colormap=:turbo)
    # Colorbar(fig[1, 2], plt)
    # action
    iters, warmup = 110, 10
    # warmup
    compute(arch, grid, launch, q, C, χ, Δt, warmup)
    # time
    for ex in 1:3
        (me==0) && (sleep(2); println("Experiment = $ex"))
        MPI.Barrier(cart_comm(topo))
        wtime = compute(arch, grid, launch, q, C, χ, Δt, (iters - warmup))
        # report
        A_eff = 7 / 1e9 * prod(nxy_l) * sizeof(Float64)
        wtime_it = wtime ./ (iters - warmup)
        T_eff = A_eff ./ wtime_it
        @printf("  Executed %d steps in = %1.3e sec (@ T_eff = %1.2f GB/s) \n", (iters - warmup), wtime, round(T_eff, sigdigits=6))
    end
    KernelAbstractions.synchronize(backend)
    # local postprocess
    # plt[3] = interior(C) |> Array
    # ax.title = "it = $nt"
    # display(fig)
    # save("out$me.png", fig)
    # global postprocess
    # gather!(arch, C_v, C)
    # if me == 0
    #     fig = Figure(; size=(400, 320))
    #     ax  = Axis(fig[1, 1]; aspect=DataAspect(), xlabel="x", ylabel="y", title="it = 0")
    #     plt = heatmap!(ax, C_v; colormap=:turbo) # how to get the global grid for axes?
    #     Colorbar(fig[1, 2], plt)
    #     save("out_gather_$nx.png", fig)
    # end
    return
end

res = 1024 * 32
main(ROCBackend(); nxy_l=(res, res) .- 2)
# main(CUDABackend(); nxy_l=(res, res) .- 2)
# main(; nxy_l=(res, res) .- 2)

MPI.Finalize()
