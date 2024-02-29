using Chmy, Chmy.Architectures, Chmy.Grids, Chmy.Fields, Chmy.BoundaryConditions, Chmy.GridOperators, Chmy.KernelLaunch
using KernelAbstractions
using Printf
# using CairoMakie

using AMDGPU
AMDGPU.allowscalar(false)
# using CUDA
# CUDA.allowscalar(false)

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
    arch = Arch(backend; device_id=1)
    # geometry
    grid = UniformGrid(arch; origin=(-1, -1), extent=(2, 2), dims=nxy_l)
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
    launch_out = Launcher(arch, grid; outer_width=(128, 8))
    # visualisation
    # fig = Figure(; size=(400, 320))
    # ax  = Axis(fig[1, 1]; aspect=DataAspect(), xlabel="x", ylabel="y", title="it = 0")
    # plt = heatmap!(ax, centers(grid)..., interior(C) |> Array; colormap=:turbo)
    # Colorbar(fig[1, 2], plt)
    # display(fig)
    # action
    iters, warmup = 110, 10
    # measure perf
    l=0
    for lau in (launch, launch_out)
        l == 0 ? println("Launch") : println("Launch outer_width"); l+=1
        # warmup
        compute(arch, grid, lau, q, C, χ, Δt, warmup)
        # time
        for ex in 1:5
            println("Experiment = $ex")
            wtime = compute(arch, grid, lau, q, C, χ, Δt, (iters - warmup))
            # report
            A_eff = 6 / 1e9 * prod(nxy_l) * sizeof(Float64)
            wtime_it = wtime ./ (iters - warmup)
            T_eff = A_eff ./ wtime_it
            @printf("  Executed %d steps in = %1.3e sec (@ T_eff = %1.2f GB/s - device %s) \n", (iters - warmup), wtime,
                    round(T_eff, sigdigits=6), AMDGPU.device_id())
        end
    end
    # plt[3] = interior(C) |> Array
    # ax.title = "it = $nt"
    # display(fig)
    return
end

res = 1024 * 1
# main(ROCBackend(); nxy_l=(1022, 1022))
# main(CUDABackend(); nxy_l=(16382, 16382))
main(ROCBackend(); nxy_l=(res, res) .- 2)
# main(CUDABackend(); nxy_l=(res, res) .- 2)
# main(; nxy_l=(256, 256))
