# This file shows how we can use the GeophysicalModelGenerator package to specify an initial model setup
# it will run on 1 core if you start this from the REPL and on multicores if you run this with:
# mpiexecjl -n 4 --project=. julia diffusion_2d_mpi_gmg.jl
# it requires GMG >= 0.7.8
using Chmy, Chmy.Architectures, Chmy.Grids, Chmy.Fields, Chmy.BoundaryConditions, Chmy.GridOperators, Chmy.KernelLaunch
using KernelAbstractions
using Printf
using CairoMakie
using GeophysicalModelGenerator


# using AMDGPU
# AMDGPU.allowscalar(false)
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

@views function main(backend=CPU(); nxy_l=(126, 126))
    arch = Arch(backend, MPI.COMM_WORLD, (0, 0))
    topo = topology(arch)
    me   = global_rank(topo)
    # geometry
    dims_l = nxy_l
    dims_g = dims_l .* dims(topo)
    grid   = UniformGrid(arch; origin=(-2, -2), extent=(4, 4), dims=dims_g)
    launch = Launcher(arch, grid, outer_width=(16, 8))
    
    #@info "mpi" me grid

    nx, ny = dims_g
    # physics
    χ = 1.0
    # numerics
    Δt = minimum(spacing(grid))^2 / χ / ndims(grid) / 2.1
    # allocate fields
    C = Field(backend, grid, Center())
    P = Field(backend, grid, Center(), Int32)   # phases
    
    q = VectorField(backend, grid)
    C_v = (me==0) ? KernelAbstractions.zeros(CPU(), Float64, size(interior(C)) .* dims(topo)) : nothing
    
    # use GMG to set the initial conditions
    add_box!(P,C,grid,  xlim=(-1.0,1.0), zlim=(-1.0,1.0), phase=ConstantPhase(4), T=ConstantTemp(400))

    # set BC's and updates the halo:
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
