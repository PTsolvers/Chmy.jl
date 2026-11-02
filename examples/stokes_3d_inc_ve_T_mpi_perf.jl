# NOTE: This file is currently run as multi-GPU test in the CSCS CI on ALPS. Consider making a copy before applying any changes.
using Chmy
using KernelAbstractions
using Printf
# using JSON
# using CairoMakie

# backend from the environment (CHMY_PERF_BACKEND=CUDA, the default, or AMDGPU), for the CSCS CI on GH200 and beverin
const PERF_BACKEND = get(ENV, "CHMY_PERF_BACKEND", "CUDA")
if PERF_BACKEND == "AMDGPU"
    using AMDGPU
    AMDGPU.allowscalar(false)
    backend = ROCBackend()
elseif PERF_BACKEND == "CUDA"
    using CUDA
    CUDA.allowscalar(false)
    backend = CUDABackend()
else
    error("CHMY_PERF_BACKEND must be CUDA or AMDGPU, got $PERF_BACKEND")
end

using MPI
MPI.Init()

function max_mpi(A)
    max_l = maximum(A)
    return MPI.Allreduce(max_l, MPI.MAX, MPI.COMM_WORLD)
end

@kernel inbounds = true function update_old!(T, τ, T_old, τ_old, O)
    I = @index(Global, NTuple)
    I = I + O
    T_old[I...] = T[I...]
    τ_old.xx[I...] = τ.xx[I...]
    τ_old.yy[I...] = τ.yy[I...]
    τ_old.zz[I...] = τ.zz[I...]
    τ_old.xy[I...] = τ.xy[I...]
    τ_old.xz[I...] = τ.xz[I...]
    τ_old.yz[I...] = τ.yz[I...]
end

@kernel inbounds = true function update_stress!(τ, Pr, ∇V, V, τ_old, η, η_ve, G, dt, dτ_Pr, dτ_r, g::StructuredGrid, O)
    I = @index(Global, NTuple)
    I = I + O
    ε̇xx = ∂x(V.x, g, I...)
    ε̇yy = ∂y(V.y, g, I...)
    ε̇zz = ∂z(V.z, g, I...)
    ε̇xy = 0.5 * (∂y(V.x, g, I...) + ∂x(V.y, g, I...))
    ε̇xz = 0.5 * (∂z(V.x, g, I...) + ∂x(V.z, g, I...))
    ε̇yz = 0.5 * (∂z(V.y, g, I...) + ∂y(V.z, g, I...))
    ∇V[I...] = divg(V, g, I...)
    Pr[I...] -= ∇V[I...] * η_ve * dτ_Pr
    r_τxx = -(τ.xx[I...] - τ_old.xx[I...]) / (G * dt) - τ.xx[I...] / η + 2.0 * (ε̇xx - ∇V[I...] / 3.0)
    r_τyy = -(τ.yy[I...] - τ_old.yy[I...]) / (G * dt) - τ.yy[I...] / η + 2.0 * (ε̇yy - ∇V[I...] / 3.0)
    r_τzz = -(τ.zz[I...] - τ_old.zz[I...]) / (G * dt) - τ.zz[I...] / η + 2.0 * (ε̇zz - ∇V[I...] / 3.0)
    r_τxy = -(τ.xy[I...] - τ_old.xy[I...]) / (G * dt) - τ.xy[I...] / η + 2.0 * ε̇xy
    r_τxz = -(τ.xz[I...] - τ_old.xz[I...]) / (G * dt) - τ.xz[I...] / η + 2.0 * ε̇xz
    r_τyz = -(τ.yz[I...] - τ_old.yz[I...]) / (G * dt) - τ.yz[I...] / η + 2.0 * ε̇yz
    τ.xx[I...] += r_τxx * η_ve * dτ_r
    τ.yy[I...] += r_τyy * η_ve * dτ_r
    τ.zz[I...] += r_τzz * η_ve * dτ_r
    τ.xy[I...] += r_τxy * η_ve * dτ_r
    τ.xz[I...] += r_τxz * η_ve * dτ_r
    τ.yz[I...] += r_τyz * η_ve * dτ_r
end

@kernel inbounds = true function update_velocity!(V, r_V, Pr, τ, ρgz, η_ve, νdτ, g::StructuredGrid, O)
    I = @index(Global, NTuple)
    I = I + O
    r_V.x[I...] = -∂x(Pr, g, I...) + ∂x(τ.xx, g, I...) + ∂y(τ.xy, g, I...) + ∂z(τ.xz, g, I...)
    r_V.y[I...] = -∂y(Pr, g, I...) + ∂y(τ.yy, g, I...) + ∂x(τ.xy, g, I...) + ∂z(τ.yz, g, I...)
    r_V.z[I...] = -∂z(Pr, g, I...) + ∂z(τ.zz, g, I...) + ∂x(τ.xz, g, I...) + ∂y(τ.yz, g, I...) - ρgz[I...]
    V.x[I...] += r_V.x[I...] * νdτ / η_ve
    V.y[I...] += r_V.y[I...] * νdτ / η_ve
    V.z[I...] += r_V.z[I...] * νdτ / η_ve
end

@kernel inbounds = true function update_thermal_flux!(qT, T, V, λ_ρCp, g::StructuredGrid, O)
    I = @index(Global, NTuple)
    I = I + O
    qT.x[I...] = -λ_ρCp * ∂x(T, g, I...) +
                 max(V.x[I...], 0.0) * leftx(T, I...) +
                 min(V.x[I...], 0.0) * rightx(T, I...)
    qT.y[I...] = -λ_ρCp * ∂y(T, g, I...) +
                 max(V.y[I...], 0.0) * lefty(T, I...) +
                 min(V.y[I...], 0.0) * righty(T, I...)
    qT.z[I...] = -λ_ρCp * ∂z(T, g, I...) +
                 max(V.z[I...], 0.0) * leftz(T, I...) +
                 min(V.z[I...], 0.0) * rightz(T, I...)
end

@kernel inbounds = true function update_thermal!(T, T_old, qT, dt, g::StructuredGrid, O)
    I = @index(Global, NTuple)
    I = I + O
    T[I...] = T_old[I...] - dt * divg(qT, g, I...)
end

@views function main(backend=CPU(); nxyz_l=(126, 126, 126))
    # one visible GPU per rank (e.g. Slurm --gpus-per-task=1) -> device 1, otherwise the device is picked by node-local rank
    ndev = isdefined(KernelAbstractions, :ndevices) ? KernelAbstractions.ndevices(backend) : 1
    arch = Arch(backend, MPI.COMM_WORLD, (0, 0, 0); device_id=(ndev == 1 ? 1 : nothing))
    topo = topology(arch)
    me   = global_rank(topo)
    # geometry
    lx, ly, lz = 2.0, 2.0, 2.0
    # mechanics
    η    = 1.0e1 # solid viscosity
    G    = 1.0e0 # shear modulus
    ρg   = (x=-0.0, y=0.0, z=1.0) # gravity force density
    # scales
    psc  = G
    τsc  = η/psc
    # heat
    T0    = 1.0               # initial temperature
    Ta    = 0.1               # atmospheric temperature
    λ_ρCp = 1e-4 * lz^2 / τsc # thermal diffusivity
    # numerics
    dims_l = nxyz_l
    dims_g = dims_l .* dims(topo)
    grid   = UniformGrid(arch; origin=(-lx/2, -ly/2, -lz/2), extent=(lx, ly, lz), dims=dims_g)
    # the time loop runs once per launcher, to report the throughput with and without the overlap of
    # boundary conditions and halo exchange with the computation
    launchers = (overlapped=Launcher(arch, grid; outer_width=(128, 8, 4)), plain=Launcher(arch, grid))
    nx, ny, nz = dims_g
    dx, dy, dz = spacing(grid)
    nt     = 8
    # niter  = 50nx
    niter, warmup = 110, 10
    ncheck = 2nx
    ϵ_it   = 1e-6
    ysl    = ceil(Int, length(ycenters(grid)) / 2) # for visu
    # PT params
    re_m    = 2.5π
    r       = 0.5
    lτ_re_m = min(lx, ly, lz) / re_m
    vdτ     = min(dx, dy, dz) / sqrt(ndims(grid) * 1.1)
    θ_dτ    = lτ_re_m * (r + 4 / 3) / vdτ
    dτ_r    = 1.0 / (θ_dτ + 1.0)
    νdτ     = vdτ * lτ_re_m
    dτ_Pr   = r / θ_dτ
    # allocate fields
    Pr    = Field(backend, grid, Center())
    ∇V    = Field(backend, grid, Center())
    V     = VectorField(backend, grid)
    r_V   = VectorField(backend, grid)
    τ     = TensorField(backend, grid)
    τ_old = TensorField(backend, grid)
    T     = Field(backend, grid, Center())
    T_old = Field(backend, grid, Center())
    qT    = VectorField(backend, grid)
    # Pr_v  = (me==0) ? KernelAbstractions.zeros(CPU(), Float64, size(interior(Pr)) .* dims(topo)) : nothing
    # initial conditions
    init_incl(x, y, z, x0, y0, z0, r, in, out) = ifelse((x - x0)^2 + (y - y0)^2 + (z - z0)^2 < r^2, in, out)
    ρgz = FunctionField(init_incl, grid, (Center(), Center(), Vertex()); parameters=(x0=0.0, y0=0.0, z0=0.0, r=0.1lx, in=ρg.z, out=0.0))
    η_ve = 0.0
    # boundary conditions
    bc_V = (V.x => (x=Dirichlet(), y=Neumann(), z=Neumann()),
            V.y => (x=Neumann(), y=Dirichlet(), z=Neumann()),
            V.z => (x=Neumann(), y=Neumann(), z=Dirichlet()))
    bc_T = (T => Neumann(),)
    # initial state, restored before the time loop of each launcher: the throughput depends on the state (on MI300A it drops by
    # about 20% over the time steps), so the launchers are compared on the same steps
    function init!()
        zero!(f) = fill!(parent(f), 0.0)
        foreach(zero!, (Pr, ∇V, T, T_old, V.x, V.y, V.z, r_V.x, r_V.y, r_V.z, qT.x, qT.y, qT.z))
        foreach(t -> foreach(zero!, (t.xx, t.yy, t.zz, t.xy, t.xz, t.yz)), (τ, τ_old))
        set!(T, grid, init_incl; parameters=(x0=0.0, y0=0.0, z0=0.0, r=0.1lx, in=T0, out=Ta))
        bc!(arch, grid, bc_V...; exchange=(V.x, V.y, V.z))
        bc!(arch, grid, bc_T...; exchange=T)
        return
    end
    # visualisation
    # fig = Figure(; size=(800, 600))
    # axs = (Pr = Axis(fig[1, 1][1, 1]; aspect=DataAspect(), xlabel="x", title="Pr"),
    #        Vx = Axis(fig[1, 2][1, 1]; aspect=DataAspect(), xlabel="x", title="Vx"),
    #        Vz = Axis(fig[2, 1][1, 1]; aspect=DataAspect(), xlabel="x", title="Vz"),
    #        T  = Axis(fig[2, 2][1, 1]; aspect=DataAspect(), xlabel="x", title="T"))
    # plt = (Pr = heatmap!(axs.Pr, xcenters(grid), zcenters(grid), interior(Pr)[:, ysl, :] |> Array; colormap=:turbo),
    #        Vx = heatmap!(axs.Vx, xvertices(grid), zcenters(grid), interior(V.x)[:, ysl, :] |> Array; colormap=:turbo),
    #        Vz = heatmap!(axs.Vz, xcenters(grid), zvertices(grid), interior(V.z)[:, ysl, :] |> Array; colormap=:turbo),
    #        T  = heatmap!(axs.T, xcenters(grid), zcenters(grid), interior(T)[:, ysl, :] |> Array; colormap=:turbo))
    # Colorbar(fig[1, 1][1, 2], plt.Pr)
    # Colorbar(fig[1, 2][1, 2], plt.Vx)
    # Colorbar(fig[2, 1][1, 2], plt.Vz)
    # Colorbar(fig[2, 2][1, 2], plt.T)
    # display(fig)
    KernelAbstractions.synchronize(backend)
    # action
    tic, wtime = 0.0, 0.0
    teff = Dict(name => Float64[] for name in keys(launchers))
    # GC.gc(); GC.gc(false)
    for (name, launch) in pairs(launchers), it in 1:nt
        (it == 1) && init!()
        (me==0) && (sleep(2); @printf("%s: it = %d/%d \n", name, it, nt))
        MPI.Barrier(cart_comm(topo))
        launch(arch, grid, update_old! => (T, τ, T_old, τ_old))
        # time step
        dt_diff = min(dx, dy, dz)^2 / λ_ρCp / ndims(grid) / 2.1
        dt_adv  = 0.01 * min(dx / max_mpi(abs.(interior(V.x))), dy / max_mpi(abs.(interior(V.y))), dz / max_mpi(abs.(interior(V.z)))) / ndims(grid) / 2.1 # needs 0.1 here ?!
        dt      = min(dt_diff, dt_adv)
        # rheology
        η_ve = 1.0 / (1.0 / η + 1.0 / (G * dt))
        (it > 2) && (ncheck = ceil(Int, 0.5nx))

        for iter in 1:niter
            (iter == warmup) && (MPI.Barrier(cart_comm(topo)); tic = time_ns())
            launch(arch, grid, update_stress! => (τ, Pr, ∇V, V, τ_old, η, η_ve, G, dt, dτ_Pr, dτ_r, grid))
            launch(arch, grid, update_velocity! => (V, r_V, Pr, τ, ρgz, η_ve, νdτ, grid); bc=batch(grid, bc_V...; exchange=(V.x, V.y, V.z)))
            # if it > 1
            #     launch(arch, grid, update_thermal_flux! => (qT, T, V, λ_ρCp, grid))
            #     launch(arch, grid, update_thermal! => (T, T_old, qT, dt, grid); bc=batch(grid, bc_T...; exchange=T))
            # end
            # if iter % ncheck == 0
            #     bc!(arch, grid, r_V.x => (x=Dirichlet(),), r_V.y => (y=Dirichlet(),), r_V.z => (z=Dirichlet(),))
            #     err = (Pr=max_mpi(abs.(interior(∇V))) * τsc,
            #            Vx=max_mpi(abs.(interior(r_V.x))) * lz / psc,
            #            Vy=max_mpi(abs.(interior(r_V.y))) * lz / psc,
            #            Vz=max_mpi(abs.(interior(r_V.z))) * lz / psc)
            #     (me==0) && @printf("  iter/nx=%.1f, err = [Pr=%1.3e, Vx=%1.3e, Vy=%1.3e, Vz=%1.3e] \n", iter / nx , err...)
            #     # stop if converged or error if NaN
            #     all(values(err) .< ϵ_it) && break
            #     any(.!isfinite.(values(err))) && error("simulation failed, err = $err")
            # end
        end
        wtime_it = (time_ns() - tic)
        wtime_it /= (niter - warmup)
        MPI.Barrier(cart_comm(topo))
        wtime_min = MPI.Allreduce(wtime_it, MPI.MIN, cart_comm(topo))
        wtime_max = MPI.Allreduce(wtime_it, MPI.MAX, cart_comm(topo))
        # report
        if me == 0
            nIO = 2*10 + 17
            Teff_min = nIO * sizeof(Float64) * prod(nxyz_l) / wtime_max
            Teff_max = nIO * sizeof(Float64) * prod(nxyz_l) / wtime_min
            push!(teff[name], Teff_min)
            printstyled("Performance ($name): T_eff [min max] = $(round(Teff_min, sigdigits=8)) $(round(Teff_max, sigdigits=8)) \n"; bold=true, color=:green)
        end
        # A_eff = (2*10 + 17) / 1e9 * prod(nxyz_l) * sizeof(Float64)
        # T_eff = A_eff ./ wtime_it
        # @printf("  Executed %d steps in = %1.3e sec (@ T_eff = %1.2f GB/s - device %s) \n", (niter - warmup), wtime,
        #         round(T_eff, sigdigits=6), AMDGPU.device_id())
    end
    # overlap report: median over the time steps of the T_eff (min over ranks) of each launcher
    if me == 0
        med(v) = sort(v)[cld(length(v), 2)]
        T_ovl, T_pln = med(teff[:overlapped]), med(teff[:plain])
        printstyled(@sprintf("Overlap: T_eff median overlapped = %.1f GB/s, plain = %.1f GB/s, ratio = %.2f \n", T_ovl, T_pln, T_ovl / T_pln); bold=true, color=:cyan)
    end
    # GC.gc(true)
    KernelAbstractions.synchronize(backend)
    # local postprocess
    # plt.Pr[3] = interior(Pr)[:, ysl, :] |> Array
    # plt.Vx[3] = interior(V.x)[:, ysl, :] |> Array
    # plt.Vz[3] = interior(V.z)[:, ysl, :] |> Array
    # plt.T[3]  = interior(T)[:, ysl, :] |> Array
    # # display(fig)
    # save("out_stokes3d_$me.png", fig)
    # global postprocess
    # gather!(arch, Pr_v, Pr)
    # if me == 0
    #     fig = Figure(; size=(400, 320))
    #     ax  = Axis(fig[1, 1]; aspect=DataAspect(), xlabel="x", ylabel="y", title="it = 0")
    #     plt = heatmap!(ax, Pr_v[:, ceil(Int, size(Pr_v, 2) / 2), :]; colormap=:turbo) # how to get the global grid for axes?
    #     Colorbar(fig[1, 2], plt)
    #     save("out_gather_stokes3d_$nx.png", fig)
    # end
    return
end

# input = open(JSON.parse, joinpath(@__DIR__, "params.json"))
# params = NamedTuple(Symbol.(keys(input)) .=> values(input))
# res = params.res
res = 512

main(backend; nxyz_l=(res, res, res) .- 2)

MPI.Finalize()
