using Chmy, Chmy.Architectures, Chmy.Grids, Chmy.Fields, Chmy.BoundaryConditions, Chmy.GridOperators, Chmy.KernelLaunch
using KernelAbstractions
using CairoMakie
using Printf

# using AMDGPU
# AMDGPU.allowscalar(false)

using Chmy.Distributed
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
    τ_old.xy[I...] = τ.xy[I...]
end

@kernel inbounds = true function update_stress!(τ, Pr, ∇V, V, τ_old, η, η_ve, G, dt, dτ_Pr, dτ_r, g::StructuredGrid, O)
    I = @index(Global, NTuple)
    I = I + O
    ε̇xx = ∂x(V.x, g, I...)
    ε̇yy = ∂y(V.y, g, I...)
    ε̇xy = 0.5 * (∂y(V.x, g, I...) + ∂x(V.y, g, I...))
    ∇V[I...] = divg(V, g, I...)
    Pr[I...] -= ∇V[I...] * η_ve * dτ_Pr
    r_τxx = -(τ.xx[I...] - τ_old.xx[I...]) / (G * dt) - τ.xx[I...] / η + 2.0 * (ε̇xx - ∇V[I...] / 3.0)
    r_τyy = -(τ.yy[I...] - τ_old.yy[I...]) / (G * dt) - τ.yy[I...] / η + 2.0 * (ε̇yy - ∇V[I...] / 3.0)
    r_τxy = -(τ.xy[I...] - τ_old.xy[I...]) / (G * dt) - τ.xy[I...] / η + 2.0 * ε̇xy
    τ.xx[I...] += r_τxx * η_ve * dτ_r
    τ.yy[I...] += r_τyy * η_ve * dτ_r
    τ.xy[I...] += r_τxy * η_ve * dτ_r
end

@kernel inbounds = true function update_velocity!(V, r_V, Pr, τ, ρgy, η_ve, νdτ, g::StructuredGrid, O)
    I = @index(Global, NTuple)
    I = I + O
    r_V.x[I...] = -∂x(Pr, g, I...) + ∂x(τ.xx, g, I...) + ∂y(τ.xy, g, I...)
    r_V.y[I...] = -∂y(Pr, g, I...) + ∂y(τ.yy, g, I...) + ∂x(τ.xy, g, I...) - ρgy[I...]
    V.x[I...] += r_V.x[I...] * νdτ / η_ve
    V.y[I...] += r_V.y[I...] * νdτ / η_ve
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
end

@kernel inbounds = true function update_thermal!(T, T_old, qT, dt, g::StructuredGrid, O)
    I = @index(Global, NTuple)
    I = I + O
    T[I...] = T_old[I...] - dt * divg(qT, g, I...)
end

@views function main(backend=CPU(); nxy::Int=126)
    arch = Arch(backend, MPI.COMM_WORLD, (0, 0))
    topo = topology(arch)
    me   = global_rank(topo)
    # geometry
    lx, ly = 2.0, 2.0
    # mechanics
    η    = 1.0e1 # solid viscosity
    G    = 1.0e0 # shear modulus
    ρg   = (x=-0.0, y=1.0) # gravity force density
    # scales
    psc  = G
    τsc  = η/psc
    # heat
    T0    = 1.0               # initial temperature
    Ta    = 0.1               # atmospheric temperature
    λ_ρCp = 1e-4 * ly^2 / τsc # thermal diffusivity
    # numerics
    nx = ny = nxy
    grid   = UniformGrid(arch; origin=(-lx/2, -ly/2), extent=(lx, ly), dims=(nx, ny))
    dx, dy = spacing(grid, Center(), 1, 1)
    nt     = 4
    niter  = 50nx
    ncheck = 2nx
    ϵ_it   = 1e-6
    # PT params
    re_m    = 2.3π
    r       = 0.5
    lτ_re_m = min(lx, ly) / re_m
    vdτ     = min(dx, dy) / sqrt(ndims(grid) * 1.1)
    θ_dτ    = lτ_re_m * (r + 4 / 3) / vdτ
    dτ_r    = 1.0 / (θ_dτ + 1.0)
    νdτ     = vdτ * lτ_re_m
    dτ_Pr   = r / θ_dτ
    # allocate fields
    Pr    = Field(backend, grid, Center())
    ∇V    = Field(backend, grid, Center())
    ρgy   = Field(backend, grid, (Center(), Vertex()))
    V     = VectorField(backend, grid)
    r_V   = VectorField(backend, grid)
    τ     = TensorField(backend, grid)
    τ_old = TensorField(backend, grid)
    T     = Field(backend, grid, Center())
    T_old = Field(backend, grid, Center())
    qT    = VectorField(backend, grid)
    Pr_v  = KernelAbstractions.zeros(CPU(), Float64, size(interior(Pr)) .* dims(topo))
    Pr_v  = (me==0) ? KernelAbstractions.zeros(CPU(), Float64, size(interior(Pr)) .* dims(topo)) : nothing
    # initial conditions
    init_incl(x, y, x0, y0, r, in, out) = ifelse((x - x0)^2 + (y - y0)^2 < r^2, in, out)
    set!(ρgy, grid, init_incl; parameters=(x0=0.0, y0=0.0, r=0.1lx, in=ρg.y, out=0.0))
    set!(T, grid, init_incl; parameters=(x0=0.0, y0=0.0, r=0.1lx, in=T0, out=Ta))
    η_ve = 0.0
    launch = Launcher(arch, grid; outer_width=(16, 8))
    # boundary conditions
    bc_V = (V.x => (x=Dirichlet(), y=Neumann()),
            V.y => (x=Neumann(), y=Dirichlet()))
    bc_T = (T => Neumann(),)
    # bc_qT = (qT.x => (x=Dirichlet(),),
    #          qT.y => (y=Dirichlet(),))
    bc!(arch, grid, bc_T...)
    # visualisation
    fig = Figure(; size=(800, 600))
    axs = (Pr = Axis(fig[1, 1][1, 1]; aspect=DataAspect(), xlabel="x", title="Pr"),
           Vx = Axis(fig[1, 2][1, 1]; aspect=DataAspect(), xlabel="x", title="Vx"),
           Vy = Axis(fig[2, 1][1, 1]; aspect=DataAspect(), xlabel="x", title="Vy"),
           T  = Axis(fig[2, 2][1, 1]; aspect=DataAspect(), xlabel="x", title="T"))
    plt = (Pr = heatmap!(axs.Pr, centers(grid)..., interior(Pr) |> Array; colormap=:turbo),
           Vx = heatmap!(axs.Vx, xvertices(grid), ycenters(grid), interior(V.x) |> Array; colormap=:turbo),
           Vy = heatmap!(axs.Vy, xcenters(grid), yvertices(grid), interior(V.y) |> Array; colormap=:turbo),
           T  = heatmap!(axs.T, centers(grid)..., interior(T) |> Array; colormap=:turbo))
    Colorbar(fig[1, 1][1, 2], plt.Pr)
    Colorbar(fig[1, 2][1, 2], plt.Vx)
    Colorbar(fig[2, 1][1, 2], plt.Vy)
    Colorbar(fig[2, 2][1, 2], plt.T)
    # display(fig)
    # action
    for it in 1:nt
        (me==0) && @printf("it = %d/%d \n", it, nt)
        launch(arch, grid, update_old! => (T, τ, T_old, τ_old))
        # time step
        dt_diff = min(dx, dy)^2 / λ_ρCp / ndims(grid) / 2.1
        dt_adv  = 0.1 * min(dx / max_mpi(abs.(interior(V.x))), dy / max_mpi(abs.(interior(V.y)))) / ndims(grid) / 2.1
        dt      = min(dt_diff, dt_adv)
        # rheology
        η_ve = 1.0 / (1.0 / η + 1.0 / (G * dt))
        (it > 2) && (ncheck = ceil(Int, 0.5nx))
        for iter in 1:niter
            launch(arch, grid, update_stress! => (τ, Pr, ∇V, V, τ_old, η, η_ve, G, dt, dτ_Pr, dτ_r, grid))
            launch(arch, grid, update_velocity! => (V, r_V, Pr, τ, ρgy, η_ve, νdτ, grid); bc=batch(grid, bc_V...; exchange=(V.x, V.y)))
            if it > 1
                # launch(arch, grid, update_thermal_flux! => (qT, T, V, λ_ρCp, grid); bc=batch(grid, bc_qT...))
                # launch(arch, grid, update_thermal! => (T, T_old, qT, dt, grid); bc=batch(grid; exchange=T))
                launch(arch, grid, update_thermal_flux! => (qT, T, V, λ_ρCp, grid))
                launch(arch, grid, update_thermal! => (T, T_old, qT, dt, grid); bc=batch(grid, bc_T...; exchange=T))
            end
            if iter % ncheck == 0
                bc!(arch, grid, r_V.x => (x=Dirichlet(),), r_V.y => (y=Dirichlet(),))
                err = (Pr=max_mpi(abs.(interior(∇V))) * τsc,
                       Vx=max_mpi(abs.(interior(r_V.x))) * ly / psc,
                       Vy=max_mpi(abs.(interior(r_V.y))) * ly / psc)
                (me==0) && @printf("  iter/nx=%.1f, err = [Pr=%1.3e, Vx=%1.3e, Vy=%1.3e] \n", iter / nx, err...)
                # stop if converged or error if NaN
                all(values(err) .< ϵ_it) && break
                any(.!isfinite.(values(err))) && error("simulation failed, err = $err")
            end
        end
    end
    KernelAbstractions.synchronize(backend)
    # local postprocess
    # plt.Pr[3] = interior(Pr) |> Array
    # plt.Vx[3] = interior(V.x) |> Array
    # plt.Vy[3] = interior(V.y) |> Array
    # plt.T[3] = interior(T) |> Array
    # # display(fig)
    # save("out_stokes2d_$me.png", fig)
    # global postprocess
    gather!(arch, Pr_v, Pr)
    if me == 0
        fig = Figure(; size=(400, 320))
        ax  = Axis(fig[1, 1]; aspect=DataAspect(), xlabel="x", ylabel="y", title="it = 0")
        plt = heatmap!(ax, Pr_v; colormap=:turbo) # how to get the global grid for axes?
        Colorbar(fig[1, 2], plt)
        save("out_gather_stokes2d.png", fig)
    end
    return
end

# main(ROCBackend(); nxy=254)
main(; nxy=254)

MPI.Finalize()
