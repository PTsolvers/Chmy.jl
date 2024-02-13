using Chmy, Chmy.Architectures, Chmy.Grids, Chmy.Fields, Chmy.BoundaryConditions, Chmy.GridOperators
using KernelAbstractions
using AMDGPU
AMDGPU.allowscalar(false)
using CairoMakie
using Printf

@kernel inbounds = true function update_old!(T, τ, T_old, τ_old)
    I = @index(Global, NTuple)
    T_old[I...] = T[I...]
    τ_old.xx[I...] = τ.xx[I...]
    τ_old.yy[I...] = τ.yy[I...]
    τ_old.zz[I...] = τ.zz[I...]
    τ_old.xy[I...] = τ.xy[I...]
    τ_old.xz[I...] = τ.xz[I...]
    τ_old.yz[I...] = τ.yz[I...]
end

@kernel inbounds = true function update_stress!(τ, Pr, ∇V, V, τ_old, η, η_ve, G, dt, dτ_Pr, dτ_r, g::StructuredGrid)
    I = @index(Global, NTuple)
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

@kernel inbounds = true function update_velocity!(V, r_V, Pr, τ, ρgz, η_ve, νdτ, g::StructuredGrid)
    I = @index(Global, NTuple)
    r_V.x[I...] = -∂x(Pr, g, I...) + ∂x(τ.xx, g, I...) + ∂y(τ.xy, g, I...) + ∂z(τ.xz, g, I...)
    r_V.y[I...] = -∂y(Pr, g, I...) + ∂y(τ.yy, g, I...) + ∂x(τ.xy, g, I...) + ∂z(τ.yz, g, I...)
    r_V.z[I...] = -∂z(Pr, g, I...) + ∂z(τ.zz, g, I...) + ∂x(τ.xz, g, I...) + ∂y(τ.yz, g, I...) - ρgz[I...]
    V.x[I...] += r_V.x[I...] * νdτ / η_ve
    V.y[I...] += r_V.y[I...] * νdτ / η_ve
    V.z[I...] += r_V.z[I...] * νdτ / η_ve
end

@kernel inbounds = true function update_thermal_flux!(qT, T, V, λ_ρCp, g::StructuredGrid)
    I = @index(Global, NTuple)
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

@kernel inbounds = true function update_thermal!(T, T_old, qT, dt, g::StructuredGrid)
    I = @index(Global, NTuple)
    T[I...] = T_old[I...] - dt * divg(qT, g, I...)
end

@views function main(backend=CPU())
    arch = Arch(backend)
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
    λ_ρCp = 1e-4 * ly^2 / τsc # thermal diffusivity
    # numerics
    nx = ny = nz = 319
    grid   = UniformGrid(arch; origin=(-lx/2, -ly/2, -lz/2), extent=(lx, ly, lz), dims=(nx, ny, nz))
    dx, dy, dz = spacing(grid, Center(), 1, 1, 1)
    nt     = 2
    niter  = 50nx
    ncheck = 2nx
    ϵ_it   = 1e-6
    ysl    = ceil(Int, ny / 2) # for visu
    # PT params
    re_m    = 2.3π
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
    ρgz   = Field(backend, grid, (Center(), Center(), Vertex()))
    V     = VectorField(backend, grid)
    r_V   = VectorField(backend, grid)
    τ     = TensorField(backend, grid)
    τ_old = TensorField(backend, grid)
    T     = Field(backend, grid, Center())
    T_old = Field(backend, grid, Center())
    qT    = VectorField(backend, grid)
    # initial conditions
    init_incl(x, y, z, x0, y0, z0, r, in, out) = ifelse((x - x0)^2 + (y - y0)^2 + (z - z0)^2 < r^2, in, out)
    set!(ρgz, grid, init_incl; parameters=(x0=0.0, y0=0.0, z0=0.0, r=0.1lx, in=ρg.z, out=0.0))
    set!(T, grid, init_incl; parameters=(x0=0.0, y0=0.0, z0=0.0, r=0.1lx, in=T0, out=Ta))
    η_ve = 0.0
    bc!(arch, grid, T => Neumann())
    # boundary conditions
    bc_qT = (qT.x => (x=Dirichlet(),),
             qT.y => (y=Dirichlet(),),
             qT.z => (z=Dirichlet(),))
    bc_V = (V.x => (x=Dirichlet(), y=Neumann(), z=Neumann()),
            V.y => (x=Neumann(), y=Dirichlet(), z=Neumann()),
            V.z => (x=Neumann(), y=Neumann(), z=Dirichlet()))
    # visualisation
    fig = Figure(; size=(800, 600))
    axs = (Pr = Axis(fig[1, 1][1, 1]; aspect=DataAspect(), xlabel="x", title="Pr"),
           Vx = Axis(fig[1, 2][1, 1]; aspect=DataAspect(), xlabel="x", title="Vx"),
           Vz = Axis(fig[2, 1][1, 1]; aspect=DataAspect(), xlabel="x", title="Vz"),
           T  = Axis(fig[2, 2][1, 1]; aspect=DataAspect(), xlabel="x", title="T"))
    plt = (Pr = heatmap!(axs.Pr, xcenters(grid), zcenters(grid), interior(Pr)[:, ysl, :] |> Array; colormap=:turbo),
           Vx = heatmap!(axs.Vx, xvertices(grid), zcenters(grid), interior(V.x)[:, ysl, :] |> Array; colormap=:turbo),
           Vz = heatmap!(axs.Vz, xcenters(grid), zvertices(grid), interior(V.z)[:, ysl, :] |> Array; colormap=:turbo),
           T  = heatmap!(axs.T, xcenters(grid), zcenters(grid), interior(T)[:, ysl, :] |> Array; colormap=:turbo))
    Colorbar(fig[1, 1][1, 2], plt.Pr)
    Colorbar(fig[1, 2][1, 2], plt.Vx)
    Colorbar(fig[2, 1][1, 2], plt.Vz)
    Colorbar(fig[2, 2][1, 2], plt.T)
    display(fig)
    # action
    @time begin
        for it in 1:nt
            @printf("it = %d/%d \n", it, nt)
            update_old!(backend, 256, size(grid, Vertex()))(T, τ, T_old, τ_old)
            # time step
            dt_diff = min(dx, dy, dz)^2 / λ_ρCp / ndims(grid) / 2.1
            dt_adv  = 0.1 * min(dx / maximum(abs.(interior(V.x))), dy / maximum(abs.(interior(V.y))), dz / maximum(abs.(interior(V.z)))) / ndims(grid) / 2.1 # needs 0.1 here ?!
            dt      = min(dt_diff, dt_adv)
            # rheology
            η_ve = 1.0 / (1.0 / η + 1.0 / (G * dt))
            (it > 2) && (ncheck = ceil(Int, 0.5nx))
            for iter in 1:niter
                update_stress!(backend, 256, size(grid, Vertex()))(τ, Pr, ∇V, V, τ_old, η, η_ve, G, dt, dτ_Pr, dτ_r, grid)
                update_velocity!(backend, 256, size(grid, Vertex()))(V, r_V, Pr, τ, ρgz, η_ve, νdτ, grid)
                bc!(arch, grid, bc_V...)
                if it > 1
                    update_thermal_flux!(backend, 256, size(grid, Vertex()))(qT, T, V, λ_ρCp, grid)
                    bc!(arch, grid, bc_qT...)
                    update_thermal!(backend, 256, size(grid, Center()))(T, T_old, qT, dt, grid)
                end
                if iter % ncheck == 0
                    bc!(arch, grid, r_V.x => (x=Dirichlet(),), r_V.y => (y=Dirichlet(),), r_V.z => (z=Dirichlet(),))
                    err = (Pr=maximum(abs.(interior(∇V))) * τsc,
                           Vx=maximum(abs.(interior(r_V.x))) * ly / psc,
                           Vy=maximum(abs.(interior(r_V.y))) * ly / psc,
                           Vz=maximum(abs.(interior(r_V.z))) * ly / psc)
                    @printf("  iter/nx=%.1f, err = [Pr=%1.3e, Vx=%1.3e, Vy=%1.3e, Vz=%1.3e] \n", iter / nx , err...)
                    # stop if converged or error if NaN
                    all(values(err) .< ϵ_it) && break
                    any(.!isfinite.(values(err))) && error("simulation failed, err = $err")
                end
            end
        end
        KernelAbstractions.synchronize(backend)
    end
    plt.Pr[3] = interior(Pr)[:, ysl, :] |> Array
    plt.Vx[3] = interior(V.x)[:, ysl, :] |> Array
    plt.Vz[3] = interior(V.z)[:, ysl, :] |> Array
    plt.T[3]  = interior(T)[:, ysl, :] |> Array
    display(fig)
    return
end

main(ROCBackend())
# main()
