# 2D Stokes solver with variable viscosity solved with the accelerate pseudo-transient method 
using Chmy, Chmy.Architectures, Chmy.Grids, Chmy.Fields, Chmy.BoundaryConditions, Chmy.GridOperators, Chmy.KernelLaunch
using KernelAbstractions
using Printf
using CairoMakie

# using AMDGPU
# AMDGPU.allowscalar(false)
#using CUDA
#CUDA.allowscalar(false)

Base.@propagate_inbounds maxloc_c(A, ix, iy)  = max(A[ix, iy], A[ix-1, iy], A[ix+1, iy], A[ix, iy-1], A[ix, iy+1])
Base.@propagate_inbounds maxloc_v(A, ix, iy)  = max(A[ix, iy], A[ix+1, iy], A[ix, iy+1], A[ix+1, iy+1])
Base.@propagate_inbounds maxloc_vc(A, ix, iy) = max(A[ix-1, iy], A[ix+1, iy], A[ix-1, iy+1], A[ix+1, iy+1], A[ix-1, iy-1], A[ix+1, iy-1])
Base.@propagate_inbounds maxloc_cv(A, ix, iy) = max(A[ix, iy-1], A[ix, iy+1], A[ix-1, iy-1], A[ix-1, iy+1], A[ix+1, iy-1], A[ix+1, iy+1],)

@kernel inbounds = true function update_stress!(τ, Pr, ∇V, V, ηc, ηnum, dτ_Pr, dτ_r, g::StructuredGrid, O)
    I = @index(Global, Cartesian)
    I = I + O
    ε̇xx = ∂x(V.x, g, I)
    ε̇yy = ∂y(V.y, g, I)
    ε̇xy = 0.5 * (∂y(V.x, g, I) + ∂x(V.y, g, I))
    ∇V[I] = divg(V, g, I)
    Pr[I] -= ∇V[I] * ηc[I] * dτ_Pr
    r_τxx = - τ.xx[I] / ηc[I] + 2.0 * (ε̇xx - ∇V[I] / 3.0)
    r_τyy = - τ.yy[I] / ηc[I] + 2.0 * (ε̇yy - ∇V[I] / 3.0)
    r_τxy = - τ.xy[I] / lerp(ηc, location(τ.xy), g, I)  + 2.0 * ε̇xy

    η_numeric = ηnum[I]
    τ.xx[I] += r_τxx * η_numeric * dτ_r
    τ.yy[I] += r_τyy * η_numeric * dτ_r

    η_numeric = lerp(ηnum, location(τ.xy), g, I)
    τ.xy[I] += r_τxy * η_numeric * dτ_r
end


@kernel inbounds = true function compute_η_numeric!(η_numeric, ηc, g::StructuredGrid, O)
    I = @index(Global, Cartesian)
    I = I + O
    η_numeric[I] = maxloc_c(ηc, I[1], I[2])
end

@kernel inbounds = true function update_velocity!(V, r_V, Pr, τ, ρgy, ηnum, νdτ, g::StructuredGrid, O)
    I = @index(Global, Cartesian)
    I = I + O
    r_V.x[I] = -∂x(Pr, g, I) + ∂x(τ.xx, g, I) + ∂y(τ.xy, g, I)
    r_V.y[I] = -∂y(Pr, g, I) + ∂y(τ.yy, g, I) + ∂x(τ.xy, g, I) - ρgy[I]

    η_numeric = lerp(ηnum, location(V.x), g, I)
    V.x[I] += r_V.x[I] * νdτ / η_numeric
    η_numeric = lerp(ηnum, location(V.y), g, I)
    V.y[I] += r_V.y[I] * νdτ / η_numeric
end

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

function smooth!(Csmooth, C, q, grid, arch, launch, nsteps)
    Δt = minimum(spacing(grid))^2/4.0
    Csmooth.data .= C.data
    for i=1:nsteps
        launch(arch, grid, compute_q! => (q, Csmooth, 1.0, grid))
        launch(arch, grid, update_C! => (Csmooth, q, Δt, grid); bc=batch(grid, Csmooth => Neumann(); exchange=Csmooth))
    end
end

function stokes_iterations(launch, arch, grid, ds, ds_PT, bc_V, niter, ncheck, ϵ_it, τsc, ly, psc)
    Pr, ∇V, V, r_V, τ, ηc, ρgy, ηnum = ds
    dτ_Pr, dτ_r, νdτ, _ = ds_PT
    
    # Compute numeric viscosity
    launch(arch, grid, compute_η_numeric! => (ηnum, ηc, grid))

    for iter in 1:niter
        launch(arch, grid, update_stress! => (τ, Pr, ∇V, V, ηc, ηnum, dτ_Pr, dτ_r, grid))
        launch(arch, grid, update_velocity! => (V, r_V, Pr, τ, ρgy, ηnum, νdτ, grid); bc=batch(grid, bc_V...; exchange=V)) # bc!(arch, grid, bc_V...)
        if iter % ncheck == 0
            bc!(arch, grid, r_V.x => (x=Dirichlet(),), r_V.y => (y=Dirichlet(),))
            err = (Pr=maximum(abs.(interior(∇V))) * τsc,
                   Vx=maximum(abs.(interior(r_V.x))) * ly / psc,
                   Vy=maximum(abs.(interior(r_V.y))) * ly / psc)
            @printf("  iter/nx=%.1f, err = [Pr=%1.3e, Vx=%1.3e, Vy=%1.3e] \n", iter / nx, err...)
            # stop if converged or error if NaN
            all(values(err) .< ϵ_it) && break
            any(.!isfinite.(values(err))) && error("simulation failed, err = $err")
        end
    end
end

function initialize_stokes_fields(backend, grid, ρgy)
    Pr    = Field(backend, grid, Center())
    ∇V    = Field(backend, grid, Center())
    V     = VectorField(backend, grid)
    r_V   = VectorField(backend, grid)
    τ     = TensorField(backend, grid)
    ηc    = Field(backend, grid, Center())
    ηnum  = Field(backend, grid, Center())
    q     = VectorField(backend, grid)
    return (; Pr, ∇V, V, r_V, τ, ηc, ρgy, ηnum, q)
end

function initialize_PT_params(grid; vdτ_fac=2.2, r=0.5, re_m=2.3π)
    # PT params
    lx, ly  = grid.axes[1].extent, grid.axes[2].extent 
    lτ_re_m = min(lx, ly) / re_m
    vdτ     = minimum(spacing(grid)) / sqrt(ndims(grid) * vdτ_fac)
    θ_dτ    = lτ_re_m * (r + 4 / 3) / vdτ
    dτ_r    = 1.0 / (θ_dτ + 1.0)
    νdτ     = vdτ * lτ_re_m
    dτ_Pr   = r / θ_dτ

    return (; dτ_Pr, dτ_r, νdτ, re_m, r, lτ_re_m, θ_dτ)
end


@views function main(backend=CPU(); nxy::Int=126)
    arch = Arch(backend)
    # geometry
    lx, ly = 2.0, 2.0
    # mechanics
    ρg   = (x=-0.0, y=1.0) # gravity force density
    # scales
    psc  = 1.0
    ηsc  = 1.0
    τsc  = ηsc/psc
    # numerics
    nx = ny = nxy
    grid   = UniformGrid(arch; origin=(-lx/2, -ly/2), extent=(lx, ly), dims=(nx, ny))
    launch = Launcher(arch, grid)
    niter  = 500nx
    ncheck = 10 #2nx
    ϵ_it   = 1e-6
    # PT params
    ds_PT  = initialize_PT_params(grid; vdτ_fac=2.2, r=0.5, re_m=2.3π) 
   
    # initial conditions
    init_incl(x, y, x0, y0, r, in, out) = ifelse((x - x0)^2 + (y - y0)^2 < r^2, in, out)
    ρgy = FunctionField(init_incl, grid, (Center(), Vertex()); parameters=(x0=0.0, y0=0.0, r=0.1lx, in=ρg.y, out=0.0))
    
    # allocate fields
    ds      = initialize_stokes_fields(backend, grid, ρgy)
    set!(ds.ηc, grid, init_incl; parameters=(x0=0.0, y0=0.0, r=0.1lx, in=1000.0, out=1.0))
    
    # smooth viscosity field (required for large jumps)
    smooth!(ds.ηc, ds.ηc, ds.q, grid, arch, launch, 10)
    
    # boundary conditions
    bc_V = (ds.V.x => (x=Dirichlet(), y=Neumann()),
            ds.V.y => (x=Neumann(), y=Dirichlet()))
    # visualisation
    fig = Figure(; size=(800, 600))
    axs = (Pr = Axis(fig[1, 1][1, 1]; aspect=DataAspect(), xlabel="x", title="Pr"),
           Vx = Axis(fig[1, 2][1, 1]; aspect=DataAspect(), xlabel="x", title="Vx"),
           Vy = Axis(fig[2, 1][1, 1]; aspect=DataAspect(), xlabel="x", title="Vy"),
           T  = Axis(fig[2, 2][1, 1]; aspect=DataAspect(), xlabel="x", title="ηc"))
    plt = (Pr = heatmap!(axs.Pr, centers(grid)..., interior(ds.Pr) |> Array; colormap=:turbo),
           Vx = heatmap!(axs.Vx, xvertices(grid), ycenters(grid), interior(ds.V.x) |> Array; colormap=:turbo),
           Vy = heatmap!(axs.Vy, xcenters(grid), yvertices(grid), interior(ds.V.y) |> Array; colormap=:turbo),
           η  = heatmap!(axs.T, centers(grid)..., interior(ds.ηc) |> Array; colormap=:turbo)
           )
    Colorbar(fig[1, 1][1, 2], plt.Pr)
    Colorbar(fig[1, 2][1, 2], plt.Vx)
    Colorbar(fig[2, 1][1, 2], plt.Vy)
    Colorbar(fig[2, 2][1, 2], plt.η)
    display(fig)
    # action
    @time begin
        
        # Perform one timestep 
        stokes_iterations(launch, arch, grid, ds, ds_PT, bc_V, niter, ncheck, ϵ_it, τsc, ly, psc)

        KernelAbstractions.synchronize(backend)
    end
    plt.Pr[3] = interior(ds.Pr) |> Array
    plt.Vx[3] = interior(ds.V.x) |> Array
    plt.Vy[3] = interior(ds.V.y) |> Array
    plt.η[3] = interior(ds.ηc) |> Array
    display(fig)
    return
end

# main(ROCBackend(); nxy=254)
#main(CUDABackend(); nxy=510)
main(; nxy=254)
