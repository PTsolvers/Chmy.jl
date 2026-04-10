# 2D Laplace equation solver example
#
# Solves the Laplace equation ∇²f = 0 on a 2D grid using an iterative method.
# The problem is set up with:
# - Zero flux (Neumann) boundary conditions on left and right sides
# - Fixed values (Dirichlet) of +1 at bottom and -1 at top boundaries
# The solver uses Chmy's symbolic field algebra to define the diffusion operator
# and automatically handles boundary conditions at sides and corners.

using Chmy
using CairoMakie: Figure, Axis, Colorbar, DataAspect, heatmap!

function laplace_2d(nx, ny; niter=50_000, display_fig=true)
    # grid
    grid = Grid(nx, ny)
    i, j = indices(grid)

    # operators
    p, s = Point(), Segment()
    D    = StaggeredCentralDifference()
    grad = Gradient(D)
    divg = Divergence(D)

    # main equation definition (spelled out)
    @scalars f
    # Keep flux components structurally intact so boundary substitutions can
    # still match them after the residual is simplified.
    q = node(-grad(f))
    r = -divg(q)
    r_c = r[s, s][i, j]

    ## boundary conditions
    # normal fluxes at left and right boundaries
    q_l = q[1][p, s][i, j]
    q_r = q[1][p, s][i+1, j]
    f_c = f[s, s][i, j]
    q_b = q[2][s, p][i, j]
    q_t = q[2][s, p][i, j+1]
    # side boundary conditions
    bc_l = q_l => SLiteral(0)
    bc_r = q_r => SLiteral(0)
    bc_b = q_b => SLiteral(1) - f_c
    bc_t = q_t => f_c + SLiteral(1)
    # side residuals
    r_l = subs(r_c, bc_l)
    r_r = subs(r_c, bc_r)
    r_b = subs(r_c, bc_b)
    r_t = subs(r_c, bc_t)
    # corner residuals
    r_bl = subs(r_b, bc_l)
    r_br = subs(r_b, bc_r)
    r_tl = subs(r_t, bc_l)
    r_tr = subs(r_t, bc_r)

    sys = (c  = r_c,
           l  = r_l,
           r  = r_r,
           b  = r_b,
           t  = r_t,
           bl = r_bl,
           br = r_br,
           tl = r_tl,
           tr = r_tr)

    # arrays
    R = zeros(dims(grid, s, s))
    B = Binding(f[s, s] => rand(dims(grid, s, s)...))
    F = B[f[s, s]]

    # visualisation
    fig = Figure(; size=(650, 270))
    ax = (Axis(fig[1, 1][1, 1]; aspect=DataAspect(), title="initial guess"),
          Axis(fig[1, 2][1, 1]; aspect=DataAspect(), title="final solution"))
    plt = (heatmap!(ax[1], F; colorrange=(-1, 1), colormap=:roma),
           heatmap!(ax[2], F; colorrange=(-1, 1), colormap=:roma))
    cb = (Colorbar(fig[1, 1][1, 2], plt[1]),
          Colorbar(fig[1, 2][1, 2], plt[2]))

    Nx, Ny = dims(grid, s, s)

    # iterative loop
    @time for _ in 1:niter
        # compute residual
        # inner points
        for iy in 2:Ny-1, ix in 2:Nx-1
            R[ix, iy] = compute(sys.c, B, ix, iy)
        end
        # x sides
        for iy in 2:Ny-1
            R[1, iy]  = compute(sys.l, B, 1, iy)
            R[Nx, iy] = compute(sys.r, B, Nx, iy)
        end
        # y sides
        for ix in 2:Nx-1
            R[ix, 1]  = compute(sys.b, B, ix, 1)
            R[ix, Ny] = compute(sys.t, B, ix, Ny)
        end
        # corners
        R[1, 1]   = compute(sys.bl, B, 1, 1)
        R[Nx, 1]  = compute(sys.br, B, Nx, 1)
        R[1, Ny]  = compute(sys.tl, B, 1, Ny)
        R[Nx, Ny] = compute(sys.tr, B, Nx, Ny)
        # update solution
        @. F += 0.25 * R
    end

    # update plots
    plt[2][1] = F
    display_fig && display(fig)

    return
end

laplace_2d(101, 101)
