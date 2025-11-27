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

function laplace_2d(nx, ny)
    # grid
    grid = Grid(nx, ny)
    inds = indices(grid)

    # operators
    p, s = Point(), Segment()
    D    = StaggeredCentralDifference()
    grad = Gradient{ndims(grid)}(D)
    divg = Divergence{ndims(grid)}(D)

    # main equation definition (spelled out)
    f = Tag(:f)
    q = -grad(f)
    r = -divg(q)
    r_c = lower_stencil(r[p, p][inds...])

    ## boundary conditions
    # values at bottom and top boundaries
    f_b = f[p, p][inds[1], inds[2]-1]
    f_t = f[p, p][inds[1], inds[2]+1]
    # fluxes at left and right boundaries
    q_l = lower_stencil(q[1][s, p][inds[1]-1, inds[2]])
    q_r = lower_stencil(q[1][s, p][inds[1], inds[2]])
    # side boundary conditions
    bc_l = q_l => SUniform(0)
    bc_r = q_r => SUniform(0)
    bc_b = f_b => SUniform(+1)
    bc_t = f_t => SUniform(-1)
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

    bc = (l=r_l, r=r_r, b=r_b, t=r_t, bl=r_bl, br=r_br, tl=r_tl, tr=r_tr)

    # arrays
    R = zeros(dims(grid, p, p))
    B = Binding(f[p, p] => rand(dims(grid, p, p)...))
    F = B[f[p, p]]

    # visualisation
    fig = Figure(; size=(650, 270))
    ax = (Axis(fig[1, 1][1, 1]; aspect=DataAspect(), title="initial guess"),
          Axis(fig[1, 2][1, 1]; aspect=DataAspect(), title="final solution"))
    plt = (heatmap!(ax[1], F; colorrange=(-1, 1), colormap=:roma),
           heatmap!(ax[2], F; colorrange=(-1, 1), colormap=:roma))
    cb = (Colorbar(fig[1, 1][1, 2], plt[1]),
          Colorbar(fig[1, 2][1, 2], plt[2]))

    # iterative loop
    for iter in 1:50_000
        # compute residual
        @inbounds begin
            # inner points
            for j in 2:ny-1, i in 2:nx-1
                R[i, j] = compute(r_c, B, i, j)
            end
            # x sides
            for j in 2:ny-1
                R[1, j]  = compute(bc.l, B, 1, j)
                R[nx, j] = compute(bc.r, B, nx, j)
            end
            # y sides
            for i in 2:nx-1
                R[i, 1]  = compute(bc.b, B, i, 1)
                R[i, ny] = compute(bc.t, B, i, ny)
            end
            # corners
            R[1, 1]   = compute(bc.bl, B, 1, 1)
            R[nx, 1]  = compute(bc.br, B, nx, 1)
            R[1, ny]  = compute(bc.tl, B, 1, ny)
            R[nx, ny] = compute(bc.tr, B, nx, ny)
        end
        # update solution
        @. F += 0.25 * R
    end

    # update plots
    plt[2][1] = F
    display(fig)

    return
end

laplace_2d(51, 51)
