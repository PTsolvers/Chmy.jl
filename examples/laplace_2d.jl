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
    grad = Gradient(D)
    divg = Divergence(D)

    # main equation definition (spelled out)
    f = SScalar(:f)
    q = -grad(f)
    r = Tensor{2}(-divg(q))
    r_c = lower_stencil(r[s, s][inds...])
    qv = Tensor{2}(q)

    ## boundary conditions
    # values at bottom and top boundaries
    f_b = f[s, s][inds[1], inds[2]-1]
    f_t = f[s, s][inds[1], inds[2]+1]
    # fluxes at left and right boundaries
    q_l = lower_stencil(qv[1][p, s][inds[1], inds[2]])
    q_r = lower_stencil(qv[1][p, s][inds[1]+1, inds[2]])
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

    r_c = simplify(r_c)

    bc = (l  = simplify(r_l),
          r  = simplify(r_r),
          b  = simplify(r_b),
          t  = simplify(r_t),
          bl = simplify(r_bl),
          br = simplify(r_br),
          tl = simplify(r_tl),
          tr = simplify(r_tr))

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
    for iter in 1:50_000
        # compute residual
        begin
            # inner points
            for j in 2:Ny-1, i in 2:Nx-1
                R[i, j] = compute(r_c, B, i, j)
            end
            # x sides
            for j in 2:Ny-1
                R[1, j]  = compute(bc.l, B, 1, j)
                R[Nx, j] = compute(bc.r, B, Nx, j)
            end
            # y sides
            for i in 2:Nx-1
                R[i, 1]  = compute(bc.b, B, i, 1)
                R[i, Ny] = compute(bc.t, B, i, Ny)
            end
            # corners
            R[1, 1]   = compute(bc.bl, B, 1, 1)
            R[Nx, 1]  = compute(bc.br, B, Nx, 1)
            R[1, Ny]  = compute(bc.tl, B, 1, Ny)
            R[Nx, Ny] = compute(bc.tr, B, Nx, Ny)
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
