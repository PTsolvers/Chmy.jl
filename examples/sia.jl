using Chmy.Architectures, Chmy.Grids, Chmy.Fields, Chmy.BoundaryConditions, Chmy.GridOperators
using KernelAbstractions

using Printf
using CairoMakie

@kernel inbounds = true function update_H!(H, q, M, Δt, g)
    I = @index(Global, Cartesian)
    H[I] = max(H[I] + Δt * (-divg(q, g, I) + M[I]), 0.0)
end

@kernel inbounds = true function update_q!(q, D, S, g)
    I = @index(Global, Cartesian)
    q.x[I] = -lerp(D, location(q.x), g, I) * ∂x(S, g, I)
    q.y[I] = -lerp(D, location(q.y), g, I) * ∂y(S, g, I)
end

function main()
    backend = CPU()
    arch    = Arch(backend)

    grid = UniformGrid(arch; origin=(-1, -1), extent=(2, 2), dims=(256, 256))

    Aₛ = 0.5
    n  = 3

    H = Field(arch, grid, Center())
    q = VectorField(arch, grid)

    B = FunctionField(grid, Center()) do x, y
        @inline
        bump1 = 0.3 * exp(-((x - 0.5) / 0.3)^2 - ((y - 0.0) / 0.5)^2)
        bump2 = 0.3 * exp(-((x + 0.5) / 0.3)^2 - ((y - 0.0) / 0.5)^2)
        bump3 = 1.0 * exp(-(x / 0.5)^2 - ((y - 0.5) / 0.5)^2)
        return bump1 + bump2 + bump3
    end

    S = FunctionField(grid, Center(); discrete=true, parameters=(B, H)) do g, l, ix, iy, B, H
        @inline
        return B[ix, iy] + H[ix, iy]
    end

    ∇S = (x=FunctionField((g, l, ix, iy, S) -> ∂x(S, g, ix, iy), grid, (Vertex(), Center()); discrete=true, parameters=(S,)),
          y=FunctionField((g, l, ix, iy, S) -> ∂y(S, g, ix, iy), grid, (Center(), Vertex()); discrete=true, parameters=(S,)))

    D = FunctionField(grid, Center(); discrete=true, parameters=(H, ∇S, Aₛ, n)) do g, l, ix, iy, H, ∇S, Aₛ, n
        @inline
        return (H[ix, iy]^(n + 2) + Aₛ * H[ix, iy]^n) * vmag(∇S, g, ix, iy)^(n - 1)
    end

    M = FunctionField(grid, Center(); discrete=true, parameters=(S,)) do grid, loc, ix, iy, S
        @inline
        return min(0.5 * (S[ix, iy] - 0.3), 0.25)
    end

    fig = Figure()
    ax  = Axis(fig[1, 1][1, 1]; aspect=DataAspect(), xlabel="x", ylabel="y")
    hm  = heatmap!(ax, centers(grid)..., Array(S); colormap=:turbo)
    Colorbar(fig[1, 1][1, 2], hm)
    display(fig)

    wg = (32, 8)
    ws = size(grid, Center())

    ttot   = 1.0
    tcur   = 0.0
    it     = 1
    Δt_out = 0.1
    tout   = 0.0

    while tcur <= ttot
        update_q!(backend, wg, ws)(q, D, S, grid)
        bc!(arch, grid, q.x => (x = Dirichlet()), q.y => (y = Dirichlet()))

        # compute time step
        Δt_diff = minimum(spacing(grid))^2 / maximum(interior(D)) / 4.1
        Δt_mb   = 0.5
        Δt      = min(Δt_diff, Δt_mb, Δt_out - tout)

        update_H!(backend, wg, ws)(H, q, M, Δt, grid)
        bc!(arch, grid, H => Neumann())

        if tout >= Δt_out
            @printf("t = %1.3e/%1.3e\n", tcur, ttot)

            hm[3] = Array(H)
            # yield()
            display(fig)

            tout = 0.0
        end

        it   += 1
        tcur += Δt
        tout += Δt
    end

    return
end

main()
