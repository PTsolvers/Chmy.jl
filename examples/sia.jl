using Chmy.Architectures, Chmy.Grids, Chmy.Fields, Chmy.BoundaryConditions, Chmy.GridOperators
using KernelAbstractions

using Printf
using CairoMakie

@kernel function update_H!(H, q, M, Δt, g)
    I = @index(Global, Cartesian)
    H[I] = max(H[I] + Δt * (-divg(q, g, I) + M[I]), 0.0)
end

@kernel function update_∇S!(∇S, S, g)
    I = @index(Global, Cartesian)
    ∇S.x[I] = ∂x(S, g, I)
    ∇S.y[I] = ∂y(S, g, I)
end

@kernel function update_D!(D, H, ∇S, Aₛ, n, g)
    I = @index(Global, Cartesian)
    D[I] = (H[I]^(n + 2) + Aₛ * H[I]^n) * vmag(∇S, g, I)^(n - 1)
end

@kernel function update_q!(q, D, S, g)
    I = @index(Global, Cartesian)
    q.x[I] = -lerp(D, location(q.x), g, I) * ∂x(S, g, I)
    q.y[I] = -lerp(D, location(q.y), g, I) * ∂y(S, g, I)
end

function main()
    backend = CPU()
    arch    = Arch(backend)

    grid = UniformGrid(arch; origin=(-1, -1), extent=(2, 2), dims=(100, 100))

    Aₛ = 1.0
    n  = 3

    H  = Field(arch, grid, Center())
    D  = Field(arch, grid, Center())
    ∇S = VectorField(arch, grid)
    q  = VectorField(arch, grid)

    B = FunctionField(grid, Center()) do x, y
        bump1  = 0.1 * exp(-((x - 0.5) / 0.3)^2 - ((y - 0.0) / 0.5)^2)
        bump2  = 0.1 * exp(-((x + 0.5) / 0.3)^2 - ((y - 0.0) / 0.5)^2)
        bump3  = 0.5 * exp(-(x / 0.5)^2 - ((y - 0.5) / 0.5)^2)
        linear = 0.08 * y
        return 0.1 + bump1 + bump2 + bump3 + linear
    end

    S = FunctionField(grid, Center(); discrete=true, parameters=(B, H)) do grid, loc, ix, iy, B, H
        return B[ix, iy] + H[ix, iy]
    end

    M = FunctionField(grid, Center(); discrete=true, parameters=(S,)) do grid, loc, ix, iy, S
        return min(30.0 * (S[ix, iy] - 0.2), 0.01)
    end

    fig = Figure()
    ax  = Axis(fig[1, 1][1, 1]; aspect=DataAspect(), xlabel="x", ylabel="y")
    hm  = heatmap!(ax, centers(grid)..., Array(S))
    Colorbar(fig[1, 1][1, 2], hm)
    display(fig)

    wg = 256
    ws = size(grid, Center()) .+ 2

    ttot = 50.0
    tcur = 0.0
    it   = 1

    while tcur < ttot
        update_∇S!(backend, wg, ws)(∇S, S, grid)
        update_D!(backend, wg, ws)(D, H, ∇S, Aₛ, n, grid)
        update_q!(backend, wg, ws)(q, D, S, grid)
        bc!(arch, grid, q.x => (x = Dirichlet()), q.y => (y = Dirichlet()))

        Δt_diff = minimum(spacing(grid))^2 / maximum(interior(D)) / 4.1
        Δt_mb   = 0.1
        Δt      = inv(inv(Δt_diff) + inv(Δt_mb))

        update_H!(backend, wg, ws)(H, q, M, Δt, grid)
        bc!(arch, grid, H => Neumann())

        if it % 10 == 0
            @printf("t = %1.3e/%1.3e\n", tcur, ttot)

            hm[3] = Array(S)
            yield()
            # display(fig)
        end

        it   += 1
        tcur += Δt
    end

    return
end

main()
