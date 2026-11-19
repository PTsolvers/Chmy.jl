using Chmy

grid = Grids.MAC(nx, ny, nz)

@scalars p η
@vectors v ρg

# interpolate(u, recipe; order=2, wall=:reduce)
# wall = :reduce for gradually lowering the order, :keep for keeping the order, :none for keeping out-of-grid samples
# order = 2 for linear, 4 for cubic etc.
η = interpolate(η, grid)
ρg = interpolate(ρg, grid)

divg = Divergence(StaggeredCentralDifference())
grad = Gradient(StaggeredCentralDifference())

τ = 2η*sym(grad(v))
r_v = divg(-p * ℐ(2) + τ) + ρg

r_v[1][(:, 0)]
r_v[1][(:, 0), (:, 1//2)]
r_v[1][(:, 0), (:, 1//2), (:, 1//2)]
