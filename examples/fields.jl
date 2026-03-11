using Chmy

A = SScalar(:A)
B = SScalar(:B)

bnd = Binding(A => 3.0, B => 2.0)
expr = B^2 + A * B

compute(expr, bnd)
Chmy.to_expr(expr, Chmy.binding_types(bnd))

# ⋅--⋅--⋅--⋅--⋅--⋅--⋅--⋅

D = StaggeredCentralDifference()
D(A)

∂ = PartialDerivative(D)

i, j = SIndex(1), SIndex(2)

∂(A, 1)[i, j]

p, s = Point(), Segment()
expr = ∂(A, 2)[s, p][i, j]

lower_stencil(expr)

P = SScalar(:P)
η = SScalar(:η)
V = SVec(:V)

grad = Gradient(D)
divg = Divergence(D)

ε̇ = 1 // 2 * (grad(V) + grad(V)')
τ = 2η * ε̇

I = SIdTensor{2}()

σ = -P * I + τ

Tensor{2}(σ)

Tensor{3}(STensor{2}(:T) + STensor{2}(:T)')
