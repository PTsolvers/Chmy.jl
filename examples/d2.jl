using Chmy, KernelAbstractions

@kernel function d2!(V1, C2, C4, C1, g::StructuredGrid, O)
    I = @index(Global, NTuple)
    I = I + O
    V1[I...] = ∂x(C1, g, I...)
    I_r = Chmy.GridOperators.ir(Vertex(), Center(), Dim(1), I...)
    I_l = Chmy.GridOperators.il(Vertex(), Center(), Dim(1), I...)
    C2[I...] = (∂x(C1, g, I_r...) - ∂x(C1, g, I_l...)) * iΔ(g, Center(), Dim(1), I...)
    C4[I...] = ∂²x(C1, g, I...)
end

@kernel function d2!(C3, V1, g::StructuredGrid, O)
    I = @index(Global, NTuple)
    I = I + O
    C3[I...] = ∂x(V1, g, I...)
end

backend = CPU()
arch = Arch(backend)
grid = UniformGrid(arch; origin=(-5, ), extent=(10, ), dims=(16,))
launch = Launcher(arch, grid)

C1 = Field(backend, grid, Center())
C2 = Field(backend, grid, Center())
C3 = Field(backend, grid, Center())
C4 = Field(backend, grid, Center())
V1 = Field(backend, grid, Vertex())

set!(C1, grid, (x,) -> exp(-x^2))

launch(arch, grid, d2! => (V1, C2, C4, C1, grid))
launch(arch, grid, d2! => (C3, V1, grid))

@assert C3 == C2
@assert C3 == C4
