using Chmy, KernelAbstractions
using Metal

@kernel function fun1!(V1, V2, C2, C4, C5, C1, χ, g::StructuredGrid, O)
    I = @index(Global, NTuple)
    I = I + O
    V1.x[I...] = ∂x(C1, g, I...)
    V2.x[I...] = lerp(χ, location(V2.x), g, I...) * ∂x(C1, g, I...)
    # I_r = Chmy.GridOperators.ir(Vertex(), Center(), Dim(1), I...)
    # I_l = Chmy.GridOperators.il(Vertex(), Center(), Dim(1), I...)
    # C2[I...] = (∂x(C1, g, I_r...) - ∂x(C1, g, I_l...)) * iΔ(g, Center(), Dim(1), I...)
    C4[I...] = ∂²x(C1, g, I...)
    C5[I...] = lapl(C1, g, I...)
end

@kernel function fun2!(C3, C7, V1, V2, g::StructuredGrid, O)
    I = @index(Global, NTuple)
    I = I + O
    C3[I...] = ∂x(V1.x, g, I...)
    C7[I...] = ∂x(V2.x, g, I...)
end

@kernel function fun3!(C6, C1, χ, g::StructuredGrid, O)
    I = @index(Global, NTuple)
    I = I + O
    C6[I...] = divg_grad(C1, χ, g, I...)
end

# backend = CPU()
backend = MetalBackend()
arch = Arch(backend)
grid = UniformGrid(arch; origin=(-5f0, ), extent=(10f0, ), dims=(16,))
launch = Launcher(arch, grid)

χ  = Field(backend, grid, Center())
C1 = Field(backend, grid, Center())
C2 = Field(backend, grid, Center())
C3 = Field(backend, grid, Center())
C4 = Field(backend, grid, Center())
C5 = Field(backend, grid, Center())
C6 = Field(backend, grid, Center())
C7 = Field(backend, grid, Center())
# V1 = Field(backend, grid, Vertex())
# V2 = Field(backend, grid, Vertex())
V1 = VectorField(backend, grid)
V2 = VectorField(backend, grid)

set!(C1, grid, (x,) -> exp(-x^2))
set!(χ, grid, (x,) -> exp(-x^2))
# set!(q1.x, C1)

launch(arch, grid, fun1! => (V1, V2, C2, C4, C5, C1, χ, grid))
launch(arch, grid, fun2! => (C3, C7, V1, V2, grid))
launch(arch, grid, fun3! => (C6, C1, χ, grid))

# @assert interior(C3) == interior(C2)
@assert interior(C3) == interior(C4)
@assert interior(C3) == interior(C5)
@assert interior(C7) == interior(C6)
