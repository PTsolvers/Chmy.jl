using Chmy.Grids, Chmy.BoundaryConditions, Chmy.Architectures, Chmy.Fields
using KernelAbstractions

arch = Arch(CPU())

grid = UniformGrid(arch; origin=(0, 0), extent=(1, 1), dims=(100, 100))

C = Field(CPU(), grid, Center())
P = Field(CPU(), grid, Center(), Int)

set!(C, grid, (_, _) -> rand())
set!(P, grid, (_, _) -> rand(Int))

bc!(arch, grid, C => Neumann(), P => (y=(Dirichlet(), nothing), x=Neumann()))

bt = batch(arch, grid, C => Neumann(), P => (y=(Dirichlet(), nothing), x=Neumann()))
bc!(arch, grid, bt)
