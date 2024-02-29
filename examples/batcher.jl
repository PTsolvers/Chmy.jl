using Chmy.Grids, Chmy.BoundaryConditions, Chmy.Architectures, Chmy.Fields
using KernelAbstractions

arch = Arch(CPU())

grid = UniformGrid(arch; origin=(0, 0, 0), extent=(1, 1, 1), dims=(62, 62, 62))

C = Field(CPU(), grid, Center())
P = Field(CPU(), grid, Center(), Int)

set!(C, grid, (_, _, _) -> rand())
set!(P, grid, (_, _, _) -> rand(Int))

# pre-compute batch
bt = batch(grid, C => Neumann(), P => (y=(Dirichlet(), nothing), x=Neumann()))
bc!(arch, grid, bt)

# fused syntax
bc!(arch, grid, C => Neumann(), P => (y=(Dirichlet(), nothing), x=Neumann()))
