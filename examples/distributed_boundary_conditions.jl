using Chmy.Architectures, Chmy.Grids, Chmy.Fields, Chmy.BoundaryConditions, Chmy.Distributed

using KernelAbstractions
using MPI

MPI.Init()

arch = Arch(CPU(), MPI.COMM_WORLD, (0, 0))
topo = topology(arch)

grid = UniformGrid(arch; origin=(0, 0), extent=(1, 1), dims=(12, 12))

field = Field(backend(arch), grid, Center())
fill!(parent(field), NaN)

set!(field, global_rank(topo))

bc!(arch, grid, field => Neumann(); replace=true)

KernelAbstractions.synchronize(backend(arch))

sleep(0.2global_rank(topo))
display(interior(field; with_halo=true))

MPI.Finalize()
