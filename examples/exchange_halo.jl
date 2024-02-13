using Chmy.Architectures, Chmy.Grids, Chmy.Fields, Chmy.BoundaryConditions, Chmy.Distributed

using KernelAbstractions
using MPI

MPI.Init()

backend = CPU()
arch = Arch(backend, MPI.COMM_WORLD, (0, 0))
topo = topology(arch)

grid = UniformGrid(arch; origin=(0, 0), extent=(1, 1), dims=(12, 12))

field = Field(backend, grid, Center())
fill!(parent(field), global_rank(topo))

for _ in 1:10
    @time exchange_halo!(arch, grid, field)
end

KernelAbstractions.synchronize(backend)

sleep(0.2global_rank(topo))
display(interior(field; with_halo=true))

MPI.Finalize()
