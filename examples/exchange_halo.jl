using Chmy

using KernelAbstractions
using AMDGPU
using MPI

MPI.Init()

# backend = CPU()
backend = ROCBackend()
arch = Arch(backend, MPI.COMM_WORLD, (0, 0))
topo = topology(arch)

# needed for weak scaling
dims_l = (12, 12)
dims_g = dims_l .* dims(topo)

grid = UniformGrid(arch; origin=(0, 0), extent=(1, 1), dims=dims_g)

field = Field(backend, grid, Center(), Int)
fill!(parent(field), global_rank(topo))

for _ in 1:10
    @time exchange_halo!(arch, grid, field)
end

KernelAbstractions.synchronize(backend)

sleep(0.2global_rank(topo))
display(interior(field; with_halo=true))

MPI.Finalize()
