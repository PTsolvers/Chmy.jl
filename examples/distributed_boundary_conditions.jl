using Chmy.Architectures, Chmy.Grids, Chmy.Fields, Chmy.BoundaryConditions, Chmy.Distributed

using KernelAbstractions
using MPI

MPI.Init()

arch = Arch(CPU(), MPI.COMM_WORLD, (0, 0))
topo = topology(arch)

grid = UniformGrid(arch; origin=(0, 0), extent=(1, 1), dims=(10, 10))

field = Field(backend(arch), grid, Center())
fill!(parent(field), NaN)

set!(field, global_rank(topo))

bc = batch(arch, grid, field => Neumann(); exchange=true)

ntuple(Val(ndims(grid))) do D
    Base.@_inline_meta
    ntuple(Val(2)) do S
        bc = has_neighbor(topo, D, S) ? ExchangeData(Val(S), Val(D), field) : Neumann()
        bc!(Val(S), Val(D), arch, grid, (field,), (bc,))
    end
end

KernelAbstractions.synchronize(backend(arch))

sleep(global_rank(topo))
display(interior(field; with_halo=true))

MPI.Finalize()
