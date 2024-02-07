using Chmy.Architectures, Chmy.Grids, Chmy.Fields, Chmy.BoundaryConditions, Chmy.Distributed

using KernelAbstractions
using MPI

function main(backend=CPU())
    arch = Arch(backend, MPI.COMM_WORLD, (0, 0, 0))
    topo = topology(arch)
    
    grid = UniformGrid(arch; origin=(0, 0, 0), extent=(1, 1, 1), dims=(3*100, 3*100, 3*100))
    
    field = Field(backend, grid, Center())
    fill!(parent(field), NaN)
    
    for _ in 1:10
        @time set!(field, global_rank(topo))
        @time bc!(arch, grid, field => Neumann(); replace=true)
    end
    
    KernelAbstractions.synchronize(backend)
    
    # sleep(0.2global_rank(topo))
    # display(interior(field; with_halo=true))
end

MPI.Init()
main()
MPI.Finalize()
