# Using Chmy.jl with MPI

This tutorial dives into the `Distributed` module in [Chmy.jl](https://github.com/PTsolvers/Chmy.jl) To show how to run a code on multiple processes in a typical HPC cluster setup. We start from the [diffusion_2d.jl](https://github.com/PTsolvers/Chmy.jl/blob/main/examples/diffusion_2d.jl) code from the [Getting Started with Chmy.jl](./introduction.md) section.

!!! warning "Experience with HPC Clusters Assumed"
    In this tutorial, we assume users to be familiar with HPC clusters and the basic concepts of distributed computing. If you find anything conceptually difficult to start with, have a look at the concept documentation on the [Distributed](../concepts/distributed.md) module.

We need to make the following changes to our code to enable MPI support, in which we:

1. initialise MPI environment & specify distributed architecture

2. redefine geometry

3. avoid redundant I/O operations

4. data gathering for visualisation

5. finalise MPI Environment

## Initialise MPI & Specify Distributed Architecture

The first step is to load the `MPI.jl` module and initialise MPI with `MPI.Init()` at the beginning of the program.

```julia
using MPI
MPI.Init()
```

To make the `Architecture` object aware of MPI topology, the user can pass an MPI communicator object (here `MPI.COMM_WORLD`) and dimensions of the Cartesian topology to the `Arch` constructor as the last argument. Passing zeros will automatically spread the dimensions to be as close as possible to each other, see [MPI.jl documentation](https://juliaparallel.org/MPI.jl/stable/reference/topology/#MPI.Dims_create) for details.

```julia
arch = Arch(backend, MPI.COMM_WORLD, (0, 0))
topo = topology(arch)
me   = global_rank(topo)
```

The `global_rank()` function provides a convenient method for users to retrieve the unique process ID (global rank) from the current MPI communicator. This unique ID, stored in `me`, is utilised to assign process-specific tasks, such as I/O operations.


## Redefine Geometry

In the original single-node setup, we defined a global grid that covered the entire computational domain.

```julia
@views function main(backend=CPU(); nxy=(126, 126))
    # Before: geometry
    grid   = UniformGrid(arch; origin=(-2, -2), extent=(4, 4), dims=nxy)
    launch = Launcher(arch, grid)

    # ...
end

main(; nxy=(128, 128) .- 2)
```

In Chmy.jl, the grid constructor (here `UniformGrid`) always takes the dimensions `dims` of the global grid as input argument and returns the local `grid` object. For single-device architecture (no MPI) the local `grid` is equivalent to the global `grid` given that a single process performs the computations on the global domain. For distributed architecture, `dims` still takes as input the global grid dimension but returns the local portion of the grid corresponding to each MPI rank.

Following a GPU-centric approach, we want here to control the local dimension of the grid, `nxy_l` hereafter, to ensure optimal execution on a single GPU. We thus need to reconstruct the global grid dimensions `dims_g` based on the MPI topology and the local grid dimensions `dims_l`:

```julia
@views function main(backend=CPU(); nxy_l=(126, 126))
    # After: geometry
    dims_l = nxy_l
    dims_g = dims_l .* dims(topo)
    nx, ny = dims_g
    grid   = UniformGrid(arch; origin=(-2, -2), extent=(4, 4), dims=dims_g)
    launch = Launcher(arch, grid, outer_width=(16, 8))

    # ...
end

main(; nxy_l=(128, 128) .- 2)
```

Here, `dims_g` represents the global dimensions of the grid, which is obtained by multiplying the local grid dimensions `dims_l` by the MPI topology dimensions. The `outer_width` parameter specifies the number of grid points that constitute the boundary region of each local grid. This approach is used to perform asynchronous computations on each local domain and to overlap boundary conditions computations (including MPI communication) with inner point computations. This allows to hide MPI communication latency overlapping communication and computations.


## Avoid Redundant I/O Operations

Previously, for a single process, we can simply print out any information during the code execution, whether it is the value of some physical properties that we want to monitor about or the current number of iterations during the simulation.

```julia
# Before: prints out current no. iteration on a single node
@printf("it = %d/%d \n", it, nt)
```

In a distributed setup, on the other hand, all MPI processes execute the same code and would have displayed the same line with this statement. In order to prevent this redundancy, we utilise the unique process ID to determine whether the process that is currently running is the one that we have assigned to handle the I/O task.

```julia
# After: specifying only process with ID == 0 to perform I/O operations
(me==0) && @printf("it = %d/%d \n", it, nt)
```

## Data Gathering for Visualisation

Not addressing parallel I/O here, we want to visualise the field `C`. With MPI we've split up `C` across various distributed processes. Each process handles a portion of the computation, leading to the necessity of data gathering for visualisation. Let us define a global array `C_v` on the "master" MPI process with unique process ID equals zero (`me==0`) that should gather all data from other MPI processes to later perform visualisation. Note the the global size of `C_v` being `interior(C)) .* dims(topo)`.

```julia
C_v = (me==0) ? KernelAbstractions.zeros(CPU(), Float64, size(interior(C)) .* dims(topo)) : nothing
```

We use `gather!(arch, C_v, C)` to explicitly perform a data synchronisation and collect local values of `C` that are decomposed into different arrays stored in the memory space of other MPI processes. And similar to the `@printf` example above, only our master MPI process performs the visualisation.

```julia
# Before: local postprocess
# plt[3] = interior(C) |> Array
# ax.title = "it = $nt"
# display(fig)
# save("out$me.png", fig)

# After: global postprocess
gather!(arch, C_v, C)
if me == 0
    fig = Figure(; size=(400, 320))
    ax  = Axis(fig[1, 1]; aspect=DataAspect(), xlabel="x", ylabel="y", title="it = 0")
    plt = heatmap!(ax, C_v; colormap=:turbo)
    Colorbar(fig[1, 2], plt)
    save("out_gather_$nx.png", fig)
end
```

## Finalise MPI

At the very end of the program, we need to call `MPI.Finalize()` to clean up the MPI state.

```julia
MPI.Finalize()
```

!!! note "MPI finalisation"
    Running a Julia MPI code on a single process within the REPL (for e.g. development purpose) will require to terminate the Julia session upon MPI finalisation. Simply omitting `MPI.Finalize()` will allow for repeated execution of the code.

## Run the simulation on an MPI-parallel machine
If you want to run this on multiple MPI processes, you will need to setup the [MPI.jl](https://juliaparallel.org/MPI.jl) package, such that `mpiexecjl` is created on the command line. You can then run it on, e.g., 4 MPI ranks (processes) with:

```sh
mpiexecjl -n 4 --project=. julia diffusion_2d_mpi.jl
```

Note that we need not to do any changes for defining or launching kernels, as they are already MPI-compatible and need no further modification. The full code of the tutorial material is available under [diffusion\_2d\_mpi.jl](https://github.com/PTsolvers/Chmy.jl/blob/main/examples/diffusion_2d_mpi.jl).
