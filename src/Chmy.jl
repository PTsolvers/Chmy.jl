module Chmy

using MacroTools
using KernelAbstractions

export 
    # utils
    Dim, Side, Left, Right, remove_dim, insert_dim, Offset,

    # Architectures
    Architecture, SingleDeviceArchitecture, Arch, get_backend, get_device, activate!, set_device!, heuristic_groupsize, pointertype,
    
    # BoundaryConditions
    FieldBoundaryCondition, FirstOrderBC, Dirichlet, Neumann, bc!,
    BoundaryFunction,
    DimSide,
    AbstractBatch, FieldBatch, ExchangeBatch, EmptyBatch, BatchSet, batch,

    # Distributed
    CartesianTopology, global_rank, shared_rank, node_name, cart_comm, shared_comm,
    dims, cart_coords, neighbors, neighbor, has_neighbor, global_size, node_size,
    DistributedArchitecture, topology,
    exchange_halo!, gather!,

    # DoubleBuffer
    DoubleBuffer, swap!, front, back,

    # Fields
    AbstractField, Field, VectorField, TensorField, ConstantField, ZeroField, OneField, ValueField, FunctionField,
    location, halo, interior, set!,
    divg,
    
    # Grids

    # GridOperators

    # KernelLaunch
    Launcher,
    worksize, outer_width, inner_worksize, inner_offset, outer_worksize, outer_offset,

    # Workers
    Worker

include("macros.jl")
include("utils.jl")

include("Architectures.jl")
include("DoubleBuffers.jl")
include("Grids/Grids.jl")
include("Fields/Fields.jl")
include("GridOperators/GridOperators.jl")

include("BoundaryConditions/BoundaryConditions.jl")

include("Workers.jl")
include("Distributed/Distributed.jl")

include("KernelLaunch.jl")

using .Architectures
using .BoundaryConditions
using .Distributed
using .DoubleBuffers
using .Fields
using .Grids
using .KernelLaunch
using .Workers

end # module Chmy
