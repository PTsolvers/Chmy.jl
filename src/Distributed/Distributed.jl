"""
    Distributed

Tools for performing parallel computations in a distributed environment.
Contains tools for non-blocking halo exchange using MPI.
Implements `BoundaryConditions` API to conveniently define communication as an operation that fills halo buffers.
Enables hiding MPI communication behind computations.
"""
module Distributed

export CartesianTopology, global_rank, shared_rank, node_name, cart_comm, shared_comm
export dims, cart_coords, neighbors, neighbor, has_neighbor, global_size, node_size
export DistributedArchitecture, topology, is_gpu_aware
export exchange_halo!, gather!

using Chmy
using Chmy.Grids
using Chmy.Fields
using Chmy.Architectures
using Chmy.BoundaryConditions

import Chmy.Architectures: gpu_aware_compat

using MPI
using KernelAbstractions

include("topology.jl")
include("distributed_architecture.jl")
include("distributed_grid.jl")
include("stack_allocator.jl")
include("communication_views.jl")
include("task_local_exchanger.jl")
include("exchange_halo.jl")
include("gather.jl")

end
