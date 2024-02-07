"""
    Distributed

Tools for performing parallel computations in a distributed environment.
Contains tools for non-blocking halo exchange using MPI.
Implements `BoundaryConditions` API to conveniently define communication as an operation that fills halo buffers.
Enables hiding MPI communication behind computations.
"""
module Distributed

export CartesianTopology, global_rank, shared_rank, node_name, cart_comm, shared_comm
export dims, coords, neighbors, neighbor, has_neighbor, global_size, node_size
export DistributedArchitecture, topology
export gather!
export Connected
export Exchange, ExchangeBatch

using Chmy
using Chmy.Grids
using Chmy.Fields
using Chmy.Architectures
using Chmy.BoundaryConditions

using MPI
using KernelAbstractions

# grid connectivity for distributed topologies
struct Connected <: Connectivity end

overlap(::Vertex) = 1
overlap(::Center) = 0

const OptionalBuffer = Union{AbstractArray,Nothing}

struct Exchange{SB<:OptionalBuffer,RB<:OptionalBuffer}
    send_buffer::SB
    recv_buffer::RB
end

Exchange() = Exchange(nothing, nothing)

struct ExchangeBatch{K,F,R} <: BoundaryConditions.AbstractBatch
    fields::F
    exchanges::R
    function ExchangeBatch(fields::NTuple{K,Field}, exchanges::NTuple{K,Exchange}) where {K}
        return new{K,typeof(fields),typeof(exchanges)}(fields, exchanges)
    end
end

include("topology.jl")
include("distributed_architecture.jl")
include("distributed_grid.jl")
include("stack_allocator.jl")
include("communication_views.jl")
include("task_local_exchanger.jl")
include("boundary_conditions.jl")
include("gather.jl")

end
