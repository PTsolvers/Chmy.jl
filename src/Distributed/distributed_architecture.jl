"""
    DistributedArchitecture <: Architecture

A struct representing a distributed architecture.
"""
struct DistributedArchitecture{ChildArch,Topo} <: Architecture
    child_arch::ChildArch
    topology::Topo
    gpu_aware::Bool
end

"""
    Architectures.Arch(backend::Backend, comm::MPI.Comm, dims; device_id=nothing, gpu_aware=true)

Create a distributed Architecture using backend `backend` and `comm`.
For GPU backends, device will be selected automatically based on a process id within a node, unless specified by `device_id`.

# Arguments
- `backend::Backend`: The backend to use for the architecture.
- `comm::MPI.Comm`: The MPI communicator to use for the architecture.
- `dims`: The dimensions of the architecture.

# Keyword Arguments
- `device_id`: The ID of the device to use. If not provided, the shared rank of the topology plus one is used.
- `gpu_aware`: Whether the MPI implementation is GPU-aware. If not provided, defaults to `true`. Only applies to compatible backends.
"""
function Architectures.Arch(backend::Backend, comm::MPI.Comm, dims; device_id=nothing, gpu_aware=true)
    topology   = CartesianTopology(comm, dims)
    dev_id     = isnothing(device_id) ? shared_rank(topology) + 1 : device_id
    dev        = get_device(backend, dev_id)
    child_arch = SingleDeviceArchitecture(backend, dev)
    gpu_aware  = gpu_aware_compat(backend) ? gpu_aware : false
    return DistributedArchitecture(child_arch, topology, gpu_aware)
end

"""
    topology(arch::DistributedArchitecture)

Get the virtual MPI topology of a distributed architecture
"""
topology(arch::DistributedArchitecture) = arch.topology

# Implement Architecture API
"""
    get_backend(arch::DistributedArchitecture)

Get the backend associated with a DistributedArchitecture by delegating to the child architecture.
"""
Architectures.get_backend(arch::DistributedArchitecture) = Architectures.get_backend(arch.child_arch)

"""
    get_device(arch::DistributedArchitecture)

Get the device associated with a DistributedArchitecture by delegating to the child architecture.
"""
Architectures.get_device(arch::DistributedArchitecture) = get_device(arch.child_arch)

"""
    activate!(arch::DistributedArchitecture; kwargs...)

Activate the given DistributedArchitecture by delegating to the child architecture,
and pass through any keyword arguments. For example, the priority can be set with
accepted values being `:normal`, `:low`, and `:high`.
"""
Architectures.activate!(arch::DistributedArchitecture; kwargs...) = activate!(arch.child_arch; kwargs...)

"""
    is_gpu_aware(arch::DistributedArchitecture)

Returns whether the DistributedArchitecture is GPU-aware.
"""
is_gpu_aware(arch::DistributedArchitecture) = arch.gpu_aware
