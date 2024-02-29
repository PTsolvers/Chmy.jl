"""
    DistributedArchitecture <: Architecture

A struct representing a distributed architecture.
"""
struct DistributedArchitecture{ChildArch,Topo} <: Architecture
    child_arch::ChildArch
    topology::Topo
end

"""
    Arch(backend, comm, dims; kwargs...)

Create a distributed Architecture using backend `backend` and `comm`.
For GPU backends, device will be selected automatically based on a process id within a node.
"""
function Architectures.Arch(backend::Backend, comm::MPI.Comm, dims; device_id=nothing)
    topology   = CartesianTopology(comm, dims)
    dev_id     = isnothing(device_id) ? shared_rank(topology) + 1 : device_id
    dev        = get_device(backend, dev_id)
    child_arch = SingleDeviceArchitecture(backend, dev)
    return DistributedArchitecture(child_arch, topology)
end

topology(arch::DistributedArchitecture) = arch.topology

# Implement Architecture API
Architectures.get_backend(arch::DistributedArchitecture) = Architectures.get_backend(arch.child_arch)
Architectures.get_device(arch::DistributedArchitecture) = get_device(arch.child_arch)
Architectures.activate!(arch::DistributedArchitecture; kwargs...) = activate!(arch.child_arch; kwargs...)
