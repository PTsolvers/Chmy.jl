module Architectures

export Architecture, SingleDeviceArchitecture
export Arch, get_backend, get_device, activate!, set_device!, heuristic_groupsize

using KernelAbstractions

"""
    abstract type Architecture

Abstract type representing an architecture.
"""
abstract type Architecture end

"""
    struct SingleDeviceArchitecture{B,D} <: Architecture

A struct representing an architecture that operates on a single CPU or GPU device.
"""
struct SingleDeviceArchitecture{B,D} <: Architecture
    backend::B
    device::D
end

"""
    Arch(backend::Backend; device_id::Integer=1)

Create an architecture object for the specified backend and device.

# Arguments
- `backend`: The backend to use for computation.
- `device_id=1`: The ID of the device to use.
"""
function Arch(backend::Backend; device_id::Integer=1)
    dev = get_device(backend, device_id)
    return SingleDeviceArchitecture(backend, dev)
end

"""
    get_backend(arch::SingleDeviceArchitecture)

Get the backend associated with a SingleDeviceArchitecture.
"""
get_backend(arch::SingleDeviceArchitecture) = arch.backend

"""
    get_device(arch::SingleDeviceArchitecture)

Get the device associated with a SingleDeviceArchitecture.
"""
get_device(arch::SingleDeviceArchitecture) = arch.device

"""
    activate!(arch::SingleDeviceArchitecture; priority=:normal)

Activate the given architecture on the specified device and set the priority of the backend.
"""
function activate!(arch::SingleDeviceArchitecture; priority=:normal)
    set_device!(arch.device)
    KernelAbstractions.priority!(arch.backend, priority)
end

# CPU
get_device(::CPU, device_id) = nothing
set_device!(::Nothing) = nothing
heuristic_groupsize(::CPU, ::Val{N}) where {N} = 256

end
