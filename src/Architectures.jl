module Architectures

export Architecture, SingleDeviceArchitecture
export Arch, backend, device, activate!, set_device!, heuristic_groupsize

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
    dev = device(backend, device_id)
    return SingleDeviceArchitecture(backend, dev)
end

"""
    backend(arch::SingleDeviceArchitecture)

Get the backend associated with a SingleDeviceArchitecture.
"""
backend(arch::SingleDeviceArchitecture) = arch.backend

"""
    device(arch::SingleDeviceArchitecture)

Get the device associated with a SingleDeviceArchitecture.
"""
device(arch::SingleDeviceArchitecture) = arch.device

"""
    activate!(arch::SingleDeviceArchitecture; priority=:normal)

Activate the given architecture on the specified device and set the priority of the backend.
"""
function activate!(arch::SingleDeviceArchitecture; priority=:normal)
    set_device!(arch.device)
    KernelAbstractions.priority!(arch.backend, priority)
end

# CPU
device(::CPU, device_id) = nothing
set_device!(::Nothing) = nothing
heuristic_groupsize(::CPU, ::Val{N}) where {N} = 256

end
