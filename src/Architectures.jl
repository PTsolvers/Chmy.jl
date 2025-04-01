module Architectures

export Architecture, SingleDeviceArchitecture
export Arch, get_backend, get_device, activate!, set_device!, heuristic_groupsize, pointertype, disable_task_sync!, enable_task_sync!, modify_sync!

using Chmy
using KernelAbstractions

"""
    Architecture

Abstract type representing an architecture.
"""
abstract type Architecture end

"""
    SingleDeviceArchitecture <: Architecture

A struct representing an architecture that operates on a single CPU or GPU device.
"""
struct SingleDeviceArchitecture{B,D} <: Architecture
    backend::B
    device::D
    function SingleDeviceArchitecture(backend, device)
        set_device!(device)
        return new{typeof(backend),typeof(device)}(backend, device)
    end
end

"""
    SingleDeviceArchitecture(arch::Architecture)

Create a `SingleDeviceArchitecture` object retrieving backend and device from `arch`.
"""
SingleDeviceArchitecture(arch::Architecture) = SingleDeviceArchitecture(get_backend(arch), get_device(arch))

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
KernelAbstractions.get_backend(arch::SingleDeviceArchitecture) = arch.backend

"""
    get_device(arch::SingleDeviceArchitecture)

Get the device associated with a SingleDeviceArchitecture.
"""
get_device(arch::SingleDeviceArchitecture) = arch.device

"""
    activate!(arch::SingleDeviceArchitecture; priority=:normal)

Activate the given architecture on the specified device and set the priority of the
backend. For the priority accepted values are `:normal`, `:low` and `:high`.
"""
function activate!(arch::SingleDeviceArchitecture; priority=:normal)
    set_device!(arch.device)
    KernelAbstractions.priority!(arch.backend, priority)
end

# CPU
get_device(::CPU, device_id) = nothing
set_device!(::Nothing) = nothing
heuristic_groupsize(::CPU, ::Val{N}) where {N} = 256

Base.unsafe_wrap(::CPU, ptr::Ptr, dims) = unsafe_wrap(Array, ptr, dims)

pointertype(::CPU, T::DataType) = Ptr{T}

# because of https://github.com/JuliaGPU/CUDA.jl/pull/2335
disable_task_sync!(::Array) = nothing
enable_task_sync!(::Array) = nothing

@inline modify_sync!(x::Array, fn::F) where F = fn(x)
@inline modify_sync!(::Tuple{}, ::F)  where F = nothing

@generated function modify_sync!(x::T, fn::F) where {T,F}
    names  = fieldnames(x)
    N      = length(names)
    quote
        @inline
        Base.@nexprs $N i -> begin
            args = getfield(x, $names[i])
            modify_sync!(args, fn)
        end
    end
end

end
