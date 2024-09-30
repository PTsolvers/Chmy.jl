module ChmyMetalExt

using Metal, KernelAbstractions

import Chmy.Architectures: heuristic_groupsize, set_device!, get_device, pointertype

Base.unsafe_wrap(::MetalBackend, ptr::Metal.MtlPtr, dims) = unsafe_wrap(MtlArray, ptr, dims)

pointertype(::MetalBackend, T::DataType) = Metal.MtlPtr{T}

set_device!(dev::Metal.MTL.MTLDeviceInstance) = Metal.device!(dev)

get_device(::MetalBackend, id::Integer) = Metal.MTL.MTLDevice(id)

heuristic_groupsize(::MetalBackend, ::Val{1}) = (256,)
heuristic_groupsize(::MetalBackend, ::Val{2}) = (32, 8)
heuristic_groupsize(::MetalBackend, ::Val{3}) = (32, 8, 1)

end
