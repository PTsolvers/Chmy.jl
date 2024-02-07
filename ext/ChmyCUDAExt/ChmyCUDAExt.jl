module ChmyCUDAExt

using CUDA, CUDA.CUDAKernels, KernelAbstractions, Chmy

import Chmy.Architectures: heuristic_groupsize, set_device!, get_device

Base.unsafe_wrap(::CUDABackend, ptr::CuPtr, dims) = unsafe_wrap(CuArray, ptr, dims)

Chmy.pointertype(::CUDABackend, T::DataType) = CuPtr{T}

set_device!(dev::CuDevice) = CUDA.device!(dev)

get_device(::CUDABackend, id::Integer) = CuDevice(id - 1)

heuristic_groupsize(::CuDevice, ::Val{1}) = (256,)
heuristic_groupsize(::CuDevice, ::Val{2}) = (32, 8)
heuristic_groupsize(::CuDevice, ::Val{3}) = (32, 8, 1)

end
