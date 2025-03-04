module ChmyCUDAExt

using CUDA, KernelAbstractions

import Chmy.Architectures: heuristic_groupsize, set_device!, get_device, pointertype, disable_task_sync!

Base.unsafe_wrap(::CUDABackend, ptr::CuPtr, dims) = unsafe_wrap(CuArray, ptr, dims)

pointertype(::CUDABackend, T::DataType) = CuPtr{T}

disable_task_sync!(::CUDABackend, array) = CUDA.unsafe_disable_task_sync!(array)

set_device!(dev::CuDevice) = CUDA.device!(dev)

get_device(::CUDABackend, id::Integer) = CuDevice(id - 1)

heuristic_groupsize(::CUDABackend, ::Val{1}) = (256,)
heuristic_groupsize(::CUDABackend, ::Val{2}) = (32, 8)
heuristic_groupsize(::CUDABackend, ::Val{3}) = (32, 8, 1)

end
