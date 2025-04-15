module ChmyCUDAExt

using CUDA, KernelAbstractions

import Chmy.Architectures: heuristic_groupsize, set_device!, get_device, pointertype, gpu_aware_compat, disable_task_sync!, enable_task_sync!

Base.unsafe_wrap(::CUDABackend, ptr::CuPtr, dims) = unsafe_wrap(CuArray, ptr, dims)

pointertype(::CUDABackend, T::DataType) = CuPtr{T}

# because of https://github.com/JuliaGPU/CUDA.jl/pull/2335
disable_task_sync!(x::CuArray) = CUDA.enable_synchronization!(x, false)
enable_task_sync!(x::CuArray) = CUDA.enable_synchronization!(x, true)

set_device!(dev::CuDevice) = CUDA.device!(dev)

get_device(::CUDABackend, id::Integer) = CuDevice(id - 1)

heuristic_groupsize(::CUDABackend, ::Val{1}) = (256,)
heuristic_groupsize(::CUDABackend, ::Val{2}) = (32, 8)
heuristic_groupsize(::CUDABackend, ::Val{3}) = (32, 8, 1)

gpu_aware_compat(::CUDABackend) = true

end
