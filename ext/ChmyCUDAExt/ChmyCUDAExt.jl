module ChmyCUDAExt

using CUDA, KernelAbstractions, Chmy

Base.unsafe_wrap(::CUDABackend, ptr::CuPtr, dims) = unsafe_wrap(CuArray, ptr, dims)

Chmy.pointertype(::CUDABackend, T::DataType) = CuPtr{T}

end
