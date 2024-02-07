module ChmyAMDGPUExt

using AMDGPU, KernelAbstractions, Chmy

Base.unsafe_wrap(::ROCBackend, ptr::Ptr, dims) = unsafe_wrap(ROCArray, ptr, dims; lock=false)

Chmy.pointertype(::ROCBackend, T::DataType) = Ptr{T}

end
