module ChmyAMDGPUExt

using AMDGPU, AMDGPU.ROCKernels, KernelAbstractions, Chmy

import Chmy.Architectures: heuristic_groupsize, set_device!, get_device

Base.unsafe_wrap(::ROCBackend, ptr::Ptr, dims) = unsafe_wrap(ROCArray, ptr, dims; lock=false)

Chmy.pointertype(::ROCBackend, T::DataType) = Ptr{T}

set_device!(dev::HIPDevice) = AMDGPU.device!(dev)

get_device(::ROCBackend, id::Integer) = HIPDevice(id)

heuristic_groupsize(::HIPDevice, ::Val{1}) = (256, )
heuristic_groupsize(::HIPDevice, ::Val{2}) = (128, 2, )
heuristic_groupsize(::HIPDevice, ::Val{3}) = (128, 2, 1, )

end
