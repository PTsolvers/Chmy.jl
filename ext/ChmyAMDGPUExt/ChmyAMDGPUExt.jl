module ChmyAMDGPUExt

using AMDGPU, KernelAbstractions

import Chmy.Architectures: heuristic_groupsize, set_device!, get_device, pointertype, gpu_aware_compat

Base.unsafe_wrap(::ROCBackend, ptr::Ptr, dims) = unsafe_wrap(ROCArray, ptr, dims)

pointertype(::ROCBackend, T::DataType) = Ptr{T}

set_device!(dev::HIPDevice) = AMDGPU.device!(dev)

get_device(::ROCBackend, id::Integer) = HIPDevice(id)

heuristic_groupsize(::ROCBackend, ::Val{1}) = (256,)
heuristic_groupsize(::ROCBackend, ::Val{2}) = (128, 2)
heuristic_groupsize(::ROCBackend, ::Val{3}) = (128, 2, 1)

gpu_aware_compat(::ROCBackend) = true

end
