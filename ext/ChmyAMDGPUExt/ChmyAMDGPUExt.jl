module ChmyAMDGPUExt

using AMDGPU, KernelAbstractions

import Chmy.Architectures: heuristic_groupsize, set_device!, get_device, pointertype, disable_task_sync!

Base.unsafe_wrap(::ROCBackend, ptr::Ptr, dims) = unsafe_wrap(ROCArray, ptr, dims; lock=false)

pointertype(::ROCBackend, T::DataType) = Ptr{T}

disable_task_sync!(::ROCBackend, array::ROCArray) = array

set_device!(dev::HIPDevice) = AMDGPU.device!(dev)

get_device(::ROCBackend, id::Integer) = HIPDevice(id)

heuristic_groupsize(::ROCBackend, ::Val{1}) = (256,)
heuristic_groupsize(::ROCBackend, ::Val{2}) = (128, 2)
heuristic_groupsize(::ROCBackend, ::Val{3}) = (128, 2, 1)

end
