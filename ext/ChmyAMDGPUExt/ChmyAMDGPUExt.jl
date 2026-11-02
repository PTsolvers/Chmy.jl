module ChmyAMDGPUExt

using AMDGPU, KernelAbstractions

import Chmy.Architectures: heuristic_groupsize, set_device!, get_device, pointertype, gpu_aware_compat, disable_task_sync!, enable_task_sync!, task_sync_enabled

Base.unsafe_wrap(::ROCBackend, ptr::Ptr, dims) = unsafe_wrap(ROCArray, ptr, dims)

pointertype(::ROCBackend, T::DataType) = Ptr{T}

# available from AMDGPU.jl 2.11 on; earlier versions always synchronize (no overlap)
if isdefined(AMDGPU, :enable_synchronization!) && isdefined(AMDGPU, :synchronization_enabled)
    disable_task_sync!(x::ROCArray) = AMDGPU.enable_synchronization!(x, false)
    enable_task_sync!(x::ROCArray)  = AMDGPU.enable_synchronization!(x, true)
    task_sync_enabled(x::ROCArray)  = AMDGPU.synchronization_enabled(x)
end

set_device!(dev::HIPDevice) = AMDGPU.device!(dev)

get_device(::ROCBackend, id::Integer) = HIPDevice(id)

heuristic_groupsize(::ROCBackend, ::Val{1}) = (256,)
heuristic_groupsize(::ROCBackend, ::Val{2}) = (128, 2)
heuristic_groupsize(::ROCBackend, ::Val{3}) = (128, 2, 1)

gpu_aware_compat(::ROCBackend) = true

end
