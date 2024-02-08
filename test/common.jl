using Test
using Chmy

using KernelAbstractions

# add KA backends
backends = KernelAbstractions.Backend[CPU()]

if get(ENV, "JULIA_CHMY_BACKEND", "") == "AMDGPU"
    using AMDGPU
    AMDGPU.functional() && push!(backends, ROCBackend())
elseif get(ENV, "JULIA_CHMY_BACKEND", "") == "CUDA"
    using CUDA
    CUDA.functional() && push!(backends, CUDABackend())
end
