using Test
using Chmy

using KernelAbstractions

# testing for various floating point arithmetic precisions
precisions = [Float32, Float64]

# add KA backends
backends = KernelAbstractions.Backend[CPU()]

# do not test Float64 on Metal.jl
skip_Float64 = [false]

if get(ENV, "JULIA_CHMY_BACKEND", "") == "AMDGPU"
    using AMDGPU
    if AMDGPU.functional()
        push!(backends, ROCBackend())
        push!(skip_Float64, false)
    end
elseif get(ENV, "JULIA_CHMY_BACKEND", "") == "CUDA"
    using CUDA
    if CUDA.functional()
        push!(backends, CUDABackend())
        push!(skip_Float64, false)
    end
elseif get(ENV, "JULIA_CHMY_BACKEND", "") == "Metal"
    using Metal
    if Metal.functional()
        push!(backends, MetalBackend())
        push!(skip_Float64, true)
    end
end
