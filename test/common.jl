using Test
using Chmy

using KernelAbstractions

compatible(::Backend, ::DataType) = true

# number types to test
TEST_TYPES = [Float32, Float64]

# add backends
TEST_BACKENDS = []

if haskey(ENV, "JULIA_CHMY_BACKEND_CPU")
    push!(TEST_BACKENDS, CPU())
end

if haskey(ENV, "JULIA_CHMY_BACKEND_CUDA")
    using CUDA
    if CUDA.functional()
        push!(TEST_BACKENDS, CUDABackend())
    end
end

if haskey(ENV, "JULIA_CHMY_BACKEND_AMDGPU")
    using AMDGPU
    if AMDGPU.functional()
        push!(TEST_BACKENDS, ROCBackend())
    end
end

if haskey(ENV, "JULIA_CHMY_BACKEND_Metal")
    using Metal

    function compatible(::MetalBackend, T::DataType)
        try
            Metal.check_eltype(T)
            return true
        catch
            return false
        end
    end

    if Metal.functional()
        push!(TEST_BACKENDS, MetalBackend())
    end
end
