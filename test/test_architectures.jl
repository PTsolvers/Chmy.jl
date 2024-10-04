include("common.jl")

using Chmy.Architectures

for backend in TEST_BACKENDS
    @testset "$(basename(@__FILE__)) (backend: $backend)" begin
        device = get_device(backend, 1)
        arch = SingleDeviceArchitecture(backend, device)
        # Test for instantiation of SingleDeviceArchitecture
        @testset "SingleDeviceArchitecture instantiation" begin
            @test arch isa SingleDeviceArchitecture
            @test arch.backend == backend
            @test arch.device == device
        end
        @testset "SingleDeviceArchitecture methods" begin
            arch2 = SingleDeviceArchitecture(arch)
            @test arch2.backend == arch.backend
            @test arch2.device == arch.device
        end
        # Test for functionality of methods
        @testset "Method functionalities" begin
            @test Architectures.get_backend(arch) == backend
            @test Architectures.get_device(arch) == device
            @test Architectures.set_device!(arch.device) == device
        end
    end
end
