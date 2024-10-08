include("common.jl")

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
            @test get_backend(arch) == backend
            @test get_device(arch) == device
            @test set_device!(arch.device) == device
        end
    end
end
