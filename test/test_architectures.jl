include("common.jl")

for backend in TEST_BACKENDS
    @testset "$(basename(@__FILE__)) (backend: $backend)" begin
        @testset "SingleDeviceArchitecture" begin
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

        @testset "DistributedArchitecture" begin
            using MPI
            MPI.Init()
            device = get_device(backend, 1)
            arch = Arch(backend, MPI.COMM_WORLD, (0, 0))
            # Test for instantiation of SingleDeviceArchitecture
            @testset "DistributedArchitecture instantiation" begin
                @test arch isa DistributedArchitecture
                @test arch.child_arch isa SingleDeviceArchitecture
                @test arch.topology isa CartesianTopology
            end
            # Test for functionality of methods
            @testset "Method functionalities" begin
                @test get_backend(arch) == backend
                @test get_device(arch) == device
            end
            @testset "GPU-awareness" begin
                @test is_gpu_aware(arch) == Chmy.Architectures.gpu_aware_compat(backend)

                arch = Arch(backend, MPI.COMM_WORLD, (0, 0); gpu_aware=false)
                if Chmy.Architectures.gpu_aware_compat(backend)
                    @test is_gpu_aware(arch) == false
                end

                arch = Arch(backend, MPI.COMM_WORLD, (0, 0); gpu_aware=true)
                if !Chmy.Architectures.gpu_aware_compat(backend)
                    @test is_gpu_aware(arch) == false
                end
            end
        end
    end
end
