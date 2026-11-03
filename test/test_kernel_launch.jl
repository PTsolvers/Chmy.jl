include("common.jl")

import Chmy.Architectures: disable_task_sync!, enable_task_sync!, task_sync_enabled
import Chmy.KernelLaunch: deepmap!, with_no_task_sync!

# deepmap! must not traverse the internals of the backends' array types
const TRAVERSABLE = (Core, Base, Main, Chmy, Chmy.Fields, Chmy.Grids)
traversable(x) = x isa AbstractArray || parentmodule(typeof(x)) ∈ TRAVERSABLE

# explicit diffusion step into a second field (an in-place update would race between inner and outer kernels)
@kernel inbounds = true function diffusion_step!(Cn, Co, Δt, g::StructuredGrid, O)
    I = @index(Global, NTuple)
    I = I + O
    Cn[I...] = Co[I...] + Δt * lapl(Co, g, I...)
end

for backend in TEST_BACKENDS, T in TEST_TYPES
    if !compatible(backend, T)
        continue
    end

    @testset "$(basename(@__FILE__)) (backend: $backend, type: $T)" begin
        arch = Arch(backend)
        grid = UniformGrid(arch; origin=(T(0), T(0)), extent=(T(1), T(1)), dims=(6, 4))
        Δt   = T(1)
        C    = Field(backend, grid, Center())
        q    = VectorField(backend, grid)
        args = (C, q, Δt, grid)
        A    = typeof(parent(C)) # array type of the backend

        @testset "deepmap! traversal" begin
            visited = Any[]
            record!(x) = (push!(visited, x); nothing)

            @testset "single Field" begin
                empty!(visited)
                deepmap!(record!, C)
                @test count(x -> x isa A, visited) == 1
                @test any(x -> x === parent(C), visited)
                @test all(traversable, visited)
            end
            @testset "multiple args" begin
                empty!(visited)
                deepmap!(record!, args)
                @test count(x -> x isa A, visited) == 3
                @test all(traversable, visited)
            end
            @testset "views reach the parent array" begin
                empty!(visited)
                deepmap!(record!, interior(C))
                @test any(x -> x === parent(C), visited)
                @test all(traversable, visited)
            end
            @testset "nested containers and closures" begin
                empty!(visited)
                f = let C = C
                    () -> C
                end
                deepmap!(record!, (a=1, b=(C, (2, ())), c=f))
                @test count(x -> x === parent(C), visited) == 2
                @test all(traversable, visited)
            end
        end

        @testset "implicit synchronization" begin
            flags() = (task_sync_enabled(parent(C)), task_sync_enabled(parent(q.x)), task_sync_enabled(parent(q.y)))

            ran = Ref(false)
            with_no_task_sync!(() -> (ran[] = true), backend, args)
            @test ran[]

            if isnothing(task_sync_enabled(parent(C)))
                # the backend has no (toggleable) implicit synchronization: the toggles are no-ops
                deepmap!(disable_task_sync!, args)
                @test flags() == (nothing, nothing, nothing)
                deepmap!(enable_task_sync!, args)
                @test flags() == (nothing, nothing, nothing)
            else
                @test flags() == (true, true, true)
                deepmap!(disable_task_sync!, args)
                @test flags() == (false, false, false)
                deepmap!(enable_task_sync!, args)
                @test flags() == (true, true, true)

                @testset "suspended within with_no_task_sync!" begin
                    inside = Ref{Any}(nothing)
                    with_no_task_sync!(backend, args) do
                        inside[] = flags()
                    end
                    @test inside[] == (false, false, false)
                    @test flags() == (true, true, true)
                end
                @testset "restored after an exception" begin
                    @test_throws ErrorException with_no_task_sync!(() -> error("kernel failed"), backend, args)
                    @test flags() == (true, true, true)
                end
            end
        end

        # the worker task of an overlapped launcher activates the architecture (device, stream priority)
        if isdefined(KernelAbstractions, :device)
            @testset "worker runs on the device of the architecture" begin
                launcher = Launcher(arch, grid; outer_width=(1, 1))
                seen = Ref{Any}(nothing)
                put!(() -> (seen[] = KernelAbstractions.device(backend)), launcher.workers)
                wait(launcher.workers)
                @test seen[] == KernelAbstractions.device(backend)
            end
        end

        @testset "overlapped launch matches plain launch" begin
            for (dims, outer_width) in (((32,), (4,)), ((32, 16), (4, 2)), ((16, 12, 8), (4, 2, 2)))
                N  = length(dims)
                g  = UniformGrid(arch; origin=ntuple(_ -> T(-1), N), extent=ntuple(_ -> T(2), N), dims=dims)
                dt = minimum(spacing(g))^2 / N / T(2.1)
                function diffuse(launch; nt=20)
                    Co = Field(backend, g, Center())
                    Cn = Field(backend, g, Center())
                    set!(Co, g, (x...) -> exp(-sum(abs2, x)))
                    bc!(arch, g, Co => Neumann())
                    for _ in 1:nt
                        launch(arch, g, diffusion_step! => (Cn, Co, dt, g); bc=batch(g, Cn => Neumann()))
                        Co, Cn = Cn, Co
                    end
                    return Array(interior(Co))
                end
                @testset "$(N)D" begin
                    C_plain = diffuse(Launcher(arch, g))
                    C_ovl   = diffuse(Launcher(arch, g; outer_width))
                    @test all(isfinite, C_plain)
                    @test C_ovl ≈ C_plain
                end
            end
        end
    end
end
