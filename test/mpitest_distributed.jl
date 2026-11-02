# Multi-rank tests: run with `mpiexec -n N julia mpitest_distributed.jl` (see runtests.jl, `--mpi=N`) or as a Slurm job step.
include("common.jl")

using MPI
MPI.Init()

# GPU-aware MPI only where the launcher environment provides it (e.g. CSCS: MPICH_GPU_SUPPORT_ENABLED=1)
const GPU_AWARE = get(ENV, "JULIA_CHMY_TEST_GPU_AWARE", "0") == "1"

# one visible GPU per rank (e.g. Slurm --gpus-per-task=1) -> device 1; otherwise Chmy picks the device by node-local rank
function device_id(backend)
    isdefined(KernelAbstractions, :ndevices) || return nothing
    ndev = try
        KernelAbstractions.ndevices(backend)
    catch
        return nothing
    end
    return ndev == 1 ? 1 : nothing
end

# explicit diffusion step into a second field (an in-place update would race between inner and outer kernels)
@kernel inbounds = true function diffusion_step!(Cn, Co, Δt, g::StructuredGrid, O)
    I = @index(Global, NTuple)
    I = I + O
    Cn[I...] = Co[I...] + Δt * lapl(Co, g, I...)
end

fun(x, y) = x + 10y

for backend in TEST_BACKENDS, T in TEST_TYPES
    if !compatible(backend, T)
        continue
    end

    arch   = Arch(backend, MPI.COMM_WORLD, (0, 0); device_id=device_id(backend), gpu_aware=GPU_AWARE)
    topo   = topology(arch)
    me     = global_rank(topo)
    nprocs = global_size(topo)
    dims_l = (8, 6)
    dims_g = dims_l .* dims(topo)
    grid   = UniformGrid(arch; origin=(T(0), T(0)), extent=(T(1), T(1)), dims=dims_g)

    @testset "$(basename(@__FILE__)) (backend: $backend, type: $T, ranks: $nprocs)" begin
        @testset "topology" begin
            @test prod(dims(topo)) == nprocs
            @test MPI.Comm_size(cart_comm(topo)) == nprocs
            @test is_gpu_aware(arch) == (GPU_AWARE && Chmy.Architectures.gpu_aware_compat(backend))
            @test size(grid, Center()) == dims_l
        end

        # the halo of a field holds the neighbour's values (connected sides) or stays untouched (bounded sides)
        @testset "exchange_halo!" begin
            for loc in ((Center(), Center()), (Vertex(), Center()), (Center(), Vertex()))
                f = Field(backend, grid, loc)
                set!(f, grid, fun) # the local grid has global coordinates
                exchange_halo!(arch, grid, f)
                KernelAbstractions.synchronize(backend)
                A  = interior(f; with_halo=true) |> Array
                nx, ny = size(f)
                E  = T[fun(coord(grid, loc, i, j)...) for i in 0:nx+1, j in 0:ny+1]
                connectivity(grid, Dim(1), Side(1)) isa Bounded && (E[1, :] .= 0)
                connectivity(grid, Dim(1), Side(2)) isa Bounded && (E[end, :] .= 0)
                connectivity(grid, Dim(2), Side(1)) isa Bounded && (E[:, 1] .= 0)
                connectivity(grid, Dim(2), Side(2)) isa Bounded && (E[:, end] .= 0)
                @test A ≈ E
            end
        end

        @testset "gather!" begin
            f = Field(backend, grid, Center())
            set!(f, grid, fun)
            dst = me == 0 ? zeros(T, dims_g) : nothing
            gather!(arch, dst, f)
            if me == 0
                grid_g = UniformGrid(SingleDeviceArchitecture(arch); origin=(T(0), T(0)), extent=(T(1), T(1)), dims=dims_g)
                @test dst ≈ T[fun(coord(grid_g, Center(), i, j)...) for i in 1:dims_g[1], j in 1:dims_g[2]]
            end
        end

        # the overlapped launch (inner kernel, outer kernels, boundary conditions and halo exchange on worker tasks)
        # must match the plain launch of the same kernel
        @testset "overlapped launch with exchange matches plain launch" begin
            dt = minimum(spacing(grid))^2 / 2 / T(2.1)
            function diffuse(launch; nt=20)
                Co = Field(backend, grid, Center())
                Cn = Field(backend, grid, Center())
                set!(Co, grid, (x, y) -> exp(-(x - oftype(x, 0.5))^2 - (y - oftype(y, 0.5))^2))
                bc!(arch, grid, Co => Neumann(); exchange=Co)
                for _ in 1:nt
                    launch(arch, grid, diffusion_step! => (Cn, Co, dt, grid); bc=batch(grid, Cn => Neumann(); exchange=Cn))
                    Co, Cn = Cn, Co
                end
                return Array(interior(Co))
            end
            C_plain = diffuse(Launcher(arch, grid))
            C_ovl   = diffuse(Launcher(arch, grid; outer_width=(2, 2)))
            @test all(isfinite, C_plain)
            @test C_ovl ≈ C_plain
        end
    end
end

MPI.Finalize()
