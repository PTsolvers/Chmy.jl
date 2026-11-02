using Test
using Chmy

using Pkg

# distributed
using MPI

EXCLUDE_TESTS = []

istest(f)    = startswith(f, "test_") && endswith(f, ".jl")
ismpitest(f) = startswith(f, "mpitest_") && endswith(f, ".jl")

function parse_flag(args, flag; default=nothing, type::DataType=Nothing)
    key = findfirst(arg -> startswith(arg, flag), args)

    if isnothing(key)
        # flag not found
        return false, default
    elseif args[key] == flag
        # flag found but no value
        return true, default
    end

    splitarg = split(args[key], '=')

    if splitarg[1] != flag
        # argument started with flag but is not the flag
        return false, default
    end

    values = strip.(split(splitarg[2], ','))

    if type <: Nothing || type <: AbstractString
        # common cases, return as strings
        return true, values
    elseif !(type <: Number) || !isbitstype(type)
        error("type must be a bitstype number, got '$type'")
    end

    return true, parse.(Ref(type), values)
end

function runtests(nranks)
    testdir      = @__DIR__
    testfiles    = sort(filter(istest, readdir(testdir)))
    mpitestfiles = sort(filter(ismpitest, readdir(testdir)))

    nfail = 0
    printstyled("Testing package Chmy.jl\n"; bold=true, color=:white)

    for f in testfiles
        println("")
        if f ∈ EXCLUDE_TESTS
            @info "Skip test:" f
            continue
        end
        try
            run(`$(Base.julia_cmd()) --startup-file=no $(joinpath(testdir, f))`)
        catch ex
            @error ex
            nfail += 1
        end
    end

    # multi-rank tests (`--mpi=N`), launched with the mpiexec configured for MPI.jl
    if nranks > 0 && Sys.iswindows()
        @info "Skip MPI tests on Windows"
    elseif nranks > 0
        for f in mpitestfiles
            println("")
            try
                run(`$(MPI.mpiexec()) -n $nranks $(Base.julia_cmd()) --startup-file=no $(joinpath(testdir, f))`)
            catch ex
                @error ex
                nfail += 1
            end
        end
    else
        @info "Skip MPI tests (pass --mpi=N to run them on N ranks)"
    end
    return nfail
end

_, backends = parse_flag(ARGS, "--backends"; default=["CPU"])
_, nranks   = parse_flag(ARGS, "--mpi"; default=[0], type=Int)

for backend in backends
    if backend != "CPU"
        # preferences only apply to direct deps of the test env, so add CUDA_Runtime_jll explicitly
        # to honour a preset local CUDA toolkit (e.g. CSCS uenv), as in a regular `--project` run
        pkgs = backend == "CUDA" ? ["CUDA", "CUDA_Runtime_jll"] : [backend]
        # skip auto-precompilation: on Julia 1.13.2 it fails with "Unknown package" when the add
        # changes indirect deps (JuliaLang/Pkg.jl#4866); test files precompile on load anyway
        withenv("JULIA_PKG_PRECOMPILE_AUTO" => "0") do
            Pkg.add(pkgs)
        end
    end
    ENV["JULIA_CHMY_BACKEND_$backend"] = true
end

exit(runtests(first(nranks)))
