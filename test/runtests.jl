using Test
using Chmy

using Pkg

# distributed
using MPI

EXCLUDE_TESTS = []

istest(f) = startswith(f, "test_") && endswith(f, ".jl")

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

function runtests()
    testdir   = @__DIR__
    testfiles = sort(filter(istest, readdir(testdir)))

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
    return nfail
end

_, backends = parse_flag(ARGS, "--backends"; default=["CPU"])

for backend in backends
    if backend != "CPU"
        Pkg.add(backend)
    end
    # tmp fix to have the disable/enable task sync feature until merged in CUDA.jl
    if backend == "CUDA"
        Pkg.add(url="https://github.com/JuliaGPU/CUDA.jl", rev="vc/unsafe_stream_switching")
    end
    ENV["JULIA_CHMY_BACKEND_$backend"] = true
end

exit(runtests())
