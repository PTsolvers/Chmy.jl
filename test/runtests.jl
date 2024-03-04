using Test
using Chmy

using Pkg

excludedfiles = ["test_excluded.jl"]

# distributed
using MPI

function parse_flags!(args, flag; default=nothing, typ=typeof(default))
    for f in args
        startswith(f, flag) || continue

        if f != flag
            val = split(f, '=')[2]
            if !(typ ≡ nothing || typ <: AbstractString)
                @show typ val
                val = parse(typ, val)
            end
        else
            val = default
        end

        filter!(x -> x != f, args)
        return true, val
    end
    return false, default
end

function runtests()
    testdir   = pwd()
    istest(f) = endswith(f, ".jl") && startswith(basename(f), "test_")
    testfiles = sort(filter(istest, vcat([joinpath.(root, files) for (root, dirs, files) in walkdir(testdir)]...)))

    nfail = 0
    printstyled("Testing package Chmy.jl\n"; bold=true, color=:white)

    for f in testfiles
        println("")
        if basename(f) ∈ excludedfiles
            println("Test Skip:")
            println("$f")
            continue
        end
        try
            # if basename(f) ∈ test_distributed
            #     nprocs = contains(f, "2D") ? nprocs_2D : nprocs_3D
            #     cmd(n=nprocs) = `$(mpiexec()) -n $n $(Base.julia_cmd()) --startup-file=no --color=yes $(joinpath(testdir, f))`
            #     run(cmd())
            # else
                run(`$(Base.julia_cmd()) --startup-file=no $(joinpath(testdir, f))`)
            # end
        catch ex
            @error ex
            nfail += 1
        end
    end
    return nfail
end

_, backend_name = parse_flags!(ARGS, "--backend"; default="CPU", typ=String)

@static if backend_name == "AMDGPU"
    Pkg.add("AMDGPU")
    ENV["JULIA_CHMY_BACKEND"] = "AMDGPU"
elseif backend_name == "CUDA"
    Pkg.add("CUDA")
    ENV["JULIA_CHMY_BACKEND"] = "CUDA"
end

exit(runtests())
