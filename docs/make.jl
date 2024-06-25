using Documenter
using Chmy

push!(LOAD_PATH,"../src/")

makedocs(
    sitename = "Chmy",
    authors="Ivan Utkin, Ludovic Räss and contributors",
    format = Documenter.HTML(
        prettyurls=get(ENV, "CI", nothing) == "true", # easier local build
        ansicolor=true
        ),
    modules = [Chmy],
    warnonly = [:missing_docs],
    pages = Any[
        "Home" => "index.md",
        "Concepts" => Any["concepts/overview.md",
                        "concepts/architectures.md",
                        "concepts/grids.md",
                        "concepts/fields.md",
                        "concepts/bc.md",
                        "concepts/grid_operators.md",
                        "concepts/workers.md", 
                        "concepts/kernels.md"
        ],
        "Examples" => Any["examples/overview.md",
                          "examples/diffusion_2d.md",
                          "examples/diffusion_2d_mpi.md",
                          "examples/diffusion_2d_perf.md",
                          "examples/batcher.md"
        ],
        "Usage" => Any["usage/runtests.md"],
        "Library" => Any["lib/modules.md"]
    ]
)

deploydocs(
    repo = "github.com/PTsolvers/Chmy.jl.git",
    devbranch = "main",
    push_preview = true
)
