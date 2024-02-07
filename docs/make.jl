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
        "Usage" => Any["usage/runtests.md"],
        "Library" => Any["lib/modules.md"]
    ]
)

deploydocs(
    repo = "github.com/PTsolvers/Chmy.jl.git",
    devbranch = "lr/doc-ci",
    push_preview = true
)
