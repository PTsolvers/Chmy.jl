using Documenter
using DocumenterVitepress

using Chmy

makedocs(
    sitename = "Chmy.jl",
    authors="Ivan Utkin, Ludovic Räss and contributors",
    format = DocumenterVitepress.MarkdownVitepress(
        repo = "github.com/PTsolvers/Chmy.jl",
        devbranch = "main",
        devurl = "dev",
    ),
    modules = [Chmy],
    warnonly = [:missing_docs],
    pages = [
        "Home" => "index.md",
        "Getting Started" => [
            "getting_started/introduction.md",
            "getting_started/using_chmy_with_mpi.md"
        ],
        "Concepts" => [
            "concepts/architectures.md",
            "concepts/grids.md",
            "concepts/fields.md",
            "concepts/bc.md",
            "concepts/grid_operators.md",
            "concepts/kernels.md",
            "concepts/distributed.md"
        ],
        "Examples" => ["examples/overview.md"],
        "Library" => ["lib/modules.md"],
        "Developer Doc" => [
            "developer_doc/running_tests.md",
            "developer_doc/workers.md"
        ],
    ]
)

deploydocs(
    repo = "github.com/PTsolvers/Chmy.jl",
    devbranch = "main",
    push_preview = true
)
