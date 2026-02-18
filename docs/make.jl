using Documenter
using DocumenterVitepress

using Chmy

makedocs(
    sitename = "Chmy.jl",
    authors="Ivan Utkin and contributors",
    format = DocumenterVitepress.MarkdownVitepress(
        repo = "github.com/PTsolvers/Chmy.jl",
        devbranch = "main",
        devurl = "dev",
    ),
    modules = [Chmy],
    warnonly = [:missing_docs],
    pages = [
        "Home" => "index.md",
    ]
)

deploydocs(
    repo = "github.com/PTsolvers/Chmy.jl",
    devbranch = "main",
    push_preview = true
)
