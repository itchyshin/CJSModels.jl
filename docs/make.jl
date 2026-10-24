using Documenter, CJSModels

makedocs(
    modules = [CJSModels],
    checkdocs = :exports,
    sitename = "CJSModels.jl",
    authors = "Shinichi Nakagawa",
    remotes = nothing,
    format = Documenter.HTML(prettyurls = false, edit_link = nothing, repolink = "https://github.com/itchyshin/CJSModels.jl"),
    pages = [
        "Home" => "index.md",
        "Fixed-effects tutorial" => "tutorial.md",
        "API reference" => "reference.md",
        "Tested scope" => "status.md",
    ],
)

# deploydocs normally supplies this file; local unversioned previews need it too.
write(joinpath(@__DIR__, "build", "siteinfo.js"),
      "var DOCUMENTER_VERSION_SELECTOR_DISABLED = true;\n")
# Bundle version metadata locally so a standalone preview has no parent-site dependency.
write(joinpath(@__DIR__, "build", "versions.js"), "var DOC_VERSIONS = [];\n")
for page in filter(p -> endswith(p, ".html"), readdir(joinpath(@__DIR__, "build"); join=true))
    write(page, replace(read(page, String), "../versions.js" => "versions.js"))
end
