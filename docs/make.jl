using Pkg
using Documenter, PicoQuant


makedocs(
    modules = [PicoQuant],
    clean = false,
    sitename = "PicoQuant.jl",
    pages = Any[
        "Home" => "index.md",
        "Manual" => Any[
            "Layers" => Any[ "Layer 3" => "layers/layer3.md", "Layer 2" => "layers/layer2.md", "Layer 1" => "layers/layer1.md" ],
            "Backends" => "backends/backend.md",
            "Algorithms" => "algo/algos.md",
            "Visualisation" => "visuals.md",
        ],
    ]
)
deploydocs(
    repo = "github.com/ICHEC/PicoQuant.jl.git",
)
