using Documenter
using Literate
using SuperfluidDynamics
using MPI

MPI.Init()
DocMeta.setdocmeta!(SuperfluidDynamics, :DocTestSetup,
                    :(using SuperfluidDynamics;
                      grid = Grid((32, 32), ((-12, 12), (-12, 12)));
                      field = Field(grid, ComplexField());
                      grid3 = Grid((32, 32, 32), ((-12,  12), (-12,  12), (-12,  12)));
                      field3 = Field(grid3, ComplexField());
                     ); recursive=true)

run_example = true

if run_example
    examples = [joinpath("examples/BEC_2D/BEC_2D.jl"),
                joinpath("examples/QT_TG_2D/QT_TG_2D.jl"),
                joinpath("examples/QT_TG_3D/QT_TG_3D.jl"),
                joinpath("examples/VR/VR_2D.jl"),
                joinpath("examples/NS_2D/NS_2D.jl"),
                joinpath("examples/NSGP_2D/NSGP_2D.jl"),
                joinpath("examples/HVBK_2D/HVBK_2D.jl"),
                joinpath("examples/BDG/BDG_2D.jl")]
    examples_md = ["generated/BEC_2D.md",
                   "generated/VR_2D.md",
                   "generated/QT_TG_2D.md",
                   "generated/QT_TG_3D.md",
                   "generated/NS_2D.md",
                   "generated/NSGP_2D.md",
                   "generated/HVBK_2D.md",
                   "generated/BDG_2D.md"]

    for example in examples
        Literate.markdown(example, "docs/src/generated"; flavor=Literate.DocumenterFlavor(),
                          credit=false)
    end
else
    examples_md = []
end

pages = ["Home" => "index.md",
         "QuickStart" => "quickstart.md",
         "Physic models" => ["Gross-Pitaevskii" => ["grosspitaevskii/grosspitaevskii.md",
                                                    "grosspitaevskii/init.md",
                                                    "grosspitaevskii/potential.md",
                                                    "grosspitaevskii/nummodel.md"],
                             "Bogoliubov-de Gennes" => "bdg.md",
                             "Navier-Stokes and two-fluid models" => "navierstokes.md"],
         "Library" => ["grid.md",
                       "field.md",
                       "plan.md",
                       "gradientfield.md"],
         "Plots" => ["plots/plots.md"],
         "API" => "api.md"]

if run_example
    insert!(pages, 5, "Examples" => examples_md)
end

makedocs(; authors="Corentin Lothode <corentin.lothode@inrae.fr> and contributors.",
         repo="https://plmlab.math.cnrs.fr/lothode/SuperfluidDynamics.jl",
         sitename="SuperfluidDynamics.jl",
         # `prettyurls=false`: page `foo.html` (not `foo/index.html`) so the
         # per-page Bonito/JSServe assets are referenced with a stable relative
         # path. `size_threshold=nothing`: the interactive WebGL figures are
         # inlined in the page HTML, which exceeds the default 200 KiB limit.
         format=Documenter.HTML(prettyurls=false, size_threshold=nothing),
         doctest=false,
         warnonly=true,   # old `strict=false`: never fail the build, only warn
         clean=true,
         modules=[SuperfluidDynamics],
         pages=pages)

# Documenter can also automatically deploy documentation to gh-pages.
# See "Hosting Documentation" and deploydocs() in the Documenter manual
# for more information.
#=deploydocs(
    repo = "<repository url>"
)=#
