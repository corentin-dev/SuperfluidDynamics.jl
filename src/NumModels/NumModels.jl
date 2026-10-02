"Abstract supertype for numerical models."
abstract type AbstractNumModel{AbstractField,AbstractParameters,AbstractPlan} end

include("GrossPitaevskii/GrossPitaevskii.jl")
include("BdG/BdG.jl")
include("NavierStokes/NavierStokes.jl")
include("NavierStokes/NSGP.jl")

export NumModelSplit1, NumModelSplit2, NumModelGPRK
export NumModelRK4Imp
export NumModelBdG, bdg_mu, bdg_apply!
export NSGPParameters, NumModelNSGP
export NumModelBackwardEuler, NumModelBackwardEulerNoPrecond, NumModelBackwardEulerNL
export NumModelCrankNicolson, NumModelCrankNicolsonQuasiNewton, NumModelCrankNicolsonT,
       NumModelCrankNicolsonQuasiNewtonT
export NumModelExternalVelocity
export solve!

include("krylov.jl")

include("derivatives.jl")

function solve!(n::AbstractNumModel; istart=1, plot=false)
    if plot
        @eval using SuperfluidDynamics.Plots
        p = Plot(n.f)
        createPlot!(p)
    end
    res = Tuple{Int64,Int64,Float64,Float64,Float64,Float64}[]
    nkrylovtotal = 0
    write!(n.writers; prefix="res", istep=istart, Δt=n.Δt)
    for it in istart:(istart + n.niter)
        println_parallel("iteration $(it)")
        E = energy(n, true)
        nkrylov = timeStep!(n)
        nkrylovtotal += nkrylov
        push!(res, (nkrylov, nkrylovtotal, E...))
        if it % n.freqbckp == 0
            write!(n.writers; prefix="res", istep=it, Δt=n.Δt)
        end
        if plot
            updatePlot!(p)
        end
    end
    return res
end
