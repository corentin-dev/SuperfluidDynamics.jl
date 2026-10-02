# # 3D Quantum Turbulence
#
# ## Introduction
#
# To test quantum turbulence with this package, we first relax a Gross-Pitaevskii
# field to a stationary state under a strong external flow, then restart an
# unsteady simulation from it.
#
# We first load the package, in order to have all the constructors and functions available.

using SuperfluidDynamics

# We setup the topology used, if we want to use `MPI` for parallelization.
# In this case, run with `mpirun -np 4 julia --project examples/QT_TG_3D/QT_TG_3D.jl`
# if $4$ is the desired number of processes.

# Here, the topoology is a pencil topology, meaning that $y$ and $z$ directions can be
# decomposed (if $n_\text{proc}$ is not prime). The pencil is transposed for some operations
# (FFT, finite differences, etc).

mpi_topo = SuperfluidDynamics.MPITopo2D()

# We use [`Makie`](https://makie.juliaplots.org/stable/) (with `WGLMakie`,
# which produces interactive WebGL plots in the documentation) for the plots:

using WGLMakie # hide
WGLMakie.activate!() # hide
WGLMakie.Bonito.Page(; exportable=true, offline=true) # hide

# If a checkpoint exists (git-LFS `docs/save`), we restart from it so the page
# does not have to recompute the full stationary relaxation:

use_saves = true # hide
#if you want to re-run the page, change to false # hide

# We setup the wanted discretization. We want $n_x \times n_y \times n_z = 128\times 128\times 128$.
# The domain bounds are set to $[0,2\pi]\times[0,2\pi]\times[0,2\pi]$.

# With those informations, we can create `Grid`, using the constructor:

nx = 85
ny = 85
nz = 85

xrange = (0, 2 * π)
yrange = (0, 2 * π)
zrange = (0, 2 * π)

grid = Grid((nx, ny, nz), (xrange, yrange, zrange); array_type=Array)

# A field, containing the simulation informations, is setup. The field
# is decomposed, hence the information of `mpi_topo` (which is not mandatory).
# In this specific simulation, for the Gross-Pitaevskii equation, we need
# a complex field ``ϕ``, and we specify it through the singleton `ComplexField()`.

field = Field(grid, ComplexField(); mpi_topo=mpi_topo)

# ## First step : convergence to a stationary field
#
# In order to solve the Gross-Pitaevskii equation, we setup the parameters.
# ```math
# i \dfrac{dϕ}{dt} = -0.05 Δϕ + 40 |ϕ|²ϕ  + V(x)ϕ
# ```
#
# Here, `V` is a [`PotentialExternalVelocity`](@ref). It contains the following informations:
#
# - $u_{\text{adv}_x} =  \sin(x)\cos(y)\cos(z)$
# - $u_{\text{adv}_y} = -\cos(x)\sin(y)\cos(z)$
# - $u_{\text{adv}_z} = 0$
#
# and $V = \dfrac{\left( u_{\text{adv}_x}^2 + u_{\text{adv}_y}^2 + u_{\text{adv}_z}^2 \right)}{α}

param = GrossPitaevskiiParameters(; coeffΔ=-0.075,
                                  β=27,
                                  pot=PotentialExternalVelocity(field; α=4 * 27 * 0.075))

# initialisation

init = InitExternalVelocity(field, param.coeffΔ, param.β)
initField!(init)

volume((first(grid.x), last(grid.x)), (first(grid.y), last(grid.y)), (first(grid.z), last(grid.z)), parent(abs2.(field.ϕ)); algorithm=:iso, isovalue=0.4,
           isorange=0.2)

# The solver uses [`NumModelExternalVelocity`](@ref).
# It uses the following scheme:
# ```math
# \dfrac{ϕ^{(n+1)}-ϕ^{(n)}}{Δt} = \dfrac{α}{2} Δϕ^{(n+1)} +
# \left(
#    \dfrac{α}{2} Δ - i u⃗_\text{adv} ⋅ ∇ - \dfrac{|u⃗_\text{adv}|²}{4α}
#    + β - β |ϕ^{(n)}|²
# \right) ϕ^{(n)}
# ```

# We setup the solver for the stationary, using:

Δt = 0.005
niter = 500
freqbckp = 100

nummodel = NumModelExternalVelocity(field, param, Δt, niter, freqbckp)

# Solve the problem:

if isfile("../../save/QT_TG_3D-490.h5") && use_saves # hide
    read!(nummodel.writers.writerList[2]; prefix="../../save/QT_TG_3D", istep=490) # hide
    nummodel.niter = 10 # hide
    res = solve!(nummodel; istart=490, plot=false) # hide
    nummodel.niter = 500 # hide
else # hide
    res = solve!(nummodel; plot=false)
end # hide
res[end]

# Plot the solution

volume((first(grid.x), last(grid.x)), (first(grid.y), last(grid.y)), (first(grid.z), last(grid.z)), parent(abs2.(field.ϕ)); algorithm=:iso, isovalue=0.4,
           isorange=0.2)

# Cut at ``z = π``:

surface(grid.x, grid.y, abs2.(parent(field.ϕ)[:, :, nz ÷ 2]))

# ## Second step: unstationary restart

# We start by duplicating the field:

field_insta = Field(grid, ComplexField())
field_insta.ϕ .= field.ϕ;

# The parameters are similar to the previous stationary simulation, but
# the potential is different (``V=0``)

param_insta = GrossPitaevskiiParameters(; coeffΔ=param.coeffΔ,
                                        β=param.β,
                                        pot=PotentialZero(field_insta))

# We instantiate a `NumModelSplit2` which corresponds to
# a second order Strangle scheme.

Δt_insta = Δt
niter_insta = 1000
freqbckp_insta = 10
nummodel_insta = NumModelSplit2(field_insta, param_insta, Δt_insta, niter_insta,
                              freqbckp_insta)

# And we start the solver.

if isfile("../../save/QT_TG_3D-1490.h5") && use_saves # hide
    read!(nummodel_insta.writers.writerList[2]; prefix="../../save/QT_TG_3D", istep=1490) # hide
    nummodel_insta.niter = 10 # hide
    res_insta = solve!(nummodel_insta; istart=1490, plot=false) # hide
    nummodel_insta.niter = 1000 # hide
else # hide
    res_insta = solve!(nummodel_insta; istart=niter, plot=false)
end # hide
res_insta[end]

# Plot the solution.

volume((first(grid.x), last(grid.x)), (first(grid.y), last(grid.y)), (first(grid.z), last(grid.z)), parent(abs2.(field_insta.ϕ)); algorithm=:iso, isovalue=0.4,
           isorange=0.2)

# Cut at ``z = π``:

surface(grid.x, grid.y, abs2.(parent(field_insta.ϕ)[:, :, nz ÷ 2]))
