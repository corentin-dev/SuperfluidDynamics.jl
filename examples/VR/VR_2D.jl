# # 2D Vortex Pair

using SuperfluidDynamics

# ## Discretization

# We setup the topology used, if we want to use `MPI` for parallelization.
# In this case, run with `mpirun -np 4 julia --project example/QT_TG_3D/QT_TG_3D.jl`
# if ``4`` is the desired number of processes.

# Here, the topoology is a slab topology, meaning that ``y`` direction can be
# decomposed. The slab is transposed for some operations (FFT, finite differences, etc).

mpi_topo = SuperfluidDynamics.MPITopo1D()

# We use [`Makie`](https://makie.juliaplots.org/stable/) (with `WGLMakie`,
# which produces interactive WebGL plots in the documentation) for the plots:

using WGLMakie # hide
WGLMakie.activate!() # hide
WGLMakie.Bonito.Page(; exportable=true, offline=true) # hide

# We setup the wanted discretization. We want $n_x \times n_y = 128\times 128$.
# The domain bounds are set to $[0,2π]\times[0,2π]$.
# With those informations, we can create [`Grid`](@ref), using the constructor:

nx = 128
ny = 128
xrange = (0, 2π)
yrange = (0, 2π)

grid = Grid((nx, ny), (xrange, yrange))
field = Field(grid, ComplexField(); mpi_topo=mpi_topo)

# ## Physical parameters

# We want to specify the healing length $ξ$ from the grid size.
# Here $ξ=15Δx$. We determine the coefficient $α$ in front of $Δϕ$
# and $β$ from this information. We force the distance $d_\text{vor}$
# between the two vortices to be equal to 20 times the healing length $ξ$.

ξ = 15.0 / minimum((nx, ny))
coeffΔ = -2ξ / √2
β = -coeffΔ / ξ^2
d = 20.0
v₊ = (grid.Lx / 4 + grid.xmin, grid.Ly / 2 + d * ξ / 2 + grid.ymin, +1)
v₋ = (grid.Lx / 4 + grid.xmin, grid.Ly / 2 - d * ξ / 2 + grid.ymin, -1)

# We want to have a constant advection velocity:
# $$\vec{u} = \begin{pmatrix} -α\left( \dfrac{1 + \cos(d_{\text{vor}})}{\sin(d_{\text{vor}})} + \dfrac{d_{\text{vor}}}{π} \right) \\ 0 \end{pmatrix}$$
# In order to obtain this velocity, we use a [`PotentialExternalVelocity`](@ref)
# with a custom function called `uadv_function_VR_2D`:
uadv_function_VR_2D = ((x, y) -> -coeffΔ * ((1 + cos(d * ξ)) / sin(d * ξ) + d * ξ / π),
                       (x, y) -> 0.0);

pot = PotentialExternalVelocity(field; uadv_function=uadv_function_VR_2D, α=-4 * coeffΔ)

# We then create [`GrossPitaevskiiParameters`](@ref) using this potential:
param = GrossPitaevskiiParameters(; coeffΔ=coeffΔ,
                                  β=β,
                                  pot=pot)

# ## Initialization

# We want to initialize properly a vortex pair.
# First we create a function that gives the module with respect to the radius distance
# from the vortex core.
# $$f(r) = \sqrt{ \dfrac{ a_1 r^2 + a_2 r^4 }{1 + b_1 r^2 + b_2 r^4} }$$
# We call it `ρ_vortex`:

function ρ_vortex(r)
    a₁, a₂, b₁, b₂ = 11.0 / 32.0, 11.0 / 384.0, 1.0 / 3.0, 11.0 / 384.0
    return √((a₁ * r^2 + a₂ * r^4) / (1 + b₁ * r^2 + b₂ * r^4))
end

# The following function allows to create a vortex pair:

function init_VR_tanh(x, y, ξ, v₊, v₋, grid)
    x₋, y₋ = (x - v₋[1]), (y - v₋[2])
    x₊, y₊ = (x - v₊[1]), (y - v₊[2])

    ρ = ρ_vortex(√(x₋^2 + y₋^2) / ξ) *
        ρ_vortex(√(x₊^2 + y₊^2) / ξ)

    θ = (v₋[1] - v₊[1]) * y / 2π
    for k in -5:5
        θ += (atan(tanh(0.5 * (y₋ + 2π * k)) * tan(0.5(x₋ - π)))
              -
              atan(tanh(0.5 * (y₊ + 2π * k)) * tan(0.5(x₊ - π)))
              +
              π * (((x₊ > 0) ? 1.0 : 0.0) - ((x₋ > 0) ? 1.0 : 0.0)))
    end
    return ρ * exp(im * θ)
end

f0(x, y) = init_VR_tanh(x, y, ξ, v₊, v₋, grid)
@. field.ϕ = f0(field.x, field.y)

# The phase, with the two vortices (density dips) at $x = \pm 12.5$:

contourf(grid.x, grid.y, atan.(imag.(field.ϕ), real.(field.ϕ)))

# and the density $|ϕ|²$:

contourf(grid.x, grid.y, abs2.(field.ϕ))

# To clean the solution a bit, we use the ARGLE scheme
# using [`NumModelExternalVelocity`](@ref)
# and $Δt=2.5×10⁻³$, $n_\text{iter}=2000$:

Δt = 0.0025
niter = 2000
freqbckp = 200
nummodel = NumModelExternalVelocity(field, param, Δt, niter, freqbckp)
res = solve!(nummodel; plot=false);
contourf(grid.x, grid.y, abs2.(field.ϕ))

# ## Unsteady computation

# We create a new field to restart with a real time algorithm:
field_insta = Field(grid, ComplexField())
field_insta.ϕ .= field.ϕ;

# The parameters are similar to the previous stationary simulation, but
# the potential is different (``V=0``)

param_insta = GrossPitaevskiiParameters(; coeffΔ=param.coeffΔ,
                                        β=param.β,
                                        pot=PotentialZero(field_insta))

# We instantiate a `NumModelSplit2` which corresponds to
# a second order Strang splitting scheme.

Δt_insta = Δt / 2.5
niter_insta = 20000
freqbckp_insta = 400
nummodel_insta = NumModelSplit2(field_insta, param_insta, Δt_insta, niter_insta,
                              freqbckp_insta)

# And we start the solver.

res_insta = solve!(nummodel_insta; istart=niter, plot=false);
contourf(grid.x, grid.y, abs2.(field_insta.ϕ))
