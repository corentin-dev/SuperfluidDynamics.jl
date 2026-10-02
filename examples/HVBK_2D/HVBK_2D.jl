# # 2D HVBK two-fluid (linear)

# We simulate the linear [HVBK two-fluid
# model](https://en.wikipedia.org/wiki/Two-fluid_model) with
# [`NumModelHVBK`](@ref). It describes a classical normal fluid `u_n` and an
# inviscid superfluid `u_s`, each advecting its own vorticity and coupled by the
# **linear** mutual-friction force ``F = -\tfrac12 r_b |w| (u_n - u_s)``
# where ``w = u_n - u_s`` is the relative velocity. The normal-fluid viscosity
# is treated implicitly.

# We first load the package, in order to have all the constructors and functions available.

using SuperfluidDynamics

# We setup the topology used, if we want to use `MPI` for parallelization.
# In this case, run with `mpirun -np 4 julia --project examples/HVBK_2D/HVBK_2D.jl`

mpi_topo = SuperfluidDynamics.MPITopo1D()

# We use [`Makie`](https://makie.juliaplots.org/stable/) (with `WGLMakie`,
# which produces interactive WebGL plots in the documentation) for the plots:

using WGLMakie # hide
WGLMakie.activate!() # hide
WGLMakie.Bonito.Page(; exportable=true, offline=true) # hide

# ## Discretization

# The two fluids share a periodic ``48\times36`` grid on $[-4,4]\times[-5,5]$.
# Each is a complex vector field (``u_n`` and ``u_s``):

grid = Grid((48, 36), ((-4, 4), (-5, 5)))
fn = Field(grid, ComplexField(); ndims=2)   # normal fluid
fs = Field(grid, ComplexField(); ndims=2)   # superfluid

# ## Initialisation

# Both fluids start from divergence-free Taylor-Green-like background modes with
# slightly different wavelengths, so the relative motion (and hence the mutual
# friction) is non-trivial:

kx = 2π / grid.Lx; ky = 2π / grid.Ly
@. fn.ux = sin(ky * fn.y) * cos(kx * fn.x)
@. fn.uy = -cos(ky * fn.y) * sin(kx * fn.x)
@. fs.ux = 0.7 * cos(ky * fs.y) * sin(kx * fs.x)
@. fs.uy = 0.7 * sin(ky * fs.y) * cos(kx * fs.x)

# ## Parameters

# [`HVBKParameters`](@ref): `ν` the normal viscosity, `νs` the (negligible)
# superfluid viscosity, `rb` the mutual-friction coefficient, `ρn`/`ρs` the two masses.

param = HVBKParameters(; ν=0.01, νs=0.001, rb=1.5, ρn=1.0, ρs=1.0)

# ## Solver

# We integrate with the second-order scheme (`stepper="RK2"`):

Δt = 0.01
niter = 400
freqbckp = 100

model = NumModelHVBK(fn, fs, param, Δt, niter, freqbckp; stepper="RK2")

# The scalar (2D) vorticity is computed spectrally the same way as inside the
# model — `ω = i(k_x u_y − k_y u_x)` — reusing the model's pre-allocated
# spectral scratch buffers (reading them fresh, not the internal buffers that
# are only filled during `solve!`):

function vorticity(model, f, uhat, tmp, out) # hide
    g = SuperfluidDynamics.spectral_grid(model.plan)
    @. tmp[2] = 1im * (g.x * uhat[2] - g.y * uhat[1])
    SuperfluidDynamics.ldiv_all!(out, model.plan, tmp)
    return real(parent(out[2]))
end # hide

# Initial normal-fluid vorticity:

heatmap(grid.x, grid.y, vorticity(model, model.fn, model.un_hat, model.tmp_hat, model.un_vort))

# Initial superfluid vorticity:

heatmap(grid.x, grid.y, vorticity(model, model.fs, model.us_hat, model.tmp_hat, model.us_vort))

# ## Time integration

# Solve with [`solve!`](@ref). The mutual friction relaxes the relative velocity
# toward zero while the normal viscosity dissipates the normal vorticity:

res = solve!(model; plot=false)

# After $t = 4$ the two vorticities have evolved; we plot the final state:

# Final normal-fluid vorticity:

heatmap(grid.x, grid.y, vorticity(model, model.fn, model.un_hat, model.tmp_hat, model.un_vort))

# Final superfluid vorticity:

heatmap(grid.x, grid.y, vorticity(model, model.fs, model.us_hat, model.tmp_hat, model.us_vort))
