# # 2D coupled Gross-Pitaevskii / Navier-Stokes (two-fluid)

# We simulate the coupled GP–NS two-fluid model of "Coupling Navier-Stokes and
# Gross-Pitaevskii equations for the numerical simulation of two-fluid quantum
# flows" (Brachet et al., arXiv:2211.07361), implemented by
# [`NumModelNSGP`](@ref).
# The superfluid is described by the Gross-Pitaevskii field ``ψ`` and the normal
# fluid by a Navier-Stokes velocity ``v_n``; the two are coupled through a
# Hall–Vinen–Bekarevich–Khalatnikov (HVBK) mutual-friction force.
# The scheme is a second-order implicit-explicit Runge-Kutta (`"RK2Imp"`), with
# the normal-fluid viscosity treated implicitly.

# We first load the package, in order to have all the constructors and functions available.

using SuperfluidDynamics

# We setup the topology used, if we want to use `MPI` for parallelization.
# In this case, run with `mpirun -np 4 julia --project examples/NSGP_2D/NSGP_2D.jl`

mpi_topo = SuperfluidDynamics.MPITopo1D()

# We use [`Makie`](https://makie.juliaplots.org/stable/) (with `WGLMakie`,
# which produces interactive WebGL plots in the documentation) for the plots:

using WGLMakie # hide
WGLMakie.activate!() # hide
WGLMakie.Bonito.Page(; exportable=true, offline=true) # hide

# ## Discretization

# Both fluids share the same periodic grid, a ``64\times64`` mesh on
# $[-8,8]^2$. The GP field is a complex scalar, the NS field a complex vector.

n = 64
grid = Grid((n, n), ((-8, 8), (-8, 8)))
fgp = Field(grid, ComplexField(); ndims=2)
fns = Field(grid, ComplexField(); ndims=2)

# ## Initialisation

# The superfluid is initialised with a **regularised quantum vortex**
# ``ψ = A(r)\,e^{iθ}`` with core profile $A(r)=r/\sqrt{r^2+a^2}$, which gives a
# finite-core superfluid velocity ``u_s = 2|α|\,r/(r^2+a^2)\,e_\theta``.
# The normal fluid starts from a weak Taylor-Green background.

a, α = 0.8, -0.02
X = vec(grid.x); Y = vec(grid.y)
X2 = reshape(X, :, 1); Y2 = reshape(Y, 1, :)
r2 = X2.^2 .+ Y2.^2
fgp.ϕ .= (sqrt.(r2) ./ sqrt.(r2 .+ a^2)) .* exp.(1im * atan.(Y2, X2))

kx = 2π / grid.Lx; ky = 2π / grid.Ly
@. fns.ux = 0.3 * sin(ky * fns.y) * cos(kx * fns.x)
@. fns.uy = -0.3 * cos(ky * fns.y) * sin(kx * fns.x)

# ## Parameters

# [`NSGPParameters`](@ref) derives the HVBK friction coefficients `B★`, `B'★`
# from the tabulated Hall-Vinen `Btab`,`Bptab`. We use the default one-way
# coupling for the superfluid momentum equation:

param = NSGPParameters(; α=α, ν=0.01, β=1.0, ρn=0.5, ρs=0.5,
                        Btab=0.4, Bptab=0.1, ξ=1.0, ε2=0.05, one_way=true)

# ## Solver

Δt = 0.002
niter = 500
freqbckp = 100

model = NumModelNSGP(fgp, fns, param, Δt, niter, freqbckp; stepper="RK2Imp")

# A small helper to compute the **normal-fluid** vorticity
# ``ω_n = \partial_x v_{n,y} - \partial_y v_{n,x}`` from the physical velocity,
# using a standalone [`Plan`](@ref):

function normal_vorticity(fns) # hide
    pl = SuperfluidDynamics.Plan(fns)
    g = SuperfluidDynamics.spectral_grid(pl)
    uhat = [similar(fns.ux) for _ in 1:2]
    SuperfluidDynamics.mul_all!(uhat, pl, fns.u)
    ωh = @. 1im * (g.x * uhat[2] - g.y * uhat[1])
    ω = similar(fns.ux)
    SuperfluidDynamics.ldiv_all!(ω, pl, ωh)
    return real(ω)
end # hide

# We plot the initial superfluid density $|ψ|^2$ (the vortex core appears as a
# dip in the middle):

heatmap(grid.x, grid.y, abs2.(fgp.ϕ))

# ## Time integration

# Solve the coupled problem with [`solve!`](@ref):

res = solve!(model; plot=false)

# We plot the normal-fluid vorticity after $t = 1$: the counter-rotating
# interaction between the vortex and the background is visible:

heatmap(grid.x, grid.y, normal_vorticity(fns))
