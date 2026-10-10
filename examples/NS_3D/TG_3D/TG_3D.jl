using SuperfluidDynamics

# =====================================================================
# 3D Navier-Stokes — Taylor-Green vortex, semi-implicit RK4
#
# du/dt = ν Δu - P(u × ∇×u),  ∇·u = 0
# viscosity handled implicitly as exp(-ν Δt |k|²), advection explicit (RK4)
#
# The lowest energy mode (1,1,1) decays at rate 2ν|k|² = 2ν (2π/L)²·3 with
# L the box length; the run-time decay of E below is consistent with that.
# =====================================================================

# simulation parameters
nx = 64
ny = 48
nz = 56

xrange = (-4, 4)
yrange = (-5, 5)
zrange = (-6, 6)

# equation
param = NavierStokesParameters(; ν=0.01)

# solver
Δt = 0.02
niter = 200
freqbckp = 25

# creating a grid
grid = Grid((nx, ny, nz), (xrange, yrange, zrange))
println(grid)
# allocating a vector field (u = (ux, uy, uz))
field = Field(grid, ComplexField(); ndims=3)
# solver
nummodel = NumModelRK4Imp(field, param, Δt, niter, freqbckp)
println(nummodel)
# initialisation: 3D Taylor-Green vortex
taylor_green!(field, grid.x, grid.y, grid.z)
# solving
res = solve!(nummodel; plot=false)

# diagnostics: final energy and divergence
E0 = res[1][4]
E = res[end][4]
d = SuperfluidDynamics.divergence(nummodel)
println("E0 = $(E0), E = $(E), decay = $(E / E0)")
println("max |div u| = $(maximum(abs.(parent(d))))")
