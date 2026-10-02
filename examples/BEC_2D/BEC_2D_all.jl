using SuperfluidDynamics

mpi_topo = SuperfluidDynamics.MPITopo1D();

# simulation parameters
nx = 128
ny = 128

xrange = (-12, 12)
yrange = (-12, 12)

# creating a grid
grid = Grid((nx, ny), (xrange, yrange))
println_parallel(grid)
# allocating a field
field = Field(grid, ComplexField(); mpi_topo=mpi_topo)
println_parallel(field)

# potential
α = 0
γx = 1
γy = 1

# equation
param = GrossPitaevskiiParameters(; β=1000,
                                  Ω=0.9,
                                  pot=PotentialQuadratic(field; γx=γx, γy=γy))

# solver
Δt = 0.01
niter = 25
freqbckp = 10
istart = 0

# initialisation
init = InitThomasFermi(field, param.β; γx=γx, γy=γy)
# init = InitGauss(field, Ω = param.Ω)
initField!(init)

# BackwardEuler (with precond)
nummodel = NumModelBackwardEuler(field, param, Δt, niter, freqbckp)
println_parallel(nummodel)
solve!(nummodel; plot=false, istart=istart)
istart += niter

# BackwardEulerNoPrecond
nummodel = NumModelBackwardEulerNoPrecond(field, param, Δt, niter, freqbckp)
println_parallel(nummodel)
solve!(nummodel; plot=false, istart=istart)
istart += niter

# CrankNicolson
nummodel = NumModelCrankNicolson(field, param, Δt, niter, freqbckp)
println_parallel(nummodel)
solve!(nummodel; plot=false, istart=istart)
istart += niter

# CrankNicolsonQuasiNewton
nummodel = NumModelCrankNicolsonQuasiNewton(field, param, Δt, niter, freqbckp)
println_parallel(nummodel)
solve!(nummodel; plot=false, istart=istart)
istart += niter

# Split1
nummodel = NumModelSplit1(field, param, Δt, niter, freqbckp)
println_parallel(nummodel)
solve!(nummodel; plot=false, istart=istart)
istart += niter

# Split2
nummodel = NumModelSplit2(field, param, Δt, niter, freqbckp)
println_parallel(nummodel)
solve!(nummodel; plot=false, istart=istart)
istart += niter

# CrankNicolsonT
nummodel = NumModelCrankNicolsonT(field, param, Δt * 0.1, niter, freqbckp)
println_parallel(nummodel)
solve!(nummodel; plot=false, istart=istart)
istart += niter

# CrankNicolsonQuasiNewtonT
nummodel = NumModelCrankNicolsonQuasiNewtonT(field, param, Δt * 0.1, niter, freqbckp)
println_parallel(nummodel)
solve!(nummodel; plot=false, istart=istart)
istart += niter

nothing
