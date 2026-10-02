# Strong-scaling benchmark: Gross-Pitaevskii, NumModelSplit2 (second order
# Strang splitting), 3D, MPI pencil decomposition (PencilArrays).
#
# Run on one node:
#   mpiexec -n 8 julia --project=. -t1 -O3 benchmarks/scalability_gp.jl
#
# Environment:
#   GP_N      grid size per direction (default 96)
#   GP_STEPS  number of time steps measured (default 20)

using MPI
MPI.Init()

using SuperfluidDynamics
using Printf: @printf

const N = parse(Int, get(ENV, "GP_N", "96"))
const NSTEPS = parse(Int, get(ENV, "GP_STEPS", "20"))

comm = MPI.COMM_WORLD
rank = MPI.Comm_rank(comm)
nranks = MPI.Comm_size(comm)

# A 3D field takes the 2D (pencil) decomposition; MPITopo1D is for 2D grids.
grid = Grid((N, N, N), ((-π, π), (-π, π), (-π, π)))
field = Field(grid, ComplexField())

# A smooth non-stationary state: a modulated Gaussian, so every step is real work.
init = InitGauss(field; γz=1.0, Ω=1.0)
initField!(init)

param = GrossPitaevskiiParameters(; coeffΔ=-0.075, β=27.0,
                                  pot=PotentialZero(field))
n = NumModelSplit2(field, param, 0.005, 1, 1)

# one step to trigger the plans, then measure
SuperfluidDynamics.timeStep!(n)
MPI.Barrier(comm)

t0 = time_ns()
for _ in 1:NSTEPS
    SuperfluidDynamics.timeStep!(n)
end
MPI.Barrier(comm)
dt = (time_ns() - t0) / 1e9 / NSTEPS
dt_max = MPI.Allreduce(dt, max, comm)

if rank == 0
    pts = N^3
    @printf("nranks=%d  bench=gp  N=%d  steps=%d  s/step=%.5f  Mpts/s=%.2f\n",
            nranks, N, NSTEPS, dt_max, pts / 1e6 / dt_max)
end
MPI.Finalize()
