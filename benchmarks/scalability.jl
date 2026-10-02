# Strong-scaling benchmark: incompressible Navier-Stokes (NumModelRK4Imp),
# 3D Taylor-Green vortex, MPI pencil decomposition (PencilArrays).
#
# Run on one node:
#   mpiexec -n 8 julia --project=. -t1 -O3 benchmarks/scalability.jl
#
# Environment:
#   NS_N      grid size per direction (default 96)
#   NS_STEPS  number of time steps measured (default 20)

using MPI
MPI.Init()

using SuperfluidDynamics
using Printf: @printf

const N = parse(Int, get(ENV, "NS_N", "96"))
const NSTEPS = parse(Int, get(ENV, "NS_STEPS", "20"))

comm = MPI.COMM_WORLD
rank = MPI.Comm_rank(comm)
nranks = MPI.Comm_size(comm)

# A 3D field takes the 2D (pencil) decomposition; MPITopo1D is for 2D grids.
grid = Grid((N, N, N), ((-2π, 2π), (-2π, 2π), (-2π, 2π)))
field = Field(grid, ComplexField(); ndims=3)
n = NumModelRK4Imp(field, NavierStokesParameters(; ν=0.0), 0.01, 1, 1)
taylor_green!(field, field.x, field.y, field.z)

# one step to trigger the FFT plans, then measure
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
    local_pts = length(parent(field.ux))
    @printf("nranks=%d  N=%d  steps=%d  s/step=%.5f  Mpts/s=%.2f  local_pts/rank=%d\n",
            nranks, N, NSTEPS, dt_max, pts / 1e6 / dt_max, local_pts)
end
MPI.Finalize()
