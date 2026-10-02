# Strong-scaling benchmark: HVBK two-fluid model (NumModelHVBK), 3D, MPI pencil
# decomposition (PencilArrays).
#
# Run on one node:
#   mpiexec -n 8 julia --project=. -t1 -O3 benchmarks/scalability_hvbk.jl
#
# Environment:
#   HB_N      grid size per direction (default 64)
#   HB_STEPS  number of time steps measured (default 20)

using MPI
MPI.Init()

using SuperfluidDynamics
using Printf: @printf

const N = parse(Int, get(ENV, "HB_N", "64"))
const NSTEPS = parse(Int, get(ENV, "HB_STEPS", "20"))

comm = MPI.COMM_WORLD
rank = MPI.Comm_rank(comm)
nranks = MPI.Comm_size(comm)

grid = Grid((N, N, N), ((-2π, 2π), (-2π, 2π), (-2π, 2π)))
fn = Field(grid, ComplexField(); ndims=3)   # normal fluid
fs = Field(grid, ComplexField(); ndims=3)   # superfluid

# taylor_green! is divergence-free by construction; the superfluid takes a
# scaled counter-flow, so both fields survive the spectral projection and the
# relative motion keeps mutual friction doing work at every step.
taylor_green!(fn, fn.x, fn.y, fn.z)
@. fs.ux = -0.7 * fn.ux
@. fs.uy = -0.7 * fn.uy
@. fs.uz = 0.0

param = HVBKParameters(; ν=0.01, νs=0.001, rb=1.5, ρn=1.0, ρs=1.0)
model = NumModelHVBK(fn, fs, param, 0.01, 1, 1; stepper="RK2")

# one step to trigger the plans, then measure
SuperfluidDynamics.timeStep!(model)
MPI.Barrier(comm)

t0 = time_ns()
for _ in 1:NSTEPS
    SuperfluidDynamics.timeStep!(model)
end
MPI.Barrier(comm)
dt = (time_ns() - t0) / 1e9 / NSTEPS
dt_max = MPI.Allreduce(dt, max, comm)

if rank == 0
    pts = N^3
    @printf("nranks=%d  bench=hvbk  N=%d  steps=%d  s/step=%.5f  Mpts/s=%.2f\n",
            nranks, N, NSTEPS, dt_max, pts / 1e6 / dt_max)
end
MPI.Finalize()
