# Weak-scaling benchmark: work per rank held constant, rank count grows.
# In 3D the grid grows as r^(1/3), so each rank keeps ~64^3 grid points.
# Ideal weak scaling => the time per step stays constant as ranks are added;
# the growth factor reports the cost of the pencil transposes at fixed load.
#
#   mpiexec -n 8 julia --project=. -t1 -O3 benchmarks/scalability_weak.jl
#
# Environment:
#   MODEL   ns (default) or gp
#   BASE    per-rank cube edge at 1 rank (default 64)
#   STEPS   time steps measured (default 20)

using MPI
MPI.Init()

using SuperfluidDynamics
using Printf: @printf

const MODEL = get(ENV, "MODEL", "ns")
const BASE = parse(Int, get(ENV, "BASE", "64"))
const NSTEPS = parse(Int, get(ENV, "STEPS", "20"))

comm = MPI.COMM_WORLD
rank = MPI.Comm_rank(comm)
nranks = MPI.Comm_size(comm)

# N grows so that N^3 / nranks stays BASE^3.
N = BASE * max(1, round(Int, nranks^(1 / 3)))

grid = Grid((N, N, N), ((-2π, 2π), (-2π, 2π), (-2π, 2π)))

if MODEL == "ns"
    field = Field(grid, ComplexField(); ndims=3)
    taylor_green!(field, field.x, field.y, field.z)
    model = NumModelRK4Imp(field, NavierStokesParameters(; ν=0.0), 0.01, 1, 1)
    bench = "nsweak"
elseif MODEL == "gp"
    field = Field(grid, ComplexField())
    initField!(InitGauss(field; γz=1.0, Ω=1.0))
    model = NumModelSplit2(field,
                           GrossPitaevskiiParameters(; coeffΔ=-0.075, β=27.0,
                                                     pot=PotentialZero(field)),
                           0.005, 1, 1)
    bench = "gpweak"
else
    error("MODEL must be ns or gp")
end

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
    per_rank = N^3 / nranks
    @printf("nranks=%d  bench=%s  N=%d  steps=%d  s/step=%.5f  Mpts/s=%.2f  pts_per_rank=%d\n",
            nranks, bench, N, NSTEPS, dt_max, N^3 / 1e6 / dt_max, round(Int, per_rank))
end
MPI.Finalize()
