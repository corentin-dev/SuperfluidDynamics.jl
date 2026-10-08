# Strong-scaling benchmark: ARGLE relaxation (NumModelExternalVelocity), 3D,
# MPI pencil decomposition — the state-preparation stage of the Kobayashi ABC
# benchmark (the real-time stage is timed by scalability_gp.jl).
#
# Configuration: ABC velocity (0.9, 1.0, 1.1)/sqrt(3), alpha=0.05, beta=40,
# box (0,2pi)^3, dtau=0.004. Reports a pseudo-step rate plus
# core_s_per_point_iter for comparison against published rates.
#
# Run on one node:
#   mpiexec -n 8 julia --project=. -t1 -O3 benchmarks/scalability_argle.jl
#
# Environment:
#   ARGLE_N      grid size per direction (default 128)
#   ARGLE_STEPS  pseudo-steps measured (default 20)

using MPI
MPI.Init()

using SuperfluidDynamics
using Printf: @printf

const N = parse(Int, get(ENV, "ARGLE_N", "128"))
const NSTEPS = parse(Int, get(ENV, "ARGLE_STEPS", "20"))

comm = MPI.COMM_WORLD
rank = MPI.Comm_rank(comm)
nranks = MPI.Comm_size(comm)

const α, β = 0.05, 40.0
const A, B, C = 0.9 / sqrt(3), 1.0 / sqrt(3), 1.1 / sqrt(3)

grid = Grid((N, N, N), ((0.0, 2π), (0.0, 2π), (0.0, 2π)))
field = Field(grid, ComplexField())

# ABC velocity, exactly as the supplier's harness abc_velocity (run_case.jl).
vx(x, y, z) = sum(B * cos(k * y) + C * sin(k * z) for k in (1, 2))
vy(x, y, z) = sum(C * cos(k * z) + A * sin(k * x) for k in (1, 2))
vz(x, y, z) = sum(A * cos(k * x) + B * sin(k * y) for k in (1, 2))

pot = PotentialExternalVelocity(field; uadv_function=(vx, vy, vz), α=4α * β)
param = GrossPitaevskiiParameters(; coeffΔ=-α, β=β, pot=pot)

# Initial state: eq. (82)-(83) of the paper, same expression as the harness
# abc_psi0 but written on the LOCAL coordinates (field.x/y/z are the local
# views, so this also works with the distributed pencil layout; the harness is
# single-rank only — need_single_rank("qt") — and indexes full axes).
X, Y, Z = field.x, field.y, field.z
ϕ = field.ϕ
ϕ .= one(ComplexF64)
for k in (1, 2)
    ϕ .*= cis.(round.(A * sin.(k .* X) ./ (2α)) .* Y .+ round.(A * cos.(k .* X) ./ (2α)) .* Z)
    ϕ .*= cis.(round.(B * sin.(k .* Y) ./ (2α)) .* Z .+ round.(B * cos.(k .* Y) ./ (2α)) .* X)
    ϕ .*= cis.(round.(C * sin.(k .* Z) ./ (2α)) .* X .+ round.(C * cos.(k .* Z) ./ (2α)) .* Y)
end
normalize!(field)

n = NumModelExternalVelocity(field, param, 0.004, 1, 1)
SuperfluidDynamics.timeStep!(n)          # trigger the FFT plans
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
    core_s = dt_max * nranks / pts
    @printf("nranks=%d  bench=argle  N=%d  steps=%d  s/step=%.5f  Mpts/s=%.2f  core_s_per_point_iter=%.4e\n",
            nranks, N, NSTEPS, dt_max, pts / 1e6 / dt_max, core_s)
end
MPI.Finalize()
