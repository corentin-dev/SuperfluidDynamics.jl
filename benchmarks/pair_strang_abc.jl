# Paired cost measurement: real-time GP evolution on the ABC box, OUR solver vs
# the GPS (Fortran) code. This is the counterpart of the GPS input written for
# `examples/3D/quad`-style runs: model 42 (TS2_ADI = Strang splitting) on
#   box (0,2pi)^3, n=128^3, beta=40, coeff_delta=-0.05 (alpha=0.05),
#   potential npot=9 (ABC u_adv + |u_adv|^2/(4|coeff_delta|)), IC initcondition=9
#   (ABC, eq. (82)-(83)), deltat=1e-3, 100 steps.
#
# Here: NumModelSplit2 (Lie? no — Strang), same box/grid/parameters, dt=1e-3,
# same number of steps, potential V = |u_adv|^2/(4 alpha) so the WORK PER STEP
# matches (the linear and nonlinear operators are the same ones).
#
# Both sides time the step loop only. Print is one line, machine readable.
#
#   mpiexec -n 8 julia --project=. -t1 -O3 benchmarks/pair_strang_abc.jl
#
# Environment:
#   PAIR_N      grid size per direction (default 128)
#   PAIR_STEPS  steps measured (default 100)

using MPI
MPI.Init()

using SuperfluidDynamics
using Printf: @printf

const N = parse(Int, get(ENV, "PAIR_N", "128"))
const NSTEPS = parse(Int, get(ENV, "PAIR_STEPS", "100"))

comm = MPI.COMM_WORLD
rank = MPI.Comm_rank(comm)
nranks = MPI.Comm_size(comm)

const α, β = 0.05, 40.0

grid = Grid((N, N, N), ((0.0, 2π), (0.0, 2π), (0.0, 2π)))
field = Field(grid, ComplexField())

# Same ABC velocity as the GPS (npot==9) and as the supplier's harness.
const A, B, C = 0.9 / sqrt(3), 1.0 / sqrt(3), 1.1 / sqrt(3)
vx(x, y, z) = sum(B * cos(k * y) + C * sin(k * z) for k in (1, 2))
vy(x, y, z) = sum(C * cos(k * z) + A * sin(k * x) for k in (1, 2))
vz(x, y, z) = sum(A * cos(k * x) + B * sin(k * y) for k in (1, 2))

pot = PotentialExternalVelocity(field; uadv_function=(vx, vy, vz), α=4α * β)
param = GrossPitaevskiiParameters(; coeffΔ=-α, β=β, pot=pot)

X, Y, Z = field.x, field.y, field.z
ϕ = field.ϕ
ϕ .= one(ComplexF64)
for k in (1, 2)
    ϕ .*= cis.(round.(A * sin.(k .* X) ./ (2α)) .* Y .+ round.(A * cos.(k .* X) ./ (2α)) .* Z)
    ϕ .*= cis.(round.(B * sin.(k .* Y) ./ (2α)) .* Z .+ round.(B * cos.(k .* Y) ./ (2α)) .* X)
    ϕ .*= cis.(round.(C * sin.(k .* Z) ./ (2α)) .* X .+ round.(C * cos.(k .* Z) ./ (2α)) .* Y)
end
normalize!(field)

dt = 1e-3
n = NumModelSplit2(field, param, dt, NSTEPS, NSTEPS + 1)
SuperfluidDynamics.timeStep!(n)              # FFT plans
MPI.Barrier(comm)

t0 = time_ns()
for _ in 1:NSTEPS
    SuperfluidDynamics.timeStep!(n)
end
MPI.Barrier(comm)
tot = (time_ns() - t0) / 1e9
tot_max = MPI.Allreduce(tot, max, comm)
s_per_step = tot_max / NSTEPS

if rank == 0
    pts = N^3
    @printf("nranks=%d  bench=pair_strang_abc  N=%d  steps=%d  total_s=%.4f  s/step=%.6f  Mpts/s=%.2f  core_s_per_point_iter=%.4e\n",
            nranks, N, NSTEPS, tot_max, s_per_step, pts / 1e6 / s_per_step,
            s_per_step * nranks / pts)
end
MPI.Finalize()
