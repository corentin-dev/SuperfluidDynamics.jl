# Strong-scaling benchmark: NSGP two-fluid model (NumModelNSGP), 2D quantum
# vortex + Taylor-Green initial state, MPI decomposition (PencilArrays).
#
# Run on one node:
#   mpiexec -n 8 julia --project=. -t1 -O3 benchmarks/scalability_nsgp.jl
#
# Environment:
#   NSGP_N      grid size per direction (default 128)
#   NSGP_STEPS  number of time steps measured (default 20)

using MPI
MPI.Init()

using SuperfluidDynamics
using Printf: @printf

const N = parse(Int, get(ENV, "NSGP_N", "128"))
const NSTEPS = parse(Int, get(ENV, "NSGP_STEPS", "20"))

comm = MPI.COMM_WORLD
rank = MPI.Comm_rank(comm)
nranks = MPI.Comm_size(comm)

# Same setup as the NSGP_2D example: regularised vortex in the superfluid,
# weak Taylor-Green background in the normal fluid.
L = 8.0
grid = Grid((N, N), ((-L, L), (-L, L)))
fgp = Field(grid, ComplexField(); ndims=2)
fns = Field(grid, ComplexField(); ndims=2)

a, α = 0.8, -0.02
X2 = reshape(vec(fgp.x), :, 1)
Y2 = reshape(vec(fgp.y), 1, :)
r2 = X2.^2 .+ Y2.^2
fgp.ϕ .= (sqrt.(r2) ./ sqrt.(r2 .+ a^2)) .* exp.(1im * atan.(Y2, X2))

kx = 2π / grid.Lx; ky = 2π / grid.Ly
@. fns.ux = 0.3 * sin(ky * fns.y) * cos(kx * fns.x)
@. fns.uy = -0.3 * cos(ky * fns.y) * sin(kx * fns.x)

param = NSGPParameters(; α=α, ν=0.01, β=1.0, ρn=0.5, ρs=0.5,
                       Btab=0.4, Bptab=0.1, ξ=1.0, ε2=0.05, one_way=true)
model = NumModelNSGP(fgp, fns, param, 0.002, 1, 1; stepper="RK2Imp")

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
    pts = N^2
    @printf("nranks=%d  bench=nsgp  N=%d  steps=%d  s/step=%.6f  Mpts/s=%.2f\n",
            nranks, N, NSTEPS, dt_max, pts / 1e6 / dt_max)
end
MPI.Finalize()
