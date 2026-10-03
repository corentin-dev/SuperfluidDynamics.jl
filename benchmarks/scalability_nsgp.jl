# Strong-scaling benchmark: NSGP two-fluid model (NumModelNSGP), quantum vortex
# (a line along z in 3D) + Taylor-Green initial state, MPI decomposition
# (PencilArrays).
#
# Run on one node:
#   mpiexec -n 8 julia --project=. -t1 -O3 benchmarks/scalability_nsgp.jl
#
# Environment:
#   NSGP_N      grid size per direction (default 128; use e.g. 64 in 3D)
#   NSGP_DIM    2 or 3 (default 2)
#   NSGP_STEPS  number of time steps measured (default 20)

using MPI
MPI.Init()

using SuperfluidDynamics
using Printf: @printf

const N = parse(Int, get(ENV, "NSGP_N", "128"))
const NSTEPS = parse(Int, get(ENV, "NSGP_STEPS", "20"))
const DIM = parse(Int, get(ENV, "NSGP_DIM", "2"))
DIM in (2, 3) || error("NSGP_DIM must be 2 or 3")

comm = MPI.COMM_WORLD
rank = MPI.Comm_rank(comm)
nranks = MPI.Comm_size(comm)

# Same setup as the NSGP_2D example: regularised vortex in the superfluid,
# weak Taylor-Green background in the normal fluid (in 3D: a vortex line along z,
# and a z-modulated Taylor-Green flow).
L = 8.0
grid = DIM == 2 ? Grid((N, N), ((-L, L), (-L, L))) :
                  Grid((N, N, N), ((-L, L), (-L, L), (-L, L)))
fgp = Field(grid, ComplexField(); ndims=DIM)
fns = Field(grid, ComplexField(); ndims=DIM)

a, α = 0.8, -0.02
X = reshape(vec(fgp.x), :, 1, (DIM == 3 ? (1,) : ())...)
Y = reshape(vec(fgp.y), 1, :, (DIM == 3 ? (1,) : ())...)
r2 = X.^2 .+ Y.^2
fgp.ϕ .= (sqrt.(r2) ./ sqrt.(r2 .+ a^2)) .* exp.(1im * atan.(Y, X))

kx = 2π / grid.Lx; ky = 2π / grid.Ly
if DIM == 2
    @. fns.ux = 0.3 * sin(ky * fns.y) * cos(kx * fns.x)
    @. fns.uy = -0.3 * cos(ky * fns.y) * sin(kx * fns.x)
else
    kz = 2π / grid.Lz
    @. fns.ux = 0.3 * sin(ky * fns.y) * cos(kx * fns.x) * cos(kz * fns.z)
    @. fns.uy = -0.3 * cos(ky * fns.y) * sin(kx * fns.x) * cos(kz * fns.z)
    fns.uz .= 0
end

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
    pts = N^DIM
    @printf("nranks=%d  bench=nsgp  dim=%d  N=%d  steps=%d  s/step=%.6f  Mpts/s=%.2f\n",
            nranks, DIM, N, NSTEPS, dt_max, pts / 1e6 / dt_max)
end
MPI.Finalize()
