# Does the BdG eigensolve work under MPI? Unlike the NS/GP/HVBK solvers, BdG
# is driven by Arpack.eigs on a DENSE, rank-local vector (2·N_local, built from
# parent(ϕ)), while the operator it applies (_bdg_apply!) is collective (FFT
# pencils, sums reduced across ranks). Convergence is therefore judged on local
# data while the iteration performs global communication: a rank can stop on
# its local residual while another keeps calling the operator — a classic
# deadlock, not a crash. This script is the fact behind the paper's one-line
# statement, run per rank count under `timeout` so a deadlock shows up as 124.
#
# Setup: the validated 2D anisotropic harmonic oscillator of the test suite
# (ωx=1, ωy=√2, β=0), analytic modes {ωx, ωy, 2ωx}.
#
#   mpiexec -n 2 julia --project=. -t1 -O3 benchmarks/scalability_bdg.jl
#
# Environment: BDG_N (default 20), BDG_NEV (default 3), BDG_STEPS (default 1).

using MPI
MPI.Init()

using SuperfluidDynamics
using Printf: @printf

const N = parse(Int, get(ENV, "BDG_N", "20"))
const NEV = parse(Int, get(ENV, "BDG_NEV", "3"))
const NSTEPS = parse(Int, get(ENV, "BDG_STEPS", "1"))

comm = MPI.COMM_WORLD
rank = MPI.Comm_rank(comm)
nranks = MPI.Comm_size(comm)

# 20x20 on [-6,6]^2 keeps the analytic-spectrum regime of the unit test;
# larger grids make the local/global mismatch likelier to bite.
grid = Grid((N, N), ((-6.0, 6.0), (-6.0, 6.0)))
field = Field(grid, ComplexField())
ωx, ωy = 1.0, sqrt(2.0)
X2 = reshape(vec(field.x), :, 1)
Y2 = reshape(vec(field.y), 1, :)
field.ϕ .= exp.(-ωx * X2.^2 / 2 .- ωy * Y2.^2 / 2)
field.ϕ ./= sqrt(sum(abs2.(field.ϕ)) * grid.Δx * grid.Δy)   # global norm

param = BdGParameters(coeffΔ=-0.5, β=0.0, pot=PotentialQuadratic(field; γx=ωx, γy=ωy^2), Ω=0.0)
model = NumModelBdG(field, param, 1, 1; nev=NEV)

SuperfluidDynamics.timeStep!(model)      # collective operator, local Arpack
MPI.Barrier(comm)

# Rank 0 checks against the analytic spectrum; other ranks report their own
# values, so a rank-dependent answer proves the eigensolve was rank-local.
expected = [ωx, ωy, 2ωx]
got = length(model.ωs) >= length(expected) ? model.ωs[1:length(expected)] : model.ωs
err = isempty(got) ? Inf : maximum(abs.(got .- expected))
ωs_str = join(string.(round.(model.ωs; digits=4)), ",")

@printf("rank %d/%d bdg: nranks=%d N=%d mu=%.6f modes=%s max_err_vs_analytic=%.2e\n",
        rank, nranks, nranks, N, model.mu, ωs_str, err)
flush(stdout)

if NSTEPS > 1
    t0 = time_ns()
    for _ in 1:NSTEPS
        SuperfluidDynamics.timeStep!(model)
    end
    MPI.Barrier(comm)
    dt = (time_ns() - t0) / 1e9 / NSTEPS
    if rank == 0
        @printf("nranks=%d  bench=bdg  N=%d  steps=%d  s/step=%.3f\n",
                nranks, N, NSTEPS, dt)
    end
end
flush(stdout)
MPI.Finalize()
