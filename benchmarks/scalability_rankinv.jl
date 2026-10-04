# Rank-invariance probe: does the number of MPI ranks change the physics?
#
# Why: the 2/3-rule dealiasing bound ξmax = (4/9)·min_d max_d'|ξ_d'|² is
# computed over the LOCAL pencil grid. Dealiasing happens on the last pencil,
# where two dimensions are distributed — so a rank may not hold the largest
# wavenumber of every direction, its bound can come out lower than the global
# one, and it then zeroes modes its peers keep. This bench does not argue, it
# measures.
#
# Protocol: a broadband initial state (six plane-wave bands at N/3…2N/3 — right
# where the 2/3-rule boundary sits — with amplitudes/phases drawn from a fixed
# seed, so every rank count starts from the identical field), a short NS or GP
# evolution, and the globally reduced energy as observable. Run at 1, 2, 4, 8
# ranks: the rank-count-to-rank-count drift of E(t) is the measurement.
# DEALIAS=0 removes the suspect mechanism and must be drift-free to round-off:
# it separates "the bound is rank-dependent" from "not bit-reproducible under
# MPI" (a legitimate, harmless cause of drift).
#
#   RANKINV_MODEL=ns RANKINV_N=96 RANKINV_STEPS=30 RANKINV_DEALIAS=1 \
#     mpiexec -n R julia --project=. -t1 -O3 benchmarks/scalability_rankinv.jl
# prints per run: rankinv: model=ns ranks=R N=96 dealias=1 E=...

using MPI
MPI.Init()
using Printf
using Random
using SuperfluidDynamics

const MODEL = get(ENV, "RANKINV_MODEL", "ns")
const N = parse(Int, get(ENV, "RANKINV_N", "96"))
const STEPS = parse(Int, get(ENV, "RANKINV_STEPS", "30"))
const DEALIAS = parse(Int, get(ENV, "RANKINV_DEALIAS", "1")) == 1
# Measured limit of the PROBE (not of the package): inviscid NS on this
# broadband state at N=96 with Δt=0.01 goes NaN before step 30 — on the
# pre-fix AND the post-fix code (checked against b8a32ba). Use RANKINV_N=48
# for the rank-count comparison (that is also where the 1.9e-3 drift was
# originally measured); N=96 needs a smaller Δt if one wants stable runs there.

comm = MPI.COMM_WORLD
rank = MPI.Comm_rank(comm)
nranks = MPI.Comm_size(comm)

grid = Grid((N, N, N), ((-2π, 2π), (-2π, 2π), (-2π, 2π)))
# NS evolves a 3-component velocity (ndims=3); GP evolves the scalar order
# parameter (the default, ndims=1 — PotentialZero and the GP models are typed
# on the scalar field).
field = Field(grid, ComplexField(); ndims=(MODEL == "ns" ? 3 : 1))

rng = MersenneTwister(12345)
# Bands sitting between the strictest rank-local cutoff and the global one.
# The 2/3 bound is (4/9)·min_d kmax_d², and with fftfreq-ordered pencils the
# "edge" ranks hold only |ξ_d| ≤ ~N/4 — their bound is (4/9)(N/4)², a factor 4
# below the global (4/9)(N/2)². Modes with 0.20N < |k| < 0.33N are therefore
# kept by some ranks and deleted by others — if the bound is rank-local.
# (The NS projection removes the divergence at the first step; what matters is
# the spectral support, not the polarisation.)
fracs = [(0.16, 0.16, 0.08), (0.25, 0.08, 0.04), (0.04, 0.25, 0.08),
         (0.08, 0.04, 0.25), (0.20, 0.20, 0.08), (0.30, 0.12, 0.06)]
# Field allocates undef data: zero the evolving arrays before accumulating.
if MODEL == "ns"
    fill!(parent(field.ux), 0.0); fill!(parent(field.uy), 0.0); fill!(parent(field.uz), 0.0)
else
    fill!(parent(field.ϕ), 0.0)
end
# identical seeded values for both models; the velocity field just has u-parts
for (f1, f2, f3) in fracs
    k1, k2, k3 = max(1, round(Int, f1 * N)), max(1, round(Int, f2 * N)), max(1, round(Int, f3 * N))
    ph = 0.4 + 0.3 * rand(rng)
    if MODEL == "ns"
        @. field.ux += sin($k1 * field.x + $k2 * field.y + $k3 * field.z) * $ph
        @. field.uy += cos($(k1 + 1) * field.x + $k2 * field.y + $k3 * field.z) * (1 - $ph)
        @. field.uz += sin($k2 * field.x + $k3 * field.y + $(k1 ÷ 2 + 1) * field.z) * $ph
    else
        # the GP observable evolves ϕ, not u: seed the same bands there
        @. field.ϕ += exp(im * ($k1 * field.x + $k2 * field.y + $k3 * field.z)) * $ph
    end
end
if MODEL == "gp"
    field.ϕ ./= sqrt(sum(abs2.(field.ϕ)) * prod(grid.Δ))
end

model = if MODEL == "ns"
    NumModelRK4Imp(field, NavierStokesParameters(; ν=0.0), 0.01, 1, 1)
elseif MODEL == "gp"
    # NumModelGPRK is a damped (gradient-flow) solver that dealiases the
    # solution after every step — the GP site of the rank-local bound. Keep
    # the damping gentle (small β, small dt) so the observable stays O(1)
    # over the few steps of the probe.
    param = GrossPitaevskiiParameters(; coeffΔ=-0.5, β=1.0,
                                      pot=PotentialZero(field))
    NumModelGPRK(field, param, 1e-4, 1, 1)
else
    error("RANKINV_MODEL must be ns or gp")
end
if !DEALIAS
    # Only the explicit GP solver carries the switch (the NS RK4 dealiases
    # unconditionally); GP with dealias=0 is therefore the round-off baseline:
    # drift there = benign non-bit-reproducibility, drift that appears only
    # with dealias=1 = the rank-local cutoff bound.
    isdefined(model, :dealias) || error("this model has no `dealias` flag " *
                                        "(only RANKINV_MODEL=gp supports DEALIAS=0)")
    model.dealias = false
end

function observable(m)
    if MODEL == "ns"
        e = 0.0
        for c in 1:length(m.f.u)
            e += sum(real.(m.f.u[c] .^ 2))   # global reduction over the pencil
        end
        return 0.5 * e * prod(m.f.g.Δ)
    else
        # NumModelGPRK is a damped (imaginary-time-like) solver: its energy
        # collapses toward the ground state, a poor drift observable. The norm
        # sum(|ϕ|²) is a global reduction of the state and reacts to any
        # spectral truncation difference — use it for GP.
        return real(sum(abs2.(m.f.ϕ)))
    end
end

E0 = observable(model)
for _ in 1:STEPS
    SuperfluidDynamics.timeStep!(model)
end
E4 = observable(model)

if rank == 0
    @printf("rankinv: model=%s ranks=%d N=%d dealias=%d E0=%.14e E=%.14e\n",
            MODEL, nranks, N, DEALIAS, E0, E4)
    flush(stdout)
end
MPI.Barrier(comm)
