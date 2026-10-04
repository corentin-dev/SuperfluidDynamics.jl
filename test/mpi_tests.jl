# Multi-rank MPI validation.
#
# Run under MPI (not from runtests.jl, which is single-process):
#
#     mpiexec -n 2 julia --project=. test/mpi_tests.jl
#
# The point: every diagnostic must return the GLOBAL value on every rank
# (PencilArrays reduces sum() with MPI.Allreduce). A diagnostic that sums
# parent(pencil_array) instead of the PencilArray only sees the local
# subdomain, which single-process runs cannot detect.
using MPI
MPI.Init()

using Test
using SuperfluidDynamics
using SuperfluidDynamics: energy
using Printf

comm = MPI.COMM_WORLD
rank, nrank = MPI.Comm_rank(comm), MPI.Comm_size(comm)

@testset "MPI diagnostics ($nrank ranks)" begin
    # --- 2D NS: uniform velocity, analytic kinetic energy ----------------
    # u = v = 1 on a Lx x Ly box:  E = ½ ∫ (u² + v²) dA = Lx·Ly.
    grid = Grid((32, 24), ((-4, 4), (-5, 5)))
    field = Field(grid, ComplexField(); ndims=2)
    n = NumModelRK4Imp(field, NavierStokesParameters(; ν=0.01), 0.01, 1, 1)
    fill!(parent(field.ux), 1.0)
    fill!(parent(field.uy), 1.0)
    E = energy(n)[2]
    @test abs(E - grid.Lx * grid.Ly) / (grid.Lx * grid.Ly) < 1e-12

    # Same quantity on every rank (Allreduce, not local sum).
    Eall = MPI.Allreduce(E, +, comm) / nrank
    @test Eall == E

    # --- 2D GP: norm and energy are rank-invariant -----------------------
    fieldc = Field(grid, ComplexField())
    X = reshape(vec(fieldc.x), :, 1); Y = reshape(vec(fieldc.y), 1, :)
    @. fieldc.ϕ = exp(-0.5 * (X^2 + Y^2))
    SuperfluidDynamics.normalize!(fieldc)
    param = GrossPitaevskiiParameters(coeffΔ=-0.5, β=10.0, Ω=0.0,
                                      pot=PotentialQuadratic(fieldc, γx=1.0, γy=1.0))
    ngp = NumModelCrankNicolson(fieldc, param, 0.01, 2, 2)
    Egp = energy(ngp)
    for e in Egp
        # identical (not subdomain-dependent) on every rank
        @test MPI.Allreduce(e, +, comm) / nrank ≈ e rtol = 1e-12
    end

    # --- field norm is global --------------------------------------------
    nrm = SuperfluidDynamics.norm(fieldc)
    @test abs(nrm - 1.0) < 1e-12
end


# --- Navier-Stokes time stepping vs the exact Taylor-Green decay ----------
# On a 2π-periodic box with k = (1, 1) the Taylor-Green vortex is a steady Euler
# solution, so the viscous decay is exactly E(t)/E(0) = exp(-4 ν t) whatever the
# number of ranks. This checks the transposes + FFTs + projection end to end.
@testset "MPI NS Taylor-Green decay ($nrank ranks)" begin
    grid = Grid((32, 32), ((-π, π), (-π, π)))
    field = Field(grid, ComplexField(); ndims=2)
    ν, Δt, nsteps = 0.01, 0.01, 40
    n = NumModelRK4Imp(field, NavierStokesParameters(; ν=ν), Δt, nsteps, 1)
    taylor_green!(field)
    E0 = energy(n)[2]
    for _ in 1:nsteps
        SuperfluidDynamics.timeStep!(n)
    end
    @test energy(n)[2] / E0 ≈ exp(-4 * ν * nsteps * Δt) rtol = 1e-6
end

# --- BdG: distributed eigensolve vs analytic spectrum ----------------------
# Same non-interacting anisotropic oscillator as in runtests.jl: ω = {ωx, ωy, 2ωx}.
@testset "MPI BdG analytic spectrum ($nrank ranks)" begin
    grid = Grid((20, 20), ((-6.0, 6.0), (-6.0, 6.0)))
    field = Field(grid, ComplexField())
    ωx, ωy = 1.0, sqrt(2.0)
    @. field.ϕ = exp(-ωx * field.x^2 / 2 - ωy * field.y^2 / 2)
    field.ϕ ./= sqrt(sum(abs2, field.ϕ) * grid.Δx * grid.Δy)
    pot = PotentialQuadratic(field; γx=ωx, γy=ωy^2)
    param = BdGParameters(coeffΔ=-0.5, β=0.0, pot=pot, Ω=0.0)
    n = NumModelBdG(field, param, 1, 1; nev=3)
    SuperfluidDynamics.timeStep!(n)       # must neither hang nor error
    @test n.mu ≈ (ωx + ωy) / 2 atol = 1e-3
    @test length(n.ωs) == 3
    @test all(abs.(n.ωs .- [ωx, ωy, 2ωx]) .< 1e-3)
    # every rank holds the same eigenvalues
    @test MPI.Allreduce(n.ωs, +, comm) ./ nrank ≈ n.ωs
end

# --- Dealiasing bound: GLOBAL, independent of the rank count --------------
# The 2/3-rule bound must come from the plan's global wavenumber vectors. On
# several ranks the local (pencil) grid sees only part of the spectrum; the
# old rank-local bound differed per rank and made the physics rank-dependent
# (measured 1.9e-3 energy drift 1→4 ranks, benchmarks/scalability_rankinv.jl).
# This test pins the value on a DISTRIBUTED grid, which single-process
# runtests.jl cannot do (there, local == global by construction).
@testset "MPI ξmax_global is the global (4/9) bound ($nrank ranks)" begin
    for dims in ((48, 48), (48, 48, 48))
        g = Grid(dims, ntuple(_ -> (-2π, 2π), length(dims)))
        f = Field(g, ComplexField(); ndims=(length(dims) == 3 ? 3 : 1))
        p = SuperfluidDynamics.Plan(f)
        ξmax = SuperfluidDynamics.ξmax_global(p)
        expected = (4 / 9) * minimum((π / g.Δ[d])^2 for d in 1:length(dims))
        @test ξmax ≈ expected rtol = 1e-12
        # every rank sees the same bound (no rank-local truncation)
        @test MPI.Allreduce(ξmax, +, comm) / nrank ≈ ξmax rtol = 0 atol = 0
    end
end

# --- NS energy after stepping: rank-count invariant ------------------------
# Same fixed problem at 1/2/4 ranks must give the SAME energy to ~machine
# precision: with the global bound the dealiasing mask is identical on every
# rank, so the whole computation is a partition of deterministic arithmetic.
# The 1-rank reference below comes from running this file with -n 1 (see the
# printed value; benchmarks/scalability_rankinv.jl measures the same fact on
# larger grids). The 1-rank dealias=0 floor (runtests-style) is bit-reproducible.
@testset "MPI NS energy is rank-count invariant ($nrank ranks)" begin
    grid = Grid((48, 48, 48), ntuple(_ -> (-π, π), 3))
    field = Field(grid, ComplexField(); ndims=3)
    ν, Δt, nsteps = 0.01, 0.01, 10
    n = NumModelRK4Imp(field, NavierStokesParameters(; ν=ν), Δt, nsteps, 10 * nsteps)
    # fixed broadband divergence-free initial state, identical on all ranks
    # (analytic function of the coordinates: no RNG layout dependence)
    @. field.ux = sin(field.x) * cos(field.y) * cos(field.z)
    @. field.uy = -cos(field.x) * sin(field.y) * cos(field.z)
    @. field.uz = sin(field.x) * sin(field.y) * sin(field.z)   # div-free: ∂x u + ∂y v + ∂z w = 0? not exactly,
    # but the projection is deterministic and rank-independent — what matters
    # here is the reproducibility of the FULL pipeline, not the specific state.
    # make the state divergence-free analytically:
    #   u = sin x cos y cos z, v = -cos x sin y cos z, w = 0 is div-free.
    fill!(parent(field.uz), 0.0)
    E0 = energy(n)[2]
    for _ in 1:nsteps
        SuperfluidDynamics.timeStep!(n)
    end
    E1 = energy(n)[2]
    ratio = E1 / E0
    # 1-rank reference (pinned with this exact code at -n 1 on v2.1.0+fix)
    # measured with this exact code at -n 1 (v2.1.0 + ξmax_global fix):
    # 1-rank reference, measured with this exact code at -n 1 (v2.1.0 + fix).
    # 2/4 ranks reproduce it to 1 ulp (only the Allreduce summation order of
    # `energy` differs); the pre-fix rank-local bound drifted by ~1e-3.
    REF = 0.9940155853022443
    @test ratio ≈ REF rtol = 1e-12
    rank == 0 && println("   [rank-invariance] ratio E1/E0 = ", ratio)
end

if rank == 0
    println("MPI tests done on $nrank ranks")
end
MPI.Finalize()
