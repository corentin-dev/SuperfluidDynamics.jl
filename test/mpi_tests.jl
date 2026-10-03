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

if rank == 0
    println("MPI tests done on $nrank ranks")
end
MPI.Finalize()
