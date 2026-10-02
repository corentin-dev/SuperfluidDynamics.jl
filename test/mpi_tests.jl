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

if rank == 0
    println("MPI tests done on $nrank ranks")
end
MPI.Finalize()
