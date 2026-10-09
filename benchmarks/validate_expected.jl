# Comparison against the expected-results directory provided by the reviewer
# (runs repo: expected-results/expected/). src/ is untouched: this bench maps
# the package onto the reference conventions and reports PASS/FAIL against the
# manifest tolerances.
#
# Conventions (expected/README.md): hbar=m=1, i psi_t = -1/2 Lap psi + V psi +
# beta|psi|^2 psi, norm 1, coeffDelta=-0.5. The package chemical potential is
# mu = E_kin+pot + 2 E_beta (energy(showEnergy=true) prints it), identical to
# the reference mu = E + (beta/2)·∫|psi|^4.
#
# 1D does not exist in the package (Field is 2D/3D): Bao-Du 1D rows run as a
# 2D slab with V_y=0 and Dy=1 exactly — then norm, E, mu, x_rms, phi(0) are
# the 1D quantities (Dy divides out; a constant-y field has no y-gradients).
#
#   julia --project=. -t8 -O3 --color=no benchmarks/validate_expected.jl [cas...]
#   cas ∈ baodu1d | baodu2d3d | vortex | bogoliubov     (default: baodu1d)
# env EXPECTED_DIR = path to the expected/ directory.

using Printf

const EXPECTED_DIR = get(ENV, "EXPECTED_DIR",
                         joinpath(@__DIR__, "..", "..", "SuperfluidDynamics-runs",
                                  "expected-results", "expected"))

using SuperfluidDynamics

const results = Tuple{String, Float64, Float64, Bool}[]

function record(case::AbstractString, computed::Real, expected::Real, tol)
    ok = abs(computed - expected) <= tol * max(abs(expected), 1e-12)
    push!(results, (String(case), Float64(computed), Float64(expected), ok))
    @printf("  %-42s %12.6g vs %12.6g (tol %.0e)  %s\n",
            case, computed, expected, tol, ok ? "PASS" : "FAIL")
    return ok
end

function readcsv(path)
    lines = readlines(path)
    head = Tuple(Symbol.(split(lines[1], ',')))   # a VALUE of symbols, NamedTuple{head}
    rows = NamedTuple[]
    for l in lines[2:end]
        isempty(strip(l)) && continue
        occursin('"', l) && continue   # quoted domains (2D3D file): rows used are coded below
        vals = map(split(l, ',')) do x
            tryparse(Float64, x) isa Float64 ? parse(Float64, x) : x
        end
        push!(rows, NamedTuple{head}(vals))
    end
    return rows
end

# ---------------------------------------------------------------------------
# imaginary-time ground state (fixed point: converged answer is dt-independent)
# ---------------------------------------------------------------------------
function ground_state(dim::Int, gammas; beta, L, Nx, dt, niter, m::Integer=0,
                      MT=NumModelBackwardEuler)
    dims = ntuple(i -> Nx[i], dim)
    box = ntuple(i -> (-L[i], L[i]), dim)
    grid = Grid(dims, box)
    field = Field(grid, ComplexField())
    coords = dim == 2 ? (field.x, field.y) : (field.x, field.y, field.z)
    r2 = coords[1] .^ 2
    for c in coords[2:end]
        r2 = r2 .+ c .^ 2
    end
    @. field.ϕ = exp(-0.5 * r2)
    if m > 0
        θ = atan.(coords[2], coords[1])
        @. field.ϕ *= tanh(sqrt(r2) / 1.5)^m * exp(im * m * θ)
    end
    normalize!(field)
    γnames = (:γx, :γy, :γz)
    pot = PotentialQuadratic(field; [γnames[i] => gammas[i] for i in 1:dim]...)
    param = GrossPitaevskiiParameters(; coeffΔ=-0.5, β=beta, Ω=0.0, pot=pot)
    # Backward Euler: a descent scheme, which is what drives E to the minimum.
    # Crank-Nicolson / CN-QuasiNewton are not: nothing pushes E down, so they can
    # settle on a fixed point that is not the ground state.
    n = MT(field, param, dt, niter, 10 * niter)
    prev, stall = Inf, 0
    for it in 1:niter
        SuperfluidDynamics.timeStep!(n)
        if mod1(it, 25) == 0
            _, _, _, E = SuperfluidDynamics.energy(n)
            if abs(E - prev) < 1e-13 * max(1, abs(E)); stall += 1; stall ≥ 2 && break
            else; stall = 0; end
            prev = E
        end
    end
    EΩ, EΔ, Eβ, E = SuperfluidDynamics.energy(n)
    μ = -EΩ + EΔ + 2 * Eβ
    dv = prod(grid.Δ)
    norm2 = sum(abs2.(field.ϕ)) * dv
    rms = ntuple(i -> sqrt(sum((coords[i] .^ 2) .* abs2.(field.ϕ)) * dv), dim)
    ϕmax = maximum(abs, parent(field.ϕ))     # single process: local max = global
    return (E=E, μ=μ, norm2=norm2, rms=rms, ϕmax=ϕmax)
end

# ---------------------------------------------------------------------------
# Bao & Du 1D (Table 4.1): 2D slab, Vy=0, Dy=1 exactly (box length = Ny)
# ---------------------------------------------------------------------------
function case_baodu1d()
    println("== ground_states_bao_du_1D.csv (Bao & Du 2004, Table 4.1, tol 1e-3) ==")
    for r in readcsv(joinpath(EXPECTED_DIR, "ground_states_bao_du_1D.csv"))
        β = r.beta
        # ONE y-plane, box height Ly = Ny·Δy = 1: the 2D quadrature
        # Σ|ψ|²ΔxΔy then IS the 1D quadrature Σ|ψ|²Δx — norm, E, μ, x_rms,
        # φ(0) are the 1D quantities exactly (two planes would double-count
        # the integral: the interaction term scales as 1/2, not 1).
        # γy=0: no y-confinement; the field is y-constant (one plane), so the
        # y-derivative terms vanish.
        g = ground_state(2, (1.0, 0.0); beta=β, L=(16.0, 0.5), Nx=(512, 1),
                         dt=0.05, niter=4000)
        abs(g.norm2 - 1) < 1e-8 || @warn "β=$β: norme hors de 1" g.norm2
        record("baodu1d β=$β E", g.E, r.E, 1e-3)
        record("baodu1d β=$β μ", g.μ, r.mu, 1e-3)
        record("baodu1d β=$β x_rms", g.rms[1], r.x_rms, 1e-3)
        record("baodu1d β=$β φ(0)", g.ϕmax, r.phi_g_0, 5e-3)
    end
end

# ---------------------------------------------------------------------------
# Bao & Du 2D/3D (Ex. 3-4). Rows with the gaussian stirrer are skipped: src/ has
# no stirrer potential (zero/quadratic/quartic only) and the reference files do
# not give its formula.
# ---------------------------------------------------------------------------
function case_baodu2d3d()
    println("== ground_states_bao_du_2D3D.csv (Bao & Du 2004, Ex. 3-4) ==")
    cases = [(; name="2D_I", dim=2, gam=(1.0, 4.0), L=(8.0, 4.0), Nx=(256, 256),
              tol=1e-3, E=11.1563, μ=16.3377, rms=(2.2734, 0.6074)),
             (; name="3D_I", dim=3, gam=(1.0, 2.0, 4.0), L=(8.0, 6.0, 4.0),
              Nx=(96, 96, 96), tol=1e-2, E=8.33, μ=11.03, rms=(1.67, 0.87, 0.49))]
    for c in cases
        println("  (cas $(c.name))")
        g = ground_state(c.dim, c.gam; beta=200.0, L=c.L, Nx=c.Nx, dt=0.02, niter=6000)
        abs(g.norm2 - 1) < 1e-8 || @warn "$(c.name): norme hors de 1" g.norm2
        record("$(c.name) E", g.E, c.E, c.tol)
        record("$(c.name) μ", g.μ, c.μ, c.tol)
        for d in 1:c.dim
            record("$(c.name) rms[$d]", g.rms[d], c.rms[d], 5e-3)
        end
    end
    println("  REPORT: stirrer rows NOT run — no stirrer potential in src/, exact")
    println("  V formula missing from the provided files (asked the user).")
end

# ---------------------------------------------------------------------------
# Bao & Du central vortex (Table 4.2): Cartesian 2D, V=r^2/2, winding m
# ---------------------------------------------------------------------------
function case_vortex()
    println("== central_vortex_bao_du_2D.csv (Bao & Du 2004, Table 4.2, β=200) ==")
    for r in readcsv(joinpath(EXPECTED_DIR, "central_vortex_bao_du_2D.csv"))
        m = Int(r.m)
        g = ground_state(2, (1.0, 1.0); beta=200.0, L=(8.0, 8.0), Nx=(256, 256),
                         dt=0.02, niter=8000, m=m)
        abs(g.norm2 - 1) < 1e-8 || @warn "m=$m: norme hors de 1" g.norm2
        rrms = sqrt(g.rms[1]^2 + g.rms[2]^2)   # isotropic trap: <r²>=<x²>+<y²>
        record("vortex m=$m E", g.E, r.E, 1e-3)
        record("vortex m=$m μ", g.μ, r.mu, 1e-3)
        record("vortex m=$m r_rms", rrms, r.r_rms, 5e-3)
    end
end

# ---------------------------------------------------------------------------
# Bogoliubov dispersion (uniform background gn0=1, tol 1e-8): every computed
# frequency must invert ω² = (k²/2)² + k² to an INTEGER k² = n²+m². This tests
# the values without assuming which modes Arpack returned.
# ---------------------------------------------------------------------------
function case_bogoliubov()
    println("== bogoliubov_dispersion_gn0_1.csv (dispersion exacte, tol 1e-8) ==")
    N = 32
    grid = Grid((N, N), ((-π, π), (-π, π)))
    field = Field(grid, ComplexField())
    fill!(parent(field.ϕ), 1.0)   # flat ψ0 = 1: μ = β = 1, exact GP ground state
    param = BdGParameters(; coeffΔ=-0.5, β=1.0, pot=PotentialZero(field), Ω=0.0)
    # :SM stalls on the flat background (fully degenerate k=0 eigenspace
    # exhausts the Krylov space); :LM returns the HIGH plane-wave modes,
    # which invert just as exactly through the dispersion formula.
    n = NumModelBdG(field, param, 1, 1; nev=6, which=:LM)
    SuperfluidDynamics.timeStep!(n)
    for (j, ω) in enumerate(sort(n.ωs))
        k2 = -2.0 + 2.0 * sqrt(1.0 + ω^2)      # inverse of ω²=(k²/2)²+k² at gn0=1
        record("BdG mode $j: k² entier", k2, round(k2), 1e-8 * max(1, k2))
    end
end

const CASES = Dict("baodu1d" => case_baodu1d, "baodu2d3d" => case_baodu2d3d,
                   "vortex" => case_vortex, "bogoliubov" => case_bogoliubov)
const todo = isempty(ARGS) ? ["baodu1d"] : ARGS
t0 = time()
for c in todo
    haskey(CASES, c) || error("cas inconnu: $c")
    CASES[c]()
end
nfail = count(x -> x[4], map(x -> !x[4], results))
println("\n==== $(length(results)) comparaisons, $(count(x -> !x[4], results)) échecs, ",
        round(Int, time() - t0), " s ====")
for (case, computed, expected, ok) in results
    println("validate_expected: $(ok ? "PASS" : "FAIL") $case calc=$computed attendu=$expected")
end
