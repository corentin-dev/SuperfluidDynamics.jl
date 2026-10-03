# Conservation check: long inviscid/nondissipative runs, drift of invariants.
# Answers "does the integrator hold what it should hold" — for the paper's
# validation section. One model per run, selected by MODEL=ns|gp|hvbk|nsgp.
#
#   MODEL=ns N=96  STEPS=2000 julia --project=. -t8 -O3 benchmarks/conservation.jl
#   MODEL=gp N=128 STEPS=2000 julia --project=. -t8 -O3 benchmarks/conservation.jl
#   MODEL=hvbk N=64 STEPS=2000 julia --project=. -t8 -O3 benchmarks/conservation.jl
#
# Expected behaviour (dt is fixed per model below):
#   ns (inviscid Taylor-Green, RK4): kinetic energy constant to ~1e-8 over the
#     run (no explicit dissipation; drift is integration error).
#   gp (Strang-2, defocusing): norm to machine precision by construction;
#     energy oscillates with bounded amplitude ~O(dt^2), no linear drift.
#   hvbk (two-fluid, nu>0): energy is NOT conserved (mutual friction + normal
#     viscosity dissipate); the check is monotone decay and no blow-up/NaN.
#   nsgp (2D two-fluid, nu>0): same expectation as hvbk (monotone decay of the
#     reported kinetic energy, no NaN).
# Prints one line per CHECKPOINT fraction of E/E0 and N/N0, so drift and
# oscillation are distinguishable from the log.

using SuperfluidDynamics
using SuperfluidDynamics: energy   # internal, not exported (as in mpi_tests.jl)
using Printf: @printf

MODEL = get(ENV, "MODEL", "ns")
N = parse(Int, get(ENV, "N", "96"))
STEPS = parse(Int, get(ENV, "STEPS", "2000"))
CHECKPOINT = parse(Int, get(ENV, "CHECKPOINT", "4"))

if MODEL == "ns"
    dt = 0.01
    grid = Grid((N, N, N), ((-2π, 2π), (-2π, 2π), (-2π, 2π)))
    field = Field(grid, ComplexField(); ndims=3)
    taylor_green!(field, field.x, field.y, field.z)
    model = NumModelRK4Imp(field, NavierStokesParameters(; ν=0.0), dt, 1, 1)
    inv0 = (energy(model)[4], nothing)  # [4] = total energy (NS and GP)
elseif MODEL == "hvbk"
    dt = 0.01
    grid = Grid((N, N, N), ((-2π, 2π), (-2π, 2π), (-2π, 2π)))
    fn_ = Field(grid, ComplexField(); ndims=3)
    fs_ = Field(grid, ComplexField(); ndims=3)
    # taylor_green! is divergence-free by construction (package kernel); the
    # superfluid counter-flow is a scaled copy, so both fields survive the
    # spectral projection (no spurious initial energy drop).
    taylor_green!(fn_, fn_.x, fn_.y, fn_.z)
    @. fs_.ux = -0.7 * fn_.ux
    @. fs_.uy = -0.7 * fn_.uy
    @. fs_.uz = 0.0
    model = NumModelHVBK(fn_, fs_, HVBKParameters(; ν=0.01, νs=0.001, rb=1.5,
                                                  ρn=1.0, ρs=1.0),
                         dt, 1, 1; stepper="RK2")
    inv0 = (energy(model)[4], nothing)
elseif MODEL == "nsgp"
    # 2D setup of the NSGP_2D example: regularised vortex + Taylor-Green.
    # NSGP_DIM=3 runs the same state, invariant along z (the 3D branch of the
    # friction force is then the one being applied, and the 2D/3D agreement is
    # established separately in test/runtests.jl).
    dt = 0.002
    DIM = parse(Int, get(ENV, "NSGP_DIM", "2"))
    NZ = parse(Int, get(ENV, "NSGP_NZ", "32"))
    dims = DIM == 3 ? (N, N, NZ) : (N, N)
    bounds = DIM == 3 ? ((-8, 8), (-8, 8), (-8, 8)) : ((-8, 8), (-8, 8))
    grid = Grid(dims, bounds)
    fgp_ = Field(grid, ComplexField(); ndims=DIM)
    fns_ = Field(grid, ComplexField(); ndims=DIM)
    aa = 0.8
    shp = DIM == 3 ? (:, 1, 1) : (:, 1)
    X2 = reshape(vec(fgp_.x), shp...)
    shp2 = DIM == 3 ? (1, :, 1) : (1, :)
    Y2 = reshape(vec(fgp_.y), shp2...)
    r2 = X2.^2 .+ Y2.^2
    fgp_.ϕ .= (sqrt.(r2) ./ sqrt.(r2 .+ aa^2)) .* exp.(1im * atan.(Y2, X2))
    kx = 2π / grid.Lx; ky = 2π / grid.Ly
    @. fns_.ux = 0.3 * sin(ky * fns_.y) * cos(kx * fns_.x)
    @. fns_.uy = -0.3 * cos(ky * fns_.y) * sin(kx * fns_.x)
    DIM == 3 && (@. fns_.uz = 0)  # z-invariant state, no mean flow along z
    model = NumModelNSGP(fgp_, fns_,
                         NSGPParameters(; α=-0.02, ν=0.01, β=1.0, ρn=0.5, ρs=0.5,
                                        Btab=0.4, Bptab=0.1, ξ=1.0, ε2=0.05,
                                        one_way=true),
                         dt, 1, 1; stepper="RK2Imp")
    inv0 = (energy(model)[4], nothing)
elseif MODEL == "gp"
    dt = 0.005
    grid = Grid((N, N, N), ((-π, π), (-π, π), (-π, π)))
    field = Field(grid, ComplexField())
    initField!(InitGauss(field; γz=1.0, Ω=1.0))
    SuperfluidDynamics.normalize!(field)
    param = GrossPitaevskiiParameters(; coeffΔ=-0.075, β=27.0,
                                      pot=PotentialZero(field))
    model = NumModelSplit2(field, param, dt, 1, 1)
    inv0 = (energy(model)[4], SuperfluidDynamics.norm(field))
else
    error("MODEL must be ns or gp")
end

# the NSGP 2D and 3D runs are different measurements of the same MODEL: tag
# the dimension into the header so the collector keeps them apart
dimtag = (MODEL == "nsgp" && @isdefined(DIM)) ? string("-d", DIM) : ""
@printf("conservation: model=%s%s N=%d steps=%d dt=%g E0=%.10e\n",
        MODEL, dimtag, N, STEPS, dt, inv0[1])

period = max(1, div(STEPS, CHECKPOINT))
t0 = time_ns()
for s in 1:STEPS
    SuperfluidDynamics.timeStep!(model)
    if s % period == 0
        E = energy(model)[4]
        isfinite(E) || (@printf("  step=%-6d E=%s NON FINI\n", s, E); break)
        if inv0[2] === nothing
            @printf("  step=%-6d E/E0=%.10f\n", s, E / inv0[1])
        else
            nrm = SuperfluidDynamics.norm(field)
            @printf("  step=%-6d E/E0=%.10f  N/N0=%.12f\n", s, E / inv0[1], nrm / inv0[2])
        end
    end
end
@printf("conservation: model=%s done in %.0f s\n", MODEL, (time_ns() - t0) / 1e9)
