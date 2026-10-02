# Conservation check: long inviscid/nondissipative runs, drift of invariants.
# Answers "does the integrator hold what it should hold" — for the paper's
# validation section. One model per run, selected by MODEL=ns|gp.
#
#   MODEL=ns N=96  STEPS=2000 julia --project=. -t8 -O3 benchmarks/conservation.jl
#   MODEL=gp N=128 STEPS=2000 julia --project=. -t8 -O3 benchmarks/conservation.jl
#
# Expected behaviour (dt is fixed per model below):
#   ns (inviscid Taylor-Green, RK4): kinetic energy constant to ~1e-8 over the
#     run (no explicit dissipation; drift is integration error).
#   gp (Strang-2, defocusing): norm to machine precision by construction;
#     energy oscillates with bounded amplitude ~O(dt^2), no linear drift.
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

@printf("conservation: model=%s N=%d steps=%d dt=%g E0=%.10e\n",
        MODEL, N, STEPS, dt, inv0[1])

period = max(1, div(STEPS, CHECKPOINT))
t0 = time_ns()
for s in 1:STEPS
    SuperfluidDynamics.timeStep!(model)
    if s % period == 0
        E = energy(model)[4]
        if inv0[2] === nothing
            @printf("  step=%-6d E/E0=%.10f\n", s, E / inv0[1])
        else
            nrm = SuperfluidDynamics.norm(field)
            @printf("  step=%-6d E/E0=%.10f  N/N0=%.12f\n", s, E / inv0[1], nrm / inv0[2])
        end
    end
end
@printf("conservation: model=%s done in %.0f s\n", MODEL, (time_ns() - t0) / 1e9)
