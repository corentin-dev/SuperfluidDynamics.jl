"""
    NumModelGPRK(f, param, Δt, niter, freqbckp; stepper="RK4")

Explicit Runge-Kutta integrator for the time-dependent Gross-Pitaevskii
equation, in 2D and 3D:

    i ∂tϕ = ( V + β |ϕ|² ) ϕ − ( −coeffΔ ∇² + iΩ L_z ) ϕ ,   i.e. ∂tϕ = 1im [ lapRot(n,ϕ) − (V + β|ϕ|²)ϕ ]

where `V` is the external potential, `β` the interaction strength and
`L_z = (y ∂x − x ∂y)` the rotation operator. This is the **explicit**
counterpart of the package's implicit and splitting real-time schemes
(CrankNicolsonT, Split1/2) and a port of the reference GPS `GP_RK4`
(`"UnsteadyRK4"`) time integrator. `stepper` ∈ `"RK1"`, `"RK2"`, `"RK4"`:

- `"RK1"` — forward Euler (1st order);
- `"RK2"` — explicit midpoint (2nd order);
- `"RK4"` — classical 4-stage RK (4th order), as in the reference.

Because the Laplacian is treated **explicitly** in the spectral domain, the
schemes are unconditionally *accurate* but subject to the usual dispersive
stability limit of explicit methods,

    |coeffΔ| k_max² Δt ≲ 2   (RK1),   |coeffΔ| k_max² Δt ≲ ~2.8   (RK4),

which is why the implicit / splitting schemes are preferred for long,
high-resolution GP evolutions. `NumModelGPRK` is intended for benchmarks,
short runs, and as an accuracy reference for the implicit schemes.

As in the reference, the solution is 2/3-rule dealiased after every step
(`dealias`, default `true`): without it, the explicit stages amplify the
high-frequency round-off of the purely dispersive GP spectrum.
"""
mutable struct NumModelGPRK{F,P,Plan} <: AbstractNumModel{F,P,Plan}
    f::F
    gf::Any
    param::P
    Δt::Real
    niter::Integer
    freqbckp::Integer
    "time-stepping scheme: \"RK1\", \"RK2\" or \"RK4\"."
    stepper::String
    plan::Plan
    writers::AbstractWriterCollection{F}
    "RK stage / derivative scratch (same layout as f.ϕ)."
    k1::PencilArray
    k2::PencilArray
    k3::PencilArray
    k4::PencilArray
    s::PencilArray
    "spectral scratch (last-pencil layout) for the solution dealiasing."
    shat::PencilArray
    "apply the 2/3-rule dealiasing to the solution after each step (default)."
    dealias::Bool
end

function NumModelGPRK(f::AbstractField,
                      param::AbstractParameters,
                      Δt::Real, niter::Integer, freqbckp::Integer;
                      stepper::String="RK4")
    gf = GradientField(f; rotation=true)
    plan = Plan(f)
    writer = WriterVTK(f); saver = WriterSave(f)
    writers = WriterCollection([writer, saver])
    bfun() = similar(f.ϕ)
    npen = getfield(SuperfluidDynamics, :last_pencil)(plan)
    shat = PencilArray{eltype(f.ϕ)}(undef, npen)
    return NumModelGPRK{typeof(f),typeof(param),typeof(plan)}(
        f, gf, param, Δt, niter, freqbckp, stepper, plan, writers,
        bfun(), bfun(), bfun(), bfun(), bfun(), shat, true)
end

function Base.show(io::IO, n::NumModelGPRK)
    return print(io,
                 "Gross-Pitaevskii explicit $(n.stepper)\n",
                 "  ├───────  time step: $(n.Δt)\n",
                 "  └──────────── solve: number of iterations $(n.niter), backup frequency $(n.freqbckp)")
end

"""
    gp_rhs!(n, out, ψ)

Right-hand side of the time-dependent GP equation, ``∂tψ = F(ψ)``:

    F(ψ) = 1im [ ( −coeffΔ ∇² + iΩ L_z ) ψ − ( V + β |ψ|² ) ψ ]

computed with the package spectral operators (`lapRot`), written into `out`
(same layout as `ψ`).
"""
function gp_rhs!(n::NumModelGPRK, out, ψ)
    lin = lapRot(n, ψ)                          # (−coeffΔ ∇² + iΩ L_z) ψ, spectral
    V, β = n.param.pot.V, n.param.β
    @. out = 1im * (lin - (V + β * abs2(ψ)) * ψ)
    return out
end

function timeStep!(n::NumModelGPRK)
    ψ₀ = n.f.ϕ
    Δt = n.Δt
    if n.stepper == "RK1"
        k1 = gp_rhs!(n, n.k1, ψ₀)
        @. n.f.ϕ = ψ₀ + Δt * k1
    elseif n.stepper == "RK2"
        k1 = gp_rhs!(n, n.k1, ψ₀)
        @. n.s   = ψ₀ + 0.5 * Δt * k1
        k2 = gp_rhs!(n, n.k2, n.s)
        @. n.f.ϕ = ψ₀ + Δt * k2
    elseif n.stepper == "RK4"
        k1 = gp_rhs!(n, n.k1, ψ₀)
        @. n.s   = ψ₀ + 0.5 * Δt * k1
        k2 = gp_rhs!(n, n.k2, n.s)
        @. n.s   = ψ₀ + 0.5 * Δt * k2
        k3 = gp_rhs!(n, n.k3, n.s)
        @. n.s   = ψ₀ + Δt * k3
        k4 = gp_rhs!(n, n.k4, n.s)
        @. n.f.ϕ = ψ₀ + Δt / 6 * (k1 + 2k2 + 2k3 + k4)
    else
        throw(ArgumentError("NumModelGPRK stepper \"$(n.stepper)\" unknown (use \"RK1\", \"RK2\" or \"RK4\")."))
    end
    # 2/3-rule dealiasing of the solution (as in the reference GP_RK4):
    # removes the high-frequency round-off that explicit RK would otherwise amplify.
    if n.dealias
        _gp_dealias!(n)
    end
    return 0
end

"""
    _gp_dealias!(n)

2/3-rule dealiasing of the model wavefunction: FFT → zero the modes with
``|k|² > (4/9) min(|k|_max)²`` → IFFT. Applied to the solution after each
explicit step to control the high-frequency round-off growth that explicit
Runge-Kutta methods exhibit on the dispersive Gross-Pitaevskii spectrum.
"""
function _gp_dealias!(n::NumModelGPRK)
    gridξ = getfield(SuperfluidDynamics, :spectral_grid)(n.plan)
    mul_all!(n.shat, n.plan, n.f.ϕ)
    if ndims(gridξ) == 3
        func = x -> x^2
        ξmax = 4 / 9 * minimum((maximum(func, gridξ.x.data),
                                maximum(func, gridξ.y.data),
                                maximum(func, gridξ.z.data)))
        @. n.shat *= (gridξ.x^2 + gridξ.y^2 + gridξ.z^2) < ξmax
    else
        func = x -> x^2
        ξmax = 4 / 9 * minimum((maximum(func, gridξ.x.data),
                                maximum(func, gridξ.y.data)))
        @. n.shat *= (gridξ.x^2 + gridξ.y^2) < ξmax
    end
    ldiv_all!(n.f.ϕ, n.plan, n.shat)
    return nothing
end
