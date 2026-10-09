"""
    NumModelGPRK(f, param, Δt, niter, freqbckp; stepper="RK4", filter=:product)

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
- `"RK4"` — classical 4-stage RK (4th order).

Because the Laplacian is treated **explicitly** in the spectral domain, the
schemes are subject to the usual dispersive stability limit of explicit methods,

    |coeffΔ| k_max² Δt ≲ 2   (RK1),   |coeffΔ| k_max² Δt ≲ ~2.8   (RK4),

so the implicit / splitting schemes are preferred for long, high-resolution GP
evolutions.

`filter` places the 2/3-rule filter, which zeroes the modes above
`(2/3)·k_max`:

- `:product` — filter the nonlinear term `(V + β|ϕ|²)ϕ` inside every stage;
- `:solution` — filter the state after each step;
- `:none` — no filter.
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
    "2/3-rule filter bound (GLOBAL, rank-independent), cached at construction"
    " — recomputing it per step would be a device reduction + sync on the GPU."
    ξmax::Float64
    "where the 2/3-rule filter is applied: `:product`, `:solution` or `:none`."
    filter::Symbol
    writers::AbstractWriterCollection{F}
    "RK stage / derivative scratch (same layout as f.ϕ)."
    k1::PencilArray
    k2::PencilArray
    k3::PencilArray
    k4::PencilArray
    "physical-space scratch for the filtered nonlinear term (same layout as f.ϕ)."
    nl::PencilArray
    "spectral scratch (last-pencil layout) for filtering."
    shat::PencilArray
    s::PencilArray
end

function NumModelGPRK(f::AbstractField,
                      param::AbstractParameters,
                      Δt::Real, niter::Integer, freqbckp::Integer;
                      stepper::String="RK4", filter::Symbol=:product)
    filter in (:product, :solution, :none) ||
        throw(ArgumentError("NumModelGPRK filter $(filter) unknown (use :product, :solution or :none)."))
    gf = GradientField(f; rotation=true)
    plan = Plan(f)
    writer = WriterVTK(f); saver = WriterSave(f)
    writers = WriterCollection([writer, saver])
    bfun() = similar(f.ϕ)
    npen = getfield(SuperfluidDynamics, :last_pencil)(plan)
    shat = PencilArray{eltype(f.ϕ)}(undef, npen)
    return NumModelGPRK{typeof(f),typeof(param),typeof(plan)}(
        f, gf, param, Δt, niter, freqbckp, stepper, plan,
        SuperfluidDynamics.ξmax_global(plan), filter, writers,
        bfun(), bfun(), bfun(), bfun(), bfun(), shat, bfun())
end

function Base.show(io::IO, n::NumModelGPRK)
    return print(io,
                 "Gross-Pitaevskii explicit $(n.stepper) (2/3 filter: $(n.filter))\n",
                 "  ├───────  time step: $(n.Δt)\n",
                 "  └──────────── solve: number of iterations $(n.niter), backup frequency $(n.freqbckp)")
end

"""
    gp_rhs!(n, out, ψ)

Right-hand side of the time-dependent GP equation, ``∂tψ = F(ψ)``:

    F(ψ) = 1im [ ( −coeffΔ ∇² + iΩ L_z ) ψ − ( V + β |ψ|² ) ψ ]

computed with the package spectral operators (`lapRot`), written into `out`
(same layout as `ψ`). With `filter = :product` the nonlinear term is
2/3-filtered before being subtracted.
"""
function gp_rhs!(n::NumModelGPRK, out, ψ)
    lin = lapRot(n, ψ)                          # (−coeffΔ ∇² + iΩ L_z) ψ, spectral
    if n.filter === :product
        # The filter acts on the term, not on ψ: the stages stay a polynomial in Δt,
        # so the order of `stepper` is preserved.
        _gp_nl!(n, n.nl, ψ)
        @. out = 1im * (lin - n.nl)
    else
        V, β = n.param.pot.V, n.param.β
        @. out = 1im * (lin - (V + β * abs2(ψ)) * ψ)
    end
    return out
end

"""
    _gp_nl!(n, out, ψ)

Nonlinear term ``(V + β|ψ|²)ψ``, 2/3-filtered: built in physical space, filtered
in the spectral domain, returned in the layout of `ψ`. `out` must not alias `ψ`.
"""
function _gp_nl!(n::NumModelGPRK, out, ψ)
    V, β = n.param.pot.V, n.param.β
    @. out = (V + β * abs2(ψ)) * ψ
    mul_all!(n.shat, n.plan, out)
    _gp_filter!(n, n.shat)
    ldiv_all!(out, n.plan, n.shat)
    return out
end

"""
    _gp_filter!(n, ϕ̂)

Zero the modes of the spectral field `ϕ̂` above the 2/3-rule bound, in place
(same threshold as `dealias!` / `dealias2!` / `_dealias_scalar!`).
"""
function _gp_filter!(n::NumModelGPRK, ϕ̂)
    _dealias_scalar!(ϕ̂, getfield(SuperfluidDynamics, :spectral_grid)(n.plan), n.ξmax)
    return nothing
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
    if n.filter === :solution
        # Projecting the state is an O(Δt) perturbation: it dominates the truncation
        # error of RK4 and it removes spectral weight at every step.
        mul_all!(n.shat, n.plan, n.f.ϕ)
        _gp_filter!(n, n.shat)
        ldiv_all!(n.f.ϕ, n.plan, n.shat)
    end
    return 0
end
