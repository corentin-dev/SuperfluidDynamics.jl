mutable struct NumModelSplit1{F,P,Plan} <: AbstractNumModel{F,P,Plan}
    f::F
    gf::Any
    param::P
    Δt::Real
    niter::Integer
    freqbckp::Integer
    plan::Plan
    writers::AbstractWriterCollection{F}
end

"""
$(TYPEDSIGNATURES)

Returns a first order operator splitting scheme numerical model (only FFT).

# Detail

Each step applies the two exact flows of the equation over the full time step:

- the linear flow ``exp(i Δt (-coeffΔ ∇² + Ω L_z))``, computed in Fourier space
  direction by direction (no linear system is solved);
- the nonlinear flow ``exp(-i (V + β |ϕ|²) Δt)``, exact in physical space.

# No dealiasing

Neither flow is dealiased: no 2/3-rule filter, no padding, nothing applied to the
state after the step. This matches the time-splitting schemes of the reference GPS
implementation, which are called without its `flag_cutoff`; the package does filter
elsewhere (`NumModelGPRK(filter = …)`, and the Navier-Stokes and HVBK models), just
not here.

The consequence is a resolution constraint, not a bug. The nonlinear flow multiplies
a band-limited state by ``exp(-i (V + β |ϕ|²) Δt)``, which is not band-limited, so
content above the Nyquist frequency is generated every step and folds back onto the
resolved modes. ``Δt`` controls how much: evaluating that flow directly on the
published dipole initial datum (after projecting it onto the low 2/3 band, so the
state starts strictly band-limited) leaves ``8.2e-06`` of the kinetic energy above
``2/3·kmax`` at ``Δt = 0.01``, ``2.1e-06`` at ``0.005`` and ``8.2e-08`` at ``0.001`` —
proportional to ``Δt²``. Over long runs that content grows, so `Δt` cannot be chosen
freely once `dx` is small: on the dipole family the measured stable/exploding split
falls in ``|coeffΔ| k_max² Δt``, with no exception among the four settings tested
(0.126, 0.505 and 1.011 stable; 2.021 explodes) — not in `Δt` or `dx` alone. Energy
is conserved to ``1e-06`` over 2 000 steps at ``|coeffΔ| k_max² Δt = 0.126``, and not
conserved at all at ``2.021`` (``E/E₀ = 338`` at ``t = 100``). Use
`NumModelGPRK(filter = …)` if a filtered scheme is wanted.

# Example

```jldoctest
julia> param = GrossPitaevskiiParameters(β = 1000, Ω = 0.8, pot = PotentialZero(field));
julia> nummodel = NumModelSplit1(field, param, 0.01, 1000, 100)
Splitting Order 1
  ├───────  time step: 0.01
  └──────────── solve: number of iterations 1000, backup frequency 100
```
"""
function NumModelSplit1(f::AbstractField, param::AbstractParameters,
                        Δt::Real, niter::Integer, freqbckp::Integer)
    gf = GradientField(f; rotation=true)
    plan = Plan(f)
    writer = WriterVTK(f)
    saver = WriterSave(f)
    writers = WriterCollection([writer, saver])
    return NumModelSplit1{typeof(f),typeof(param),typeof(plan)}(f, gf, param,
                                                                Δt, niter, freqbckp,
                                                                plan,
                                                                writers)
end

function Base.show(io::IO, n::NumModelSplit1)
    return print(io,
                 "Splitting Order 1\n",
                 "  ├───────  time step: $(n.Δt)\n",
                 "  └──────────── solve: number of iterations $(n.niter), backup frequency $(n.freqbckp)")
end

function timeStep!(n::NumModelSplit1)
    solveLapRot!(n, n.Δt)
    solveNL!(n, n.Δt)
    return 1
end

mutable struct NumModelSplit2{F,P,Plan} <: AbstractNumModel{F,P,Plan}
    f::F
    gf::Any
    param::P
    Δt::Real
    niter::Integer
    freqbckp::Integer
    plan::Plan
    writers::AbstractWriterCollection{F}
end

"""
$(TYPEDSIGNATURES)

Returns a second order Strang splitting scheme numerical model (only FFT).

# Detail

Each step applies the exact linear flow over ``Δt/2``, then the exact nonlinear
flow over ``Δt``, then the exact linear flow over ``Δt/2`` again. The linear
flow ``exp(i Δt (-coeffΔ ∇² + Ω L_z))`` is computed in Fourier space direction
by direction (no linear system is solved), and the nonlinear flow
``exp(-i (V + β |ϕ|²) Δt)`` is exact in physical space.

# No dealiasing

Same as [`NumModelSplit1`](@ref): nothing is dealiased, by design and to stay
faithful to the reference GPS time-splitting schemes. `Δt` therefore carries a
``k_max²`` constraint (see [`NumModelSplit1`](@ref) for the measured numbers and
the consequence on energy conservation). Filtering is available in
`NumModelGPRK(filter = :product | :solution | :none)`; note that filtering the
*state* is destructive — measured on the dipole initial datum at ``β = 1``,
projecting onto the 2/3 band after every step removes 33 % of the energy over 2 000
steps at ``128²``, and 30 % at ``512²/Δt = 0.01``.

# Example

```jldoctest
julia> param = GrossPitaevskiiParameters(β = 1000, Ω = 0.8, pot = PotentialZero(field));
julia> nummodel = NumModelSplit2(field, param, 0.01, 1000, 100)
Splitting Order 2
  ├───────  time step: 0.01
  └──────────── solve: number of iterations 1000, backup frequency 100
```
"""
function NumModelSplit2(f::AbstractField, param::AbstractParameters,
                        Δt::Real, niter::Integer, freqbckp::Integer)
    gf = GradientField(f; rotation=true)
    plan = Plan(f)
    writer = WriterVTK(f)
    saver = WriterSave(f)
    writers = WriterCollection([writer, saver])
    return NumModelSplit2{typeof(f),typeof(param),typeof(plan)}(f, gf, param,
                                                                Δt, niter, freqbckp,
                                                                plan,
                                                                writers)
end

function Base.show(io::IO, n::NumModelSplit2)
    return print(io,
                 "Splitting Order 2\n",
                 "  ├───────  time step: $(n.Δt)\n",
                 "  └──────────── solve: number of iterations $(n.niter), backup frequency $(n.freqbckp)")
end

function timeStep!(n::NumModelSplit2)
    solveLapRot!(n, n.Δt * 0.5)
    solveNL!(n, n.Δt)
    solveLapRot!(n, n.Δt * 0.5)
    return 2
end

function solveLapRot!(n::AbstractNumModel{F}, Δtl) where {F<:AbstractField2D}
    # field
    ϕ = n.f.ϕ
    # parameters
    coeffΔ, Ω = n.param.coeffΔ, n.param.Ω
    # temporary fields
    ϕxthat = n.plan.ϕx_hat
    ϕythat = n.plan.ϕy_hat
    # FFT x
    x, y, ξx, ξy = grid_x(n.plan)
    mul_x!(ϕxthat, n.plan, ϕ)
    @. ϕxthat = exp(im * (coeffΔ * ξx^2 - Ω * y * ξx) * Δtl) * ϕxthat
    ldiv_x!(ϕ, n.plan, ϕxthat)
    # FFT y
    x, y, ξx, ξy = grid_y(n.plan)
    mul_y!(ϕythat, n.plan, ϕ)
    @. ϕythat = exp(im * (coeffΔ * ξy^2 + Ω * x * ξy) * Δtl) * ϕythat
    ldiv_y!(ϕ, n.plan, ϕythat)
    return nothing
end

function solveLapRot!(n::AbstractNumModel{F,P},
                      Δtl) where {F<:AbstractField3D,P<:GrossPitaevskiiParameters}
    # field
    ϕ = n.f.ϕ
    # parameters
    coeffΔ, Ω = n.param.coeffΔ, n.param.Ω
    # temporary fields
    ϕxthat = n.plan.ϕx_hat
    ϕythat = n.plan.ϕy_hat
    ϕzthat = n.plan.ϕz_hat
    # FFT x
    x, y, z, ξx, ξy, ξz = grid_x(n.plan)
    mul_x!(ϕxthat, n.plan, ϕ)
    @. ϕxthat = exp(im * (coeffΔ * ξx^2 - Ω * y * ξx) * Δtl) * ϕxthat
    ldiv_x!(ϕ, n.plan, ϕxthat)
    # FFT y
    x, y, z, ξx, ξy, ξz = grid_y(n.plan)
    mul_y!(ϕythat, n.plan, ϕ)
    @. ϕythat = exp(im * (coeffΔ * ξy^2 + Ω * x * ξy) * Δtl) * ϕythat
    ldiv_y!(ϕ, n.plan, ϕythat)
    # FFT z
    x, y, z, ξx, ξy, ξz = grid_z(n.plan)
    mul_z!(ϕzthat, n.plan, ϕ)
    @. ϕzthat = exp(im * (coeffΔ * ξz^2) * Δtl) * ϕzthat
    ldiv_z!(ϕ, n.plan, ϕzthat)
    return nothing
end

function solveNL!(n::AbstractNumModel{F,P},
                  Δtl) where {F<:AbstractField,P<:GrossPitaevskiiParameters}
    # ϕ ↦ exp ( -i ( V + ∥ϕ∥² ) Δt ) ϕ
    @.n.f.ϕ = exp(-1im * (n.param.pot.V + n.param.β * abs2(n.f.ϕ)) * Δtl) * n.f.ϕ
end
