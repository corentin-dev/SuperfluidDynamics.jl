export NSGPParameters, NumModelNSGP,
       compute_u_adv_Fns!, calc_nlk_Lap_GP_coupled!, calc_nlk_NS_coupled!

"""
    NSGPParameters(; α, ν, β, ρn, ρs, Btab, Bptab, ξ, ε2, kreg, ηD, one_way,
                   unin, vnin, wnin)

Parameters of the coupled GP–NS two-fluid model: `α` is the (negative) GP
diffusion coefficient, `β` the GP interaction coefficient, and the friction
coefficients `B★`,`B'★` are derived from the tabulated
Hall–Vinen–Bekarevich–Khalatnikov coefficients `Btab`,`Bptab` by
`F = -1/(|α| kreg²)`, `betaNS = B★/F`, `betapNS = B'★/F`.
"""
mutable struct NSGPParameters <: AbstractParameters
    α::Real
    ν::Real
    β::Real
    ρn::Real
    ρs::Real
    Btab::Real
    Bptab::Real
    ξ::Real
    ε2::Real
    kreg::Real
    ηD::Real
    one_way::Bool
    "2/3-rule dealiasing of the nonlinear terms and of the states (Fortran
    `filter_GP`/`filter_NS`)."
    filter::Bool
    unin::Real
    vnin::Real
    wnin::Real
    F::Real
    Bstar::Real
    Bpstar::Real
    betaNS::Real
    betapNS::Real
end

function NSGPParameters(;
                         α::Real=-0.01,
                         ν::Real=0.01,
                         β::Real=1.0,
                         ρn::Real=0.5,
                         ρs::Real=0.5,
                         Btab::Real=0.4,
                         Bptab::Real=0.1,
                         ξ::Real=1.0,
                         ε2::Real=0.1,
                         kreg::Real=0.0,
                         ηD::Real=0.0,
                         one_way::Bool=false,
                         filter::Bool=true,
                         unin::Real=0.0,
                         vnin::Real=0.0,
                         wnin::Real=0.0)
    kreg = kreg == 0.0 ? 1 / ξ : kreg
    ηD = ηD == 0.0 ? 0.02 * Btab : ηD
    F = -1 / (abs(α) * kreg^2)
    ρ = ρn + ρs
    # B★, B'★ from Btab, Bptab
    denom = ρn^2 * (Btab^2 + Bptab^2) - 2 * Bptab * ρ * ρn + ρ^2
    Bstar = Btab * ρ * ρs / denom
    Bpstar = (Bptab * ρ * ρs - ρn * ρs * (Bptab^2 + Btab^2)) / denom
    betaNS = Bstar / F
    betapNS = Bpstar / F
    return NSGPParameters(α, ν, β, ρn, ρs, Btab, Bptab, ξ, ε2, kreg, ηD,
                          one_way, filter, unin, vnin, wnin, F, Bstar, Bpstar,
                          betaNS, betapNS)
end

function Base.show(io::IO, p::NSGPParameters)
    return print(io,
                 "NSGP (coupled GP–NS two-fluid) parameters\n",
                 "  ├───────  α: $(p.α)  ν: $(p.ν)  β: $(p.β)\n",
                 "  ├───────  ρn: $(p.ρn)  ρs: $(p.ρs)\n",
                 "  ├───────  Btab: $(p.Btab)  B'tab: $(p.Bptab)  ->  B★: $(p.Bstar)  B'★: $(p.Bpstar)\n",
                 "  ├───────  ξ: $(p.ξ)  kreg: $(p.kreg)  ε²: $(p.ε2)  ηD: $(p.ηD)\n",
                 "  └───────  coupling: $(p.one_way ? "one-way" : "two-way")")
end

"""
    NumModelNSGP(fgp, fns, param, Δt, niter, freqbckp; stepper="RK2Imp")

Coupled GP–NS two-fluid model. `fgp` is the scalar GP field (superfluid
wavefunction ψ), `fns` the vector NS field (normal-fluid velocity). Both
must share the same physical grid and a complex array type.

The common time step `Δt` is applied with a Runge-Kutta scheme
(`stepper` ∈ "RK1Imp", "RK2Imp", "RK4Imp"); the NS Laplacian is implicit
(exact spectral multiplier `exp(-ν Δt |k|²)`), the GP Laplacian explicit.
Each RK stage evaluates `compute_u_adv_Fns!` (regularised superfluid
velocity, counterflow, and the three coupling quantities: friction force
`F_SN`, normal-fluid advection `u_adv = v_slip^cpl`, GP coupling potential
`V_xm`) and the GP/NS increments `calc_nlk_Lap_GP_coupled!` /
`calc_nlk_NS_coupled!`.
"""
mutable struct NumModelNSGP{FG, FN, P, PlanN} <: AbstractNumModel{FN, P, PlanN}
    "GP (superfluid) scalar field."
    fgp::FG
    "NS (normal fluid) vector field."
    fns::FN
    "parameters."
    param::P
    "time step (common to GP and NS)."
    Δt::Real
    "number of iterations."
    niter::Integer
    "frequency of backups."
    freqbckp::Integer
    "time scheme: \"RK1Imp\", \"RK2Imp\" or \"RK4Imp\"."
    stepper::String
    "single plan (built on the NS field, shared for the scalar GP FFTs)."
    plan::PlanN
    "writers (on the NS field)."
    writers::AbstractWriterCollection{FN}
    "writers (on the GP field)."
    writers_gp::AbstractWriterCollection{FG}
    "alias of `fns`: the field used by `solve!` for plotting / writers."
    f::FN

    # ---- GP state + scratch ----
    "ψ̂ (spectral state, last-pencil layout)."
    phihat::PencilArray
    "ψ (physical, pen_x)."
    psi::PencilArray
    "spectral temporary (pen_x layout) for the GP gradient."
    gp_spct_x::PencilArray
    "pen_y temporary for the GP gradient."
    gp_tmp_y::PencilArray
    "pen_z temporary for the GP gradient (3D; dummy pen_y buffer in 2D)."
    gp_tmp_z::PencilArray
    "∂x ψ (physical, pen_x)."
    dpsi_x::PencilArray
    "∂y ψ (physical, pen_x)."
    dpsi_y::PencilArray
    "∂z ψ (physical, pen_x; dummy pen_x buffer in 2D)."
    dpsi_z::PencilArray
    "GP physical scratch (pen_x): |ψ|²."
    gp_phys_a::PencilArray
    "GP physical scratch (pen_x): GP temporary staging (temp_1..3)."
    gp_phys_b::PencilArray
    "GP physical scratch (pen_x): V_xm (physical, complex)."
    gp_phys_c::PencilArray
    "GP physical scratch (pen_x): temp_1 = (-V_xm-β|ψ|²)ψ+Rx+Ry, then temp_3."
    gp_phys_d::PencilArray
    "GP physical scratch (pen_x): temp_2 = (-V_xm-β)ψ+Rx+Ry."
    gp_phys_e::PencilArray
    "GP temporary (spectral, last-pencil)."
    gp_tmp_hat::PencilArray
    "GP RK increments kᵢ (spectral)."
    gp_k1::PencilArray
    gp_k2::PencilArray
    gp_k3::PencilArray
    gp_k4::PencilArray
    "GP RK stage ψ̂ (spectral, last-pencil) = ψ̂ + Δt·k.
    The stage is the spectral sum of the physical wavefunction plus the
    spectral increment, and is the input of `compute_u_adv_Fns!` and
    `calc_nlk_Lap_GP_coupled!` at the second and later stages."
    gp_stage::PencilArray
    "GP kernel spectral temporaries (last-pencil): temp_1̂, temp_2̂, temp_3̂."
    gp_t1_hat::PencilArray
    gp_t2_hat::PencilArray
    gp_t3_hat::PencilArray

    # ---- NS state + scratch ----
    "v_n (spectral state, last-pencil layout)."
    u_hat::Vector{PencilArray}
    "v_n (physical, pen_x)."
    u_phys::Vector{PencilArray}
    "NS RK increments kᵢ (spectral)."
    ns_k1::Vector{PencilArray}
    ns_k2::Vector{PencilArray}
    ns_k3::Vector{PencilArray}
    ns_k4::Vector{PencilArray}
    "v_n at an RK stage (spectral)."
    u_hat_stage::Vector{PencilArray}
    "v_s^reg (spectral / physical)."
    us_hat::Vector{PencilArray}
    us_phys::Vector{PencilArray}
    "Ω = ∇×v_s^reg (spectral / physical)."
    omega_hat::Vector{PencilArray}
    omega_phys::Vector{PencilArray}
    "w = v_n - v_s^reg (physical)."
    w_phys::Vector{PencilArray}
    "v_slip^cpl = u_adv (spectral / physical, complex)."
    uadv_hat::Vector{PencilArray}
    uadv_phys::Vector{PencilArray}
    "F_SN (spectral / physical)."
    fns_hat::Vector{PencilArray}
    fns_phys::Vector{PencilArray}
    "V_xm (spectral, scalar, last-pencil) — kept for diagnostics; the kernel
    uses the physical V_xm in `gp_phys_c`."
    vxm_hat::PencilArray
    "NS stage vorticity (spectral)."
    ns_omega_hat::Vector{PencilArray}
    "NS projection scratch (spectral, last-pencil)."
    div_hat::PencilArray
    "NS kernel physical scratch (pen_x): u×ω + F_SN/ρn."
    ns_rhs::Vector{PencilArray}
    "NS implicit viscosity factor exp(-νΔt|k|²) (real, last-pencil)."
    nsfac::PencilArray
    "GP mass-correction chemical potential μ (complex scalar)."
    μ::ComplexF64
    "GP η_D dissipation factor (complex scalar)."
    dissip::ComplexF64
    "2/3-rule dealiasing bound (GLOBAL, rank-independent), cached at construction"
    " — recomputing it per RK stage would be a device reduction + sync on the GPU."
    ξmax::Float64
end

function NumModelNSGP(fgp::AbstractField,
                      fns::AbstractField,
                      param::NSGPParameters,
                      Δt::Real, niter::Integer, freqbckp::Integer;
                      stepper::String="RK2Imp")
    @assert fgp.g.n == fns.g.n "GP and NS fields must share the same grid"
    plan = Plan(fns)
    writer = WriterVTK(fns)
    saver = WriterSave(fns)
    writers = WriterCollection([writer, saver])
    writer_gp = WriterVTK(fgp)
    saver_gp = WriterSave(fgp)
    writers_gp = WriterCollection([writer_gp, saver_gp])
    CG = eltype(fgp.ϕ.data)
    CN = eltype(fns.u[1].data)
    is3 = (length(fns.u) == 3)
    npen = last_pencil(plan)
    bufn() = [PencilArray{CN}(undef, npen) for _ in 1:length(fns.u)]
    bufx() = [PencilArray{CN}(undef, plan.pen_x) for _ in 1:length(fns.u)]
    n = NumModelNSGP{typeof(fgp),typeof(fns),typeof(param),typeof(plan)}(
        fgp, fns, param, Δt, niter, freqbckp, stepper, plan, writers, writers_gp,
        fns,  # f (alias of fns, for solve! / plotting)
        # GP state + scratch
        PencilArray{CG}(undef, npen),     # phihat
        PencilArray{CG}(undef, plan.pen_x),  # psi
        PencilArray{CG}(undef, plan.pen_x),  # gp_spct_x
        PencilArray{CG}(undef, plan.pen_y),  # gp_tmp_y
        is3 ? PencilArray{CG}(undef, plan.pen_z) :
              PencilArray{CG}(undef, plan.pen_y),  # gp_tmp_z
        PencilArray{CG}(undef, plan.pen_x),  # dpsi_x
        PencilArray{CG}(undef, plan.pen_x),  # dpsi_y
        PencilArray{CG}(undef, plan.pen_x),  # dpsi_z
        PencilArray{CG}(undef, plan.pen_x),  # gp_phys_a
        PencilArray{CG}(undef, plan.pen_x),  # gp_phys_b
        PencilArray{CG}(undef, plan.pen_x),  # gp_phys_c
        PencilArray{CG}(undef, plan.pen_x),  # gp_phys_d
        PencilArray{CG}(undef, plan.pen_x),  # gp_phys_e
        PencilArray{CG}(undef, npen),        # gp_tmp_hat
        PencilArray{CG}(undef, npen),        # gp_k1
        PencilArray{CG}(undef, npen),        # gp_k2
        PencilArray{CG}(undef, npen),        # gp_k3
        PencilArray{CG}(undef, npen),        # gp_k4
        PencilArray{CG}(undef, npen),        # gp_stage
        PencilArray{CG}(undef, npen),        # gp_t1_hat
        PencilArray{CG}(undef, npen),        # gp_t2_hat
        PencilArray{CG}(undef, npen),        # gp_t3_hat
        # NS state + scratch
        bufn(),  # u_hat
        bufx(),  # u_phys
        bufn(), bufn(), bufn(), bufn(),  # ns_k1..k4
        bufn(),  # u_hat_stage
        bufn(),  # us_hat
        bufx(),  # us_phys
        bufn(),  # omega_hat
        bufx(),  # omega_phys
        bufx(),  # w_phys
        bufn(),  # uadv_hat
        bufx(),  # uadv_phys
        bufn(),  # fns_hat
        bufx(),  # fns_phys
        PencilArray{CG}(undef, npen),      # vxm_hat (diagnostic)
        bufn(),  # ns_omega_hat
        PencilArray{CN}(undef, npen),      # div_hat
        bufx(),                            # ns_rhs
        PencilArray{Float64}(undef, npen), # nsfac
        0.0im,  # μ
        0.0im,  # dissip
        SuperfluidDynamics.ξmax_global(plan))  # ξmax (global 2/3 bound)
    # NS implicit viscosity factor exp(-νΔt|k|²) on the last-pencil spectral grid
    gridξ0 = spectral_grid(n.plan)
    ksq0 = _kxsq(gridξ0)
    if ndims(gridξ0) == 3
        @. n.nsfac = exp(-n.param.ν * n.Δt * ksq0(gridξ0.x, gridξ0.y, gridξ0.z))
    else
        @. n.nsfac = exp(-n.param.ν * n.Δt * ksq0(gridξ0.x, gridξ0.y))
    end
    # initialize the canonical spectral state from the (physical) initial fields
    mul_all!(n.phihat, n.plan, fgp.ϕ)
    mul_all!(n.u_hat, n.plan, fns.u)
    return n
end

function Base.show(io::IO, n::NumModelNSGP)
    return print(io,
                 "Navier-Stokes + Gross-Pitaevskii two-fluid model ($(n.stepper))\n",
                 "  ├───────  time step: $(n.Δt)\n",
                 "  └──────────── solve: iterations $(n.niter), backup frequency $(n.freqbckp)")
end

"""
    gppsi(n)

The physical GP wavefunction of the model `n` (iFFT of the spectral state),
returned as a PencilArray. Diagnostic convenience.
"""
function gppsi(n::NumModelNSGP)
    ldiv_all!(n.psi, n.plan, n.phihat)
    return n.psi
end

"""
    energy(n, showEnergy=false)

Kinetic energy of the normal-fluid velocity,
`E = ½ ρn ∫ |v_n|² dA`, returned in slot 4 (consistent with the other
models; the GP energy is not returned).
"""
function energy(n::NumModelNSGP, showEnergy=false)
    f = n.fns
    E = 0.0
    for c in 1:length(f.u)
        # global reduction across MPI ranks (parent() would sum local data)
        E += sum(real.(f.u[c] .^ 2))
    end
    E *= 0.5 * n.param.ρn * f.g.Δx * f.g.Δy
    if length(f.g.n) == 3
        E *= f.g.Δz
    end
    showEnergy && println("E_NS = $(E)")
    return 0.0, 0.0, 0.0, E
end

# ---------------------------------------------------------------------
# shared helpers
# ---------------------------------------------------------------------

# |k|² closure on the last-pencil spectral grid (scalar broadcast, @. safe).
function _kxsq(gridξ)
    if ndims(gridξ) == 3
        return (x, y, z) -> x^2 + y^2 + z^2
    else
        return (x, y) -> x^2 + y^2
    end
end

"""
    _dealias_scalar!(s_hat, gridξ, ξmax)

2/3-rule dealiasing of a scalar spectral field `s_hat` (last-pencil layout),
with the same threshold as `dealias!`/`dealias2!` (GPS `filter_dealiasing`).
In-place.
"""
function _dealias_scalar!(s_hat, gridξ, ξmax)
    if ndims(gridξ) == 3
        @. s_hat *= (gridξ.x^2 + gridξ.y^2 + gridξ.z^2) < ξmax
    else
        @. s_hat *= (gridξ.x^2 + gridξ.y^2) < ξmax
    end
    return nothing
end

"""
    _project_spectral!(n, uhat)

Spectral Helmholtz projection of the vector field `uhat` (last-pencil),
in-place (same as `project!` of the NS model, on the shared plan, using
`n.div_hat` as scratch).
"""
function _project_spectral!(n::NumModelNSGP, uhat)
    gridξ = spectral_grid(n.plan)
    d = n.div_hat
    @. d = gridξ[1] * uhat[1]
    for i in 2:ndims(gridξ)
        @. d += gridξ[i] * uhat[i]
    end
    @. n.div_hat = im * d
    if ndims(gridξ) == 3
        for i in 1:3
            @. uhat[i] += im * gridξ[i] * n.div_hat /
                           ξsquared(gridξ.x, gridξ.y, gridξ.z)
        end
    else
        for i in 1:2
            @. uhat[i] += im * gridξ[i] * n.div_hat /
                           ξsquared2(gridξ.x, gridξ.y)
        end
    end
    return uhat
end

# Regularisation filter applied to a spectral vector field (last-pencil
# layout): `exp(-|k|²/kreg²)` masked by the 2/3 rule (zero above threshold).
function _gauss_smooth!(n::NumModelNSGP, uhat)
    gridξ = spectral_grid(n.plan)
    kreg2 = n.param.kreg^2
    ksq = _kxsq(gridξ)
    ξmax = n.ξmax
    if ndims(gridξ) == 3
        wvn = @. ksq(gridξ.x, gridξ.y, gridξ.z)
        filt = @. ifelse(wvn < ξmax, exp(-wvn / kreg2), 0.0)
    else
        wvn = @. ksq(gridξ.x, gridξ.y)
        filt = @. ifelse(wvn < ξmax, exp(-wvn / kreg2), 0.0)
    end
    for i in 1:length(uhat)
        @. uhat[i] *= filt
    end
    return nothing
end

"""
    _grad_psi!(n)

Physical gradient of the model's GP wavefunction `n.psi` (pen_x), stored in
`n.dpsi_x/y/z`. Manual spectral derivatives with the shared plan (separate
in/out buffers, as in `GradientField`). 2D and 3D.
"""
function _grad_psi!(n::NumModelNSGP)
    plan = n.plan
    is3 = (length(n.fns.u) == 3)
    args = is3 ? (plan.ξx, plan.ξy, plan.ξz) : (plan.ξx, plan.ξy)
    gpx = localgrid(plan.pen_x, args)
    # x
    mul_x!(n.gp_spct_x, plan, n.psi)
    @. n.gp_spct_x = 1im * gpx.x * n.gp_spct_x
    ldiv_x!(n.dpsi_x, plan, n.gp_spct_x)
    # y
    gpy = localgrid(plan.pen_y, args)
    mul_y!(n.gp_tmp_y, plan, n.psi)
    @. n.gp_tmp_y = 1im * gpy.y * n.gp_tmp_y
    ldiv_y!(n.dpsi_y, plan, n.gp_tmp_y)
    if is3
        gpz = localgrid(plan.pen_z, args)
        mul_z!(n.gp_tmp_z, plan, n.psi)
        @. n.gp_tmp_z = 1im * gpz.z * n.gp_tmp_z
        ldiv_z!(n.dpsi_z, plan, n.gp_tmp_z)
    end
    return nothing
end

# ---------------------------------------------------------------------
# core coupling
# ---------------------------------------------------------------------

"""
    compute_u_adv_Fns!(n, phihat, uhat)

From the spectral GP wavefunction `phihat` and the spectral normal velocity
`uhat`, compute and store in the model:

- `us_phys` / `us_hat`: the regularised superfluid velocity
  ``v_s^{reg} = (1+ε²) F⁻¹[e^{-k²/kreg²} F(u_s)]``,
  ``u_s = |α| Im(∇ψ·ψ̄)/(|ψ|²+ε²)`` (GPS convention),
- `omega_phys` / `omega_hat`: the regularised vorticity ``Ω = ∇×v_s^{reg}``,
- `w_phys`: the counterflow ``w = v_n - v_s^{reg}`` (or
  ``v_n^{in} - v_s^{reg}`` in one-way coupling),
- `uadv_phys` / `uadv_hat`: the complex advection velocity
  ``v_{slip}^{cpl} = (c_U + i c_V|Ω|) w_p``, ``c_U = U★``, ``c_V = F V★``),
- `fns_phys` / `fns_hat`: the friction force ``F_SN = ρ_s Ω×v_{slip}``
  (dealiased and Helmholtz-projected),
- `gp_phys_c`: the GP coupling potential (physical, complex, pen_x)
  ``V_xm = Re(Σ_c u_adv,c²)/(4|α|) - β - ½ i (∇·u_adv)``;
  `vxm_hat` holds its 2/3-dealiased spectral copy (diagnostic).

`phihat` / `uhat` are not modified.
"""
function compute_u_adv_Fns!(n::NumModelNSGP, phihat, uhat)
    p = n.param
    plan = n.plan
    nvel = length(n.fns.u)
    is3 = (nvel == 3)

    # ---------- 1. ψ = IFFT(ψ̂); u_s from ψ ----------
    ldiv_all!(n.psi, plan, phihat)
    _grad_psi!(n)
    # u_s = -|α| i (∇ψ ψ̄ − conj) (1+ε²)/(|ψ|²+ε²) = 2|α| Im(ψ̄∇ψ) (1+ε²)/(|ψ|²+ε²)
    # (Fortran compute_us, smallNumber = ε²)
    @. n.us_phys[1] = 2 * abs(p.α) * imag(conj(n.psi) * n.dpsi_x) /
                      (abs2(n.psi) + p.ε2) * (1 + p.ε2)
    @. n.us_phys[2] = 2 * abs(p.α) * imag(conj(n.psi) * n.dpsi_y) /
                      (abs2(n.psi) + p.ε2) * (1 + p.ε2)
    if is3
        @. n.us_phys[3] = 2 * abs(p.α) * imag(conj(n.psi) * n.dpsi_z) /
                          (abs2(n.psi) + p.ε2) * (1 + p.ε2)
    end

    # ---------- 2. u_s^reg = (1+ε²) F⁻¹[e^{-k²/kreg²} F(u_s)]; Ω = ∇×(·) ----------
    mul_all!(n.us_hat, plan, n.us_phys)
    _gauss_smooth!(n, n.us_hat)
    gridξ = spectral_grid(plan)
    ξmax = n.ξmax
    if is3
        # Ω = ∇×u_s^reg (spectral); same operator as the NS vorticity
        @. n.omega_hat[1] = 1im * (gridξ.y * n.us_hat[3] - gridξ.z * n.us_hat[2])
        @. n.omega_hat[2] = 1im * (gridξ.z * n.us_hat[1] - gridξ.x * n.us_hat[3])
        @. n.omega_hat[3] = 1im * (gridξ.x * n.us_hat[2] - gridξ.y * n.us_hat[1])
    else
        # 2D: Ω = (0, 0, Ωz),  Ωz = i (kx v̂ - ky û)
        @. n.omega_hat[2] = 1im * (gridξ.x * n.us_hat[2] - gridξ.y * n.us_hat[1])
    end
    # project u_s^reg (divergence-free superflow), then back to physical
    _project_spectral!(n, n.us_hat)
    ldiv_all!(n.us_phys, plan, n.us_hat)
    ldiv_all!(n.omega_phys, plan, n.omega_hat)

    # ---------- 3. counterflow w = v_n - v_s^reg ----------
    ldiv_all!(n.u_phys, plan, uhat)
    for c in 1:nvel
        if p.one_way
            un_c = c == 1 ? p.unin : (c == 2 ? p.vnin : p.wnin)
            @. n.w_phys[c] = un_c - real(n.us_phys[c])
        else
            @. n.w_phys[c] = real(n.u_phys[c]) - real(n.us_phys[c])
        end
    end

    # ---------- 4. c_U, c_V, w_p, u_adv, F_SN (physical) ----------
    CU = similar(parent(n.w_phys[1]), Float64)
    CV = similar(parent(n.w_phys[1]), Float64)
    if is3
        o1 = real(parent(n.omega_phys[1])); o2 = real(parent(n.omega_phys[2]))
        o3 = real(parent(n.omega_phys[3]))
        w1 = real(parent(n.w_phys[1])); w2 = real(parent(n.w_phys[2]))
        w3 = real(parent(n.w_phys[3]))
        Om2 = o1 .^ 2 .+ o2 .^ 2 .+ o3 .^ 2
        cUV = @. p.betaNS^2 * p.F^2 * Om2 * p.ρn^2 +
              (p.ρs + p.betapNS * p.F * p.ρn)^2
        @. CU = p.F * p.ρn * (p.betaNS^2 * p.F^3 * Om2 * p.ρn +
                              p.betapNS^2 * p.F * p.ρn + p.betapNS * p.ρs) / cUV
        @. CV = p.betaNS * p.F^2 * p.ρn * p.ρs / cUV
        wO = w1 .* o1 .+ w2 .* o2 .+ w3 .* o3
        # projection factor w·Ω/|Ω|², guarded where |Ω|²≈0 (limit: w_p → w)
        fac = ifelse.(Om2 .> 1e-30, wO ./ Om2, 0.0)
        wp1 = @. w1 - fac * o1
        wp2 = @. w2 - fac * o2
        wp3 = @. w3 - fac * o3
        amp = CU .+ 1im * (CV .* sqrt.(Om2))
        @. n.uadv_phys[1] = amp * wp1
        @. n.uadv_phys[2] = amp * wp2
        @. n.uadv_phys[3] = amp * wp3
        # F_SN = ρs v × Ω  with  v = c_U w_p + c_V (Ω×w)   [c_V already includes F]
        # (same convention as the 2D branch below, which this reduces to for
        # Ω ∥ ẑ). With c_V ≤ 0 the force opposes the counterflow:
        # F_SN·w = ρs c_V |Ω|² |w_p|² ≤ 0, i.e. it is a friction on the normal fluid.
        v1 = @. CU * wp1 + CV * (o2 * w3 - o3 * w2)
        v2 = @. CU * wp2 + CV * (o3 * w1 - o1 * w3)
        v3 = @. CU * wp3 + CV * (o1 * w2 - o2 * w1)
        @. n.fns_phys[1] = p.ρs * (v2 * o3 - v3 * o2)
        @. n.fns_phys[2] = p.ρs * (v3 * o1 - v1 * o3)
        @. n.fns_phys[3] = p.ρs * (v1 * o2 - v2 * o1)
    else
        # 2D: Ω = (0, 0, Ωz) ∥ ẑ, w ⊥ ẑ  =>  w_p = w
        vz = real(parent(n.omega_phys[2]))
        w1 = real(parent(n.w_phys[1])); w2 = real(parent(n.w_phys[2]))
        Om2 = vz .^ 2
        cUV = @. p.betaNS^2 * p.F^2 * Om2 * p.ρn^2 +
              (p.ρs + p.betapNS * p.F * p.ρn)^2
        @. CU = p.F * p.ρn * (p.betaNS^2 * p.F^3 * Om2 * p.ρn +
                              p.betapNS^2 * p.F * p.ρn + p.betapNS * p.ρs) / cUV
        @. CV = p.betaNS * p.F^2 * p.ρn * p.ρs / cUV
        # u_adv = (c_U + i c_V|Ωz|) w
        amp = CU .+ 1im * (CV .* abs.(vz))
        @. n.uadv_phys[1] = amp * w1
        @. n.uadv_phys[2] = amp * w2
        # F_SN = ρs v × Ω  with  v = c_U w + c_V (Ω×w)  (Ω = Ωz ẑ, w ⊥ ẑ):
        #   v = (c_U w1 - c_V w2 Ωz, c_U w2 + c_V w1 Ωz),  v×Ω = (v2 Ωz, -v1 Ωz).
        # F_SN·w = ρs c_V Ωz² |w|² ≤ 0 for c_V ≤ 0: friction on the normal fluid.
        usn1 = @. p.ρs * (CU * w1 - CV * w2 * vz)
        usn2 = @. p.ρs * (CU * w2 + CV * w1 * vz)
        @. n.fns_phys[1] = vz * usn2
        @. n.fns_phys[2] = -vz * usn1
    end

    # ---------- 5. spectral forms: FFT -> 2/3 dealias -> (project F_SN) -> IFFT ----------
    # (physical u_adv / fns_phys after this block are the dealiased — and, for
    # F_SN, projected — fields used in V_xm, the GP kernel and the NS kernel)
    mul_all!(n.uadv_hat, plan, n.uadv_phys)
    mul_all!(n.fns_hat, plan, n.fns_phys)
    if is3
        dealias!(n.uadv_hat, gridξ.x, gridξ.y, gridξ.z, ξmax)
        dealias!(n.fns_hat, gridξ.x, gridξ.y, gridξ.z, ξmax)
    else
        dealias2!(n.uadv_hat, gridξ.x, gridξ.y, ξmax)
        dealias2!(n.fns_hat, gridξ.x, gridξ.y, ξmax)
    end
    _project_spectral!(n, n.fns_hat)
    ldiv_all!(n.uadv_phys, plan, n.uadv_hat)
    ldiv_all!(n.fns_phys, plan, n.fns_hat)

    # ---------- 6. V_xm = Re(Σ u_adv_c²)/(4|α|) - β - ½ i (∇·u_adv) (physical) ----------
    # Re(Σ u_adv_c²) (physical), then 2/3-dealiased (FFT->mask->IFFT), as in Fortran
    @. n.gp_phys_a = n.uadv_phys[1]^2
    for c in 2:nvel
        @. n.gp_phys_a .+= n.uadv_phys[c]^2
    end
    @. n.gp_phys_a = real(n.gp_phys_a)
    mul_all!(n.vxm_hat, plan, n.gp_phys_a)
    _dealias_scalar!(n.vxm_hat, gridξ, ξmax)
    ldiv_all!(n.gp_phys_a, plan, n.vxm_hat)
    # i (∇·u_adv): spectral divergence -> physical (complex, pen_x), reusing div_hat
    @. n.div_hat = gridξ[1] * n.uadv_hat[1]
    for i in 2:ndims(gridξ)
        @. n.div_hat += gridξ[i] * n.uadv_hat[i]
    end
    @. n.div_hat *= 1im
    ldiv_all!(n.gp_phys_b, plan, n.div_hat)
    # V_xm (physical, complex, pen_x) — kept in gp_phys_c for the GP kernel
    @. n.gp_phys_c = n.gp_phys_a / (4 * abs(p.α)) - p.β - 0.5im * n.gp_phys_b
    return nothing
end

# ---------------------------------------------------------------------
# GP increment
# ---------------------------------------------------------------------

"""
    calc_nlk_Lap_GP_coupled!(n, phihat, out)

GP (superfluid) increment of the coupled model. `phihat` is the (spectral)
wavefunction of the current RK stage; `out` (spectral, last-pencil) receives
`Δψ̂` **including the time step `Δt`** (unlike the NS increment). The coupling
quantities (`uadv_phys`, `fns_phys`, `gp_phys_c` = V_xm) must have been set by a
preceding `compute_u_adv_Fns!`.

With `η_D = 0` the increment carries the mass-correction `μ`; with `η_D ≠ 0` it
also carries the dissipation factor `dissip`, the linear term being multiplied by
`(1 - i η_D)`. Both are Parseval ratios computed as MPI-global sums. `phihat` is
not modified.
"""
function calc_nlk_Lap_GP_coupled!(n::NumModelNSGP, phihat, out)
    p = n.param
    plan = n.plan
    is3 = (length(n.fns.u) == 3)
    gridξ = spectral_grid(plan)
    ξmax = n.ξmax
    args = is3 ? (plan.ξx, plan.ξy, plan.ξz) : (plan.ξx, plan.ξy)
    # |k|² on the last-pencil spectral grid
    if is3
        gpz = localgrid(plan.pen_z, args)
        ksq = @. gpz.x^2 + gpz.y^2 + gpz.z^2
    else
        gpy = localgrid(plan.pen_y, args)
        ksq = @. gpy.x^2 + gpy.y^2
    end
    # ψ = IFFT(ψ̂);  ∇ψ
    ldiv_all!(n.psi, plan, phihat)
    _grad_psi!(n)
    # Rx + Ry = i (u_adv · ∇ψ)  (physical, pen_x, complex)
    @. n.gp_phys_b = 1im * (n.uadv_phys[1] * n.dpsi_x + n.uadv_phys[2] * n.dpsi_y)
    if is3
        @. n.gp_phys_b += 1im * n.uadv_phys[3] * n.dpsi_z
    end
    # |ψ|² (physical), 2/3-dealiased
    @. n.gp_phys_a = abs2(n.psi)
    if p.filter
        mul_all!(n.gp_tmp_hat, plan, n.gp_phys_a)
        _dealias_scalar!(n.gp_tmp_hat, gridξ, ξmax)
        ldiv_all!(n.gp_phys_a, plan, n.gp_tmp_hat)
    end
    rho2 = n.gp_phys_a
    # temp_3 = (β - β|ψ|²)ψ ; temp_1 = (-V_xm - β|ψ|²)ψ + Rx+Ry ; temp_2 = (-V_xm - β)ψ + Rx+Ry
    @. n.gp_phys_e = (p.β - p.β * rho2) * n.psi
    @. n.gp_phys_d = (-n.gp_phys_c - p.β * rho2) * n.psi + n.gp_phys_b
    @. n.gp_phys_b = (-n.gp_phys_c - p.β) * n.psi + n.gp_phys_b
    # spectral copies
    mul_all!(n.gp_t1_hat, plan, n.gp_phys_d)   # temp_1̂
    mul_all!(n.gp_t2_hat, plan, n.gp_phys_b)   # temp_2̂
    mul_all!(n.gp_t3_hat, plan, n.gp_phys_e)   # temp_3̂
    if p.filter
        _dealias_scalar!(n.gp_t1_hat, gridξ, ξmax)
        _dealias_scalar!(n.gp_t2_hat, gridξ, ξmax)
        _dealias_scalar!(n.gp_t3_hat, gridξ, ξmax)
    end
    # mass correction μ = -Σ temp_2̂ ψ̄̂ / Σ|ψ̂|²   (spectral, MPI-global sums)
    cormass2 = mapreduce(x -> real(x * conj(x)), +, phihat)
    cormass1 = mapreduce((a, b) -> a * conj(b), +, n.gp_t2_hat, phihat)
    n.μ = -cormass1 / cormass2
    if p.ηD != 0
        # temp_3_spectral = α|k|²ψ̂ + temp_3̂ ;  dissip = η_D Σ temp_3_spectral ψ̄̂ / Σ|ψ̂|²
        @. n.gp_t3_hat += p.α * ksq * phihat
        cormass3 = mapreduce((a, b) -> a * conj(b), +, n.gp_t3_hat, phihat)
        n.dissip = p.ηD * cormass3 / cormass2
        @. out = n.Δt * 1im * (n.μ * phihat +
                               (1 - 1im * p.ηD) * n.gp_t3_hat +
                               1im * n.dissip * phihat + n.gp_t2_hat)
    else
        @. out = n.Δt * 1im * ((n.μ + p.α * ksq) * phihat + n.gp_t1_hat)
    end
    if p.filter
        _dealias_scalar!(out, gridξ, ξmax)
    end
    return out
end

# ---------------------------------------------------------------------
# NS increment
# ---------------------------------------------------------------------

"""
    calc_nlk_NS_coupled!(n, uhat, out)

Normal-fluid (NS) increment of the coupled model. `uhat` is the (spectral) normal velocity of the current RK stage; `out`
(spectral, last-pencil) receives ``P(u×ω + F_SN/ρ_n)`` (dealiased and
Helmholtz-projected). **The time step is NOT applied here** — it is applied at
the update together with the implicit viscosity factor. The friction force
`fns_phys` must have been set by a preceding `compute_u_adv_Fns!`. `uhat` is
not modified.
"""
function calc_nlk_NS_coupled!(n::NumModelNSGP, uhat, out)
    plan = n.plan
    is3 = (length(n.fns.u) == 3)
    gridξ = spectral_grid(plan)
    ξmax = n.ξmax
    # u = IFFT(û)  (physical)
    ldiv_all!(n.u_phys, plan, uhat)
    # ω̂ = i k × û  (spectral), then ω = IFFT(ω̂)  (physical)
    if is3
        @. n.omega_hat[1] = 1im * (gridξ.y * uhat[3] - gridξ.z * uhat[2])
        @. n.omega_hat[2] = 1im * (gridξ.z * uhat[1] - gridξ.x * uhat[3])
        @. n.omega_hat[3] = 1im * (gridξ.x * uhat[2] - gridξ.y * uhat[1])
    else
        @. n.omega_hat[2] = 1im * (gridξ.x * uhat[2] - gridξ.y * uhat[1])
    end
    ldiv_all!(n.omega_phys, plan, n.omega_hat)
    # rhs = u × ω  (physical, pen_x)
    if is3
        cross!(n.ns_rhs, n.u_phys, n.omega_phys)
    else
        cross2!(n.ns_rhs, n.u_phys, n.omega_phys[2])
    end
    # + F_SN / ρ_n
    for c in 1:length(n.fns.u)
        @. n.ns_rhs[c] += real(n.fns_phys[c]) / n.param.ρn
    end
    # out = FFT(rhs);  2/3 dealias;  Helmholtz projection
    mul_all!(out, plan, n.ns_rhs)
    if n.param.filter
        if is3
            dealias!(out, gridξ.x, gridξ.y, gridξ.z, ξmax)
        else
            dealias2!(out, gridξ.x, gridξ.y, ξmax)
        end
    end
    _project_spectral!(n, out)
    return out
end

# ---------------------------------------------------------------------
# time stepping
# ---------------------------------------------------------------------

"""
    timeStep!(n)

Advance the coupled GP–NS model by one time step `n.Δt` with the Runge-Kutta
scheme `n.stepper` ("RK1Imp", "RK2Imp" or "RK4Imp"). Each RK stage evaluates
`compute_u_adv_Fns!` (coupling), `calc_nlk_Lap_GP_coupled!` (GP increment, with
Δt) and `calc_nlk_NS_coupled!` (NS increment, without Δt). The NS Laplacian is
implicit via the exact multiplier `exp(-νΔt|k|²)`, the GP Laplacian explicit.
After the update the normal velocity is Helmholtz-projected and, if
`param.filter`, the 2/3 rule is applied to both states. The canonical fields
`fgp.ϕ` and `fns.u` are synchronised.
"""
function timeStep!(n::NumModelNSGP)
    p = n.param
    plan = n.plan
    Δt = n.Δt
    nvel = length(n.fns.u)
    nsfac = n.nsfac
    gridξ = spectral_grid(plan)
    ξmax = n.ξmax

    # ---- Stage 1: k1 ----
    compute_u_adv_Fns!(n, n.phihat, n.u_hat)
    calc_nlk_Lap_GP_coupled!(n, n.phihat, n.gp_k1)
    calc_nlk_NS_coupled!(n, n.u_hat, n.ns_k1)

    if n.stepper == "RK1Imp"
        for c in 1:nvel
            @. n.u_hat[c] = (n.u_hat[c] + Δt * n.ns_k1[c]) * nsfac
        end
        @. n.phihat += n.gp_k1
    elseif n.stepper == "RK2Imp"
        # Stage 2 (Euler predictor): stage = state + Δt·k1
        for c in 1:nvel
            @. n.u_hat_stage[c] = n.u_hat[c] + Δt * n.ns_k1[c]
        end
        @. n.gp_stage = n.phihat + n.gp_k1
        compute_u_adv_Fns!(n, n.gp_stage, n.u_hat_stage)
        calc_nlk_Lap_GP_coupled!(n, n.gp_stage, n.gp_k2)
        calc_nlk_NS_coupled!(n, n.u_hat_stage, n.ns_k2)
        for c in 1:nvel
            @. n.u_hat[c] = (n.u_hat[c] + 0.5 * Δt * (n.ns_k1[c] + n.ns_k2[c])) * nsfac
        end
        @. n.phihat += 0.5 * (n.gp_k1 + n.gp_k2)
    elseif n.stepper == "RK4Imp"
        # Stage 2: state + Δt/2·k1
        for c in 1:nvel
            @. n.u_hat_stage[c] = n.u_hat[c] + 0.5 * Δt * n.ns_k1[c]
        end
        @. n.gp_stage = n.phihat + 0.5 * n.gp_k1
        compute_u_adv_Fns!(n, n.gp_stage, n.u_hat_stage)
        calc_nlk_Lap_GP_coupled!(n, n.gp_stage, n.gp_k2)
        calc_nlk_NS_coupled!(n, n.u_hat_stage, n.ns_k2)
        # Stage 3: state + Δt/2·k2
        for c in 1:nvel
            @. n.u_hat_stage[c] = n.u_hat[c] + 0.5 * Δt * n.ns_k2[c]
        end
        @. n.gp_stage = n.phihat + 0.5 * n.gp_k2
        compute_u_adv_Fns!(n, n.gp_stage, n.u_hat_stage)
        calc_nlk_Lap_GP_coupled!(n, n.gp_stage, n.gp_k3)
        calc_nlk_NS_coupled!(n, n.u_hat_stage, n.ns_k3)
        # Stage 4: state + Δt·k3
        for c in 1:nvel
            @. n.u_hat_stage[c] = n.u_hat[c] + Δt * n.ns_k3[c]
        end
        @. n.gp_stage = n.phihat + n.gp_k3
        compute_u_adv_Fns!(n, n.gp_stage, n.u_hat_stage)
        calc_nlk_Lap_GP_coupled!(n, n.gp_stage, n.gp_k4)
        calc_nlk_NS_coupled!(n, n.u_hat_stage, n.ns_k4)
        for c in 1:nvel
            @. n.u_hat[c] = (n.u_hat[c] + Δt * (n.ns_k1[c] / 6 + n.ns_k2[c] / 3 +
                                               n.ns_k3[c] / 3 + n.ns_k4[c] / 6)) * nsfac
        end
        @. n.phihat += (n.gp_k1 / 6 + n.gp_k2 / 3 + n.gp_k3 / 3 + n.gp_k4 / 6)
    else
        throw(ArgumentError("NSGP stepper \"$(n.stepper)\" unknown (use \"RK1Imp\", " *
                            "\"RK2Imp\" or \"RK4Imp\")."))
    end

    # ---- final projection (div u = 0) + 2/3 dealiasing of the states ----
    _project_spectral!(n, n.u_hat)
    if p.filter
        for c in 1:nvel
            _dealias_scalar!(n.u_hat[c], gridξ, ξmax)
        end
        _dealias_scalar!(n.phihat, gridξ, ξmax)
    end
    # sync the canonical physical fields (for plotting / writers / next step)
    ldiv_all!(n.fns.u, plan, n.u_hat)
    ldiv_all!(n.fgp.ϕ, plan, n.phihat)
    return 0
end
