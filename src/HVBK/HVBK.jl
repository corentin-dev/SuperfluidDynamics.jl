export HVBKParameters, NumModelHVBK

"""
    HVBKParameters(; ν, νs, rb, ρn, ρs, ρt, filter)

Parameters of the linear Hall–Vinen–Bekarevich–Khalatnikov (HVBK) two-fluid
model: two incompressible velocity fields (normal `u_n` and superfluid
`u_s`), each advecting its own vorticity, coupled by the **linear mutual
friction**

```
F = -½ rb |∇×u_s| (u_n - u_s)
∂t u_n  =  ν_n Δu_n + P(u_n×∇×u_n)  + (ρs/ρt) F
∂t u_s  =  ν_s Δu_s + P(u_s×∇×u_s)  - (ρn/ρt) F
```

`P` is the Helmholtz projector, `ν_n`,`ν_s` the normal/superfluid kinematic
viscosities, `rb` the (linear) mutual-friction coefficient, `ρn`,`ρs` the
densities and `ρt` the total density (defaults to `ρn+ρs`). The friction is an
internal force: the **total momentum** `ρn u_n + ρs u_s` is conserved.

Only the linear mutual friction above is implemented: the nonlinear Iksada
friction and the transverse Anisov–Peiris–Pitaevskii term of the full HVBK
model are not.
"""
mutable struct HVBKParameters <: AbstractParameters
    "normal-fluid kinematic viscosity ν_n."
    ν::Real
    "superfluid kinematic viscosity ν_s."
    νs::Real
    "linear mutual-friction coefficient rb."
    rb::Real
    "normal-fluid density ρn."
    ρn::Real
    "superfluid density ρs."
    ρs::Real
    "total density ρt."
    ρt::Real
    "2/3-rule dealiasing."
    filter::Bool
end

function HVBKParameters(;
                         ν::Real=0.1,
                         νs::Real=0.01,
                         rb::Real=1.5,
                         ρn::Real=1.0,
                         ρs::Real=1.0,
                         ρt::Real=0.0,
                         filter::Bool=true)
    ρt = ρt == 0.0 ? ρn + ρs : ρt
    return HVBKParameters(ν, νs, rb, ρn, ρs, ρt, filter)
end

function Base.show(io::IO, p::HVBKParameters)
    return print(io,
                 "HVBK (linear two-fluid) parameters\n",
                 "  ├───────  ν: $(p.ν)  νs: $(p.νs)  rb: $(p.rb)\n",
                 "  ├───────  ρn: $(p.ρn)  ρs: $(p.ρs)  ρt: $(p.ρt)\n",
                 "  └───────  filter: $(p.filter)")
end

"""
    NumModelHVBK(fn, fs, param, Δt, niter, freqbckp; stepper="RK2")

Linear HVBK two-fluid model. `fn` is the normal-fluid velocity (2D or 3D vector
field) and `fs` the superfluid velocity; both must share the same grid and array
type. The common time step `Δt` is integrated with `stepper` ∈ "RK1" or "RK2".
Each fluid's viscous Laplacian is implicit via the exact spectral multiplier
`exp(-ν Δt |k|²)`, and both fields are Helmholtz-projected (incompressible) and
2/3-dealiased at every step.
"""
mutable struct NumModelHVBK{F,P,Plan} <: AbstractNumModel{F,P,Plan}
    "normal-fluid velocity field."
    fn::F
    "superfluid velocity field."
    fs::F
    "parameters."
    param::P
    "time step."
    Δt::Real
    "number of iterations."
    niter::Integer
    "frequency of backups."
    freqbckp::Integer
    "time scheme: \"RK1\" or \"RK2\"."
    stepper::String
    "shared FFT plan (built on the normal field)."
    plan::Plan
    "writers (on the normal field)."
    writers::AbstractWriterCollection{F}
    "writers (on the superfluid field)."
    writers_s::AbstractWriterCollection{F}
    "alias of `fn`: the field used by `solve!` for plotting / writers."
    f::F

    "normal velocity (spectral, last-pencil)."
    un_hat::Vector{PencilArray}
    "superfluid velocity (spectral, last-pencil)."
    us_hat::Vector{PencilArray}
    "increment 1, normal (spectral)."
    k1n::Vector{PencilArray}
    "increment 1, superfluid (spectral)."
    k1s::Vector{PencilArray}
    "increment 2, normal (spectral)."
    k2n::Vector{PencilArray}
    "increment 2, superfluid (spectral)."
    k2s::Vector{PencilArray}
    "RK intermediate field (spectral), normal fluid."
    stage::Vector{PencilArray}
    "RK intermediate field (spectral), superfluid."
    stage_s::Vector{PencilArray}
    "normal velocity (physical, pen_x)."
    un_phys::Vector{PencilArray}
    "superfluid velocity (physical, pen_x)."
    us_phys::Vector{PencilArray}
    "normal vorticity (physical, pen_x)."
    un_vort::Vector{PencilArray}
    "superfluid vorticity (physical, pen_x)."
    us_vort::Vector{PencilArray}
    "mutual-friction force (physical, pen_x)."
    Fr::Vector{PencilArray}
    "normal RHS u_n×ω_n + (ρs/ρt)F (physical, pen_x)."
    rhs_n::Vector{PencilArray}
    "superfluid RHS u_s×ω_s - (ρn/ρt)F (physical, pen_x)."
    rhs_s::Vector{PencilArray}
    "spectral scratch (last-pencil) for the curl."
    tmp_hat::Vector{PencilArray}
    "spectral scratch (last-pencil) for the projection divergence."
    div_hat::PencilArray
    "normal implicit viscosity factor exp(-νΔt|k|²) (real, last-pencil)."
    facn::PencilArray
    "superfluid implicit viscosity factor exp(-νsΔt|k|²) (real, last-pencil)."
    facs::PencilArray
end

function NumModelHVBK(fn::AbstractField,
                      fs::AbstractField,
                      param::HVBKParameters,
                      Δt::Real, niter::Integer, freqbckp::Integer;
                      stepper::String="RK2")
    @assert fn.g.n == fs.g.n "normal and superfluid fields must share the grid"
    plan = Plan(fn)
    FT = eltype(fn.u[1].data)
    writer = WriterVTK(fn); saver = WriterSave(fn)
    writers = WriterCollection([writer, saver])
    writer_s = WriterVTK(fs); saver_s = WriterSave(fs)
    writers_s = WriterCollection([writer_s, saver_s])
    npen = last_pencil(plan)
    bufn() = [PencilArray{FT}(undef, npen) for _ in 1:length(fn.u)]
    bufx() = [PencilArray{FT}(undef, plan.pen_x) for _ in 1:length(fn.u)]
    n = NumModelHVBK{typeof(fn),typeof(param),typeof(plan)}(
        fn, fs, param, Δt, niter, freqbckp, stepper, plan, writers, writers_s,
        fn,  # f (alias of fn, for solve! / plotting)
        bufn(),  # un_hat
        bufn(),  # us_hat
        bufn(), bufn(),  # k1n, k1s
        bufn(), bufn(),  # k2n, k2s
        bufn(),  # stage (normal intermediate)
        bufn(),  # stage_s (superfluid intermediate)
        bufx(),  # un_phys
        bufx(),  # us_phys
        bufx(),  # un_vort
        bufx(),  # us_vort
        bufx(),  # Fr
        bufx(),  # rhs_n
        bufx(),  # rhs_s
        bufn(),  # tmp_hat
        PencilArray{FT}(undef, npen),  # div_hat
        PencilArray{Float64}(undef, npen),  # facn
        PencilArray{Float64}(undef, npen))  # facs
    gridξ = spectral_grid(n.plan)
    ksq = _hvbk_ksq(gridξ)
    if ndims(gridξ) == 3
        @. n.facn = exp(-n.param.ν  * n.Δt * ksq(gridξ.x, gridξ.y, gridξ.z))
        @. n.facs = exp(-n.param.νs * n.Δt * ksq(gridξ.x, gridξ.y, gridξ.z))
    else
        @. n.facn = exp(-n.param.ν  * n.Δt * ksq(gridξ.x, gridξ.y))
        @. n.facs = exp(-n.param.νs * n.Δt * ksq(gridξ.x, gridξ.y))
    end
    # initialize the canonical spectral state from the (physical) initial fields
    mul_all!(n.un_hat, n.plan, fn.u)
    mul_all!(n.us_hat, n.plan, fs.u)
    return n
end

function Base.show(io::IO, n::NumModelHVBK)
    return print(io,
                 "HVBK linear two-fluid model ($(n.stepper))\n",
                 "  ├───────  time step: $(n.Δt)\n",
                 "  └──────────── solve: iterations $(n.niter), backup frequency $(n.freqbckp)")
end

"""
    energy(n, showEnergy=false)

Total kinetic energy of the two-fluid system
``E = ½ ρn ∫|u_n|² + ½ ρs ∫|u_s|² dV``, returned as
`(E_normal, E_superfluid, 0.0, E_total)` (slot 4 = total).
"""
function energy(n::NumModelHVBK, showEnergy=false)
    g = n.fn.g
    dV = g.Δx * g.Δy * (length(g.n) == 3 ? g.Δz : 1.0)
    En = 0.0
    Es = 0.0
    for c in 1:length(n.fn.u)
        # global reductions across MPI ranks (parent() would sum local data)
        En += sum(abs2.(n.fn.u[c]))
        Es += sum(abs2.(n.fs.u[c]))
    end
    En *= 0.5 * n.param.ρn * dV
    Es *= 0.5 * n.param.ρs * dV
    if showEnergy
        println_parallel("E_normal = $(En)   E_superfluid = $(Es)   E_total = $(En + Es)")
    end
    return En, Es, 0.0, En + Es
end

function _hvbk_ksq(gridξ)
    if ndims(gridξ) == 3
        return (x, y, z) -> x^2 + y^2 + z^2
    else
        return (x, y) -> x^2 + y^2
    end
end

"""
    _hvbk_project!(n, uhat)

Spectral Helmholtz projection of `uhat` (last-pencil), in-place (reusing
`n.div_hat` as scratch).
"""
function _hvbk_project!(n::NumModelHVBK, uhat)
    gridξ = spectral_grid(n.plan)
    d = n.div_hat
    @. d = gridξ[1] * uhat[1]
    for i in 2:ndims(gridξ)
        @. d += gridξ[i] * uhat[i]
    end
    @. d = 1im * d
    if ndims(gridξ) == 3
        for i in 1:3
            @. uhat[i] += 1im * gridξ[i] * d / ξsquared(gridξ.x, gridξ.y, gridξ.z)
        end
    else
        for i in 1:2
            @. uhat[i] += 1im * gridξ[i] * d / ξsquared2(gridξ.x, gridξ.y)
        end
    end
    return uhat
end

"""
    _hvbk_dealias!(n, uhat)

2/3-rule dealiasing of a spectral vector field (last-pencil), in-place.
"""
function _hvbk_dealias!(n::NumModelHVBK, uhat)
    gridξ = spectral_grid(n.plan)
    if ndims(gridξ) == 3
        dealias!(uhat, gridξ.x, gridξ.y, gridξ.z)
    else
        dealias2!(uhat, gridξ.x, gridξ.y)
    end
    return nothing
end

"""
    _hvbk_vorticity_phys!(n, ωphys, uhat)

Physical vorticity `∇×u` (pen_x) of a spectral velocity `uhat` (last-pencil):
spectral curl `i k×û` then IFFT.
"""
function _hvbk_vorticity_phys!(n::NumModelHVBK, ωphys, uhat)
    gridξ = spectral_grid(n.plan)
    if ndims(gridξ) == 3
        @. n.tmp_hat[1] = 1im * (gridξ.y * uhat[3] - gridξ.z * uhat[2])
        @. n.tmp_hat[2] = 1im * (gridξ.z * uhat[1] - gridξ.x * uhat[3])
        @. n.tmp_hat[3] = 1im * (gridξ.x * uhat[2] - gridξ.y * uhat[1])
    else
        @. n.tmp_hat[2] = 1im * (gridξ.x * uhat[2] - gridξ.y * uhat[1])
    end
    ldiv_all!(ωphys, n.plan, n.tmp_hat)
    return ωphys
end

"""
    rhs!(n, kn, ks)

One nonlinear evaluation of both fluids, writing the **spectral, dealiased and
projected** increments (Δt not applied) into `kn` / `ks`:

    kn = P( u_n×ω_n + (ρs/ρt) F )
    ks = P( u_s×ω_s - (ρn/ρt) F )

with the linear mutual friction `F = -½ rb |ω_s| (u_n - u_s)`. Uses the current
`n.un_hat` / `n.us_hat`; the model's scratch is overwritten.
"""
function rhs!(n::NumModelHVBK, kn, ks)
    p = n.param
    nvel = length(n.fn.u)
    is3 = (nvel == 3)
    # velocities (physical)
    ldiv_all!(n.un_phys, n.plan, n.un_hat)
    ldiv_all!(n.us_phys, n.plan, n.us_hat)
    # vorticities (physical)
    _hvbk_vorticity_phys!(n, n.un_vort, n.un_hat)
    _hvbk_vorticity_phys!(n, n.us_vort, n.us_hat)
    # |ω_s| (physical, real)
    if is3
        omag = @. sqrt(real(n.us_vort[1])^2 + real(n.us_vort[2])^2 +
                       real(n.us_vort[3])^2)
    else
        omag = @. abs(real(n.us_vort[2]))
    end
    # mutual friction F = -0.5 rb |ω_s| (u_n - u_s)  (physical, pen_x, real)
    for c in 1:nvel
        @. n.Fr[c] = -0.5 * p.rb * omag * (real(n.un_phys[c]) - real(n.us_phys[c]))
    end
    # u_n × ω_n  (physical)
    if is3
        cross!(n.rhs_n, n.un_phys, n.un_vort)
        cross!(n.rhs_s, n.us_phys, n.us_vort)
    else
        cross2!(n.rhs_n, n.un_phys, real(n.un_vort[2]))
        cross2!(n.rhs_s, n.us_phys, real(n.us_vort[2]))
    end
    # + (ρs/ρt) F  on the normal, - (ρn/ρt) F on the superfluid
    sn = p.ρs / p.ρt
    ss = p.ρn / p.ρt
    for c in 1:nvel
        @. n.rhs_n[c] += sn * n.Fr[c]
        @. n.rhs_s[c] -= ss * n.Fr[c]
    end
    # spectral, dealiased, projected
    mul_all!(kn, n.plan, n.rhs_n)
    mul_all!(ks, n.plan, n.rhs_s)
    if p.filter
        _hvbk_dealias!(n, kn)
        _hvbk_dealias!(n, ks)
    end
    _hvbk_project!(n, kn)
    _hvbk_project!(n, ks)
    return nothing
end

"""
    timeStep!(n)

Advance the HVBK model by one step `n.Δt` with `stepper` ∈ "RK1" or "RK2". The
scheme follows the Fortran `calcVelocity_forced_04` (RK2): each fluid's viscous
Laplacian is implicit via `exp(-νΔt|k|²)`, the first derivative is
viscosity-weighted and a second (corrective) derivative is taken at the
intermediate state. Both fields are Helmholtz-projected and (if `param.filter`)
2/3-dealiased after the update. The canonical fields `fn.u` / `fs.u` are
synchronised.
"""
function timeStep!(n::NumModelHVBK)
    Δt = n.Δt
    nvel = length(n.fn.u)
    facn, facs = n.facn, n.facs
    gridξ = spectral_grid(n.plan)

    if n.stepper == "RK1"
        # u_new = (u + Δt·P(NL(u)))·e^{-νΔt|k|²}, projected
        rhs!(n, n.k1n, n.k1s)
        for c in 1:nvel
            @. n.un_hat[c] = (n.un_hat[c] + Δt * n.k1n[c]) * facn
            @. n.us_hat[c] = (n.us_hat[c] + Δt * n.k1s[c]) * facs
        end
        _hvbk_project!(n, n.un_hat)
        _hvbk_project!(n, n.us_hat)
    elseif n.stepper == "RK2"
        # --- Step I: viscous-weighted Euler sub-step ---
        #   k1 = P(NL(u0)) ;  k1 ← k1·e^{-νΔt|k|²} ;  u1 = u0·e^{-νΔt|k|²} + Δt·k1
        rhs!(n, n.k1n, n.k1s)
        for c in 1:nvel
            @. n.k1n[c] *= facn
            @. n.k1s[c] *= facs
            @. n.stage[c]   = n.un_hat[c] * facn + Δt * n.k1n[c]
            @. n.stage_s[c] = n.us_hat[c] * facs + Δt * n.k1s[c]
        end
        _hvbk_project!(n, n.stage)
        _hvbk_project!(n, n.stage_s)
        # --- Step II: corrective derivative at the intermediate state ---
        # Reference (calcVelocity_forced_04):  u_new = u1 + 0.5*dt*(-k1 + k2)
        copyto!(n.un_hat, n.stage)
        copyto!(n.us_hat, n.stage_s)
        rhs!(n, n.k2n, n.k2s)                # k2 = P(NL(u1))
        for c in 1:nvel
            @. n.un_hat[c] = n.stage[c]   + 0.5 * Δt * (n.k2n[c] - n.k1n[c])
            @. n.us_hat[c] = n.stage_s[c] + 0.5 * Δt * (n.k2s[c] - n.k1s[c])
        end
        _hvbk_project!(n, n.un_hat)
        _hvbk_project!(n, n.us_hat)
    else
        throw(ArgumentError("HVBK stepper \"$(n.stepper)\" unknown (use \"RK1\" or \"RK2\")."))
    end

    # --- 2/3 dealiasing of the states (as in the Fortran step) ---
    if n.param.filter
        _hvbk_dealias!(n, n.un_hat)
        _hvbk_dealias!(n, n.us_hat)
    end
    # sync the canonical physical fields
    ldiv_all!(n.fn.u, n.plan, n.un_hat)
    ldiv_all!(n.fs.u, n.plan, n.us_hat)
    return 0
end

# Deprecated aliases (misspelling of the model name).
Base.@deprecate_binding HBVKParameters HVBKParameters
Base.@deprecate_binding NumModelHBVK NumModelHVBK
