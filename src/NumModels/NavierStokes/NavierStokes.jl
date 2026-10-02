export taylor_green!

"""
    taylor_green!(f, x, y)

Initialize `f` with a 2D Taylor-Green vortex, periodic on the box for any
box lengths:

    u = ( sin(kx x) cos(ky y),
        -(kx/ky) cos(kx x) sin(ky y) ),   k = (2π/Lx, 2π/Ly)

The y-component is scaled by `kx/ky` so that the field stays exactly
divergence-free (∂x ux + ∂y uy = (kx - kx)·cos·cos = 0) on non-cubic
boxes too.
"""
function taylor_green!(f, x, y)
    Lx, Ly = f.g.Lx, f.g.Ly
    kx, ky = 2π / Lx, 2π / Ly
    X = reshape(x, :, 1)
    Y = reshape(y, 1, :)
    @. f.ux = sin(kx * X) * cos(ky * Y)
    @. f.uy = -(kx / ky) * cos(kx * X) * sin(ky * Y)
    return nothing
end

"""
    taylor_green!(f, x, y, z)

Initialize `f` with a 3D Taylor-Green vortex, periodic on the box for any
box lengths. For a cubic box it is exactly the GPS Fortran field
(`initsolNS == "TG"` in `GPS_initial_conditions.f90`):

    u = ( sin(kx x) cos(ky y) cos(kz z),
        -cos(kx x) sin(ky y) cos(kz z),
         0 ),   k = (2π/Lx, 2π/Ly, 2π/Lz)

with the y-component scaled by `kx/ky` so that the field stays exactly
divergence-free (∂x ux + ∂y uy = (kx - kx)·cos·cos·cos = 0) on non-cubic
boxes too (the plain GPS field is only divergence-free when kx = ky).
"""
function taylor_green!(f, x, y, z)
    Lx, Ly, Lz = f.g.Lx, f.g.Ly, f.g.Lz
    kx, ky, kz = 2π / Lx, 2π / Ly, 2π / Lz
    X = reshape(x, :, 1, 1)
    Y = reshape(y, 1, :, 1)
    Z = reshape(z, 1, 1, :)
    @. f.ux = sin(kx * X) * cos(ky * Y) * cos(kz * Z)
    @. f.uy = -(kx / ky) * cos(kx * X) * sin(ky * Y) * cos(kz * Z)
    @. f.uz = 0
    return nothing
end

"""
    energy(n, showEnergy=false)

Kinetic energy of the NS velocity field of the model `n` (2D or 3D).
"""
function energy(n::AbstractNumModel{F,P},
                showEnergy=false) where {F<:AbstractField,P<:NavierStokesParameters}
    f = n.f
    E = 0.0
    for c in 1:length(f.u)
        # sum over the PencilArray: global reduction across MPI ranks (parent()
        # would sum only the local rank data).
        E += sum(real.(f.u[c] .^ 2))
    end
    E /= 2
    for d in (f.g.Δx, f.g.Δy)
        E *= d
    end
    if length(f.g.data) == 3
        E *= f.g.Δz
    end
    if (showEnergy)
        println("E = $(E)")
    end
    return 0.0, E, 0.0, E
end

"""
    NumModelRK4Imp(f, param, Δt, niter, freqbckp)

Semi-implicit Runge-Kutta 4 solver for the incompressible Navier-Stokes
equations in 2D and 3D, in the vorticity-advection form

    ∂t u = νΔu - P(u × ∇×u),   ∇⋅u = 0

where `P` is the spectral Helmholtz projector onto divergence-free fields.

The nonlinear term `-P(u × ω)` is treated explicitly (RK4), the viscosity is
treated implicitly as the exact spectral multiplier `exp(-ν Δt |k|²)` applied
at each full step. A 2/3-rule dealiasing is applied to the nonlinear term.

This is a port of `NS_model_RK4Imp` from the GPS Fortran code. The 2D and 3D
cases share the same model: only the dimension-specific kernels (curl, cross,
dealias) differ.
"""
mutable struct NumModelRK4Imp{F,P,Plan} <: AbstractNumModel{F,P,Plan}
    "field to work on."
    f::F
    "physical parameters."
    param::P
    "time step."
    Δt::Real
    "number of iterations."
    niter::Integer
    "frequency of backups."
    freqbckp::Integer
    "plan to compute derivatives."
    plan::Plan
    "writers for backups."
    writers::AbstractWriterCollection{F}

    # ---- scratch buffers (spectral, last-pencil layout) ----
    "û = FFT(u)."
    u_hat::Vector{PencilArray}
    "RK4 stage k1."
    k1::Vector{PencilArray}
    "RK4 stage k2."
    k2::Vector{PencilArray}
    "RK4 stage k3."
    k3::Vector{PencilArray}
    "RK4 stage k4."
    k4::Vector{PencilArray}
    "RK4 stage field û + Δt cᵢ kᵢ (spectral)."
    uk_hat::Vector{PencilArray}
    "ω̂ = i k × û (spectral)."
    ω_hat::Vector{PencilArray}
    "divergence diagnostic (spectral)."
    div_hat::PencilArray

    # ---- scratch buffers (physical, pen_x layout) ----
    "u at a RK4 stage (physical)."
    uk::Vector{PencilArray}
    "ω = ∇ × u (physical)."
    ω::Vector{PencilArray}
    "dU = u × ω (physical)."
    dU::Vector{PencilArray}
    "divergence diagnostic (physical)."
    div::PencilArray
end

"""
    spectral_grid(plan)

The spectral local grid of the plan's last pencil: for a 2D plan
`(gridξ.x, gridξ.y)` on `pen_y`, for a 3D plan `(gridξ.x, gridξ.y, gridξ.z)`
on `pen_z`.
"""
function spectral_grid(plan::AbstractFFTPlan)
    if plan isa PlanFFT2D
        return localgrid(plan.pen_y, (plan.ξx, plan.ξy))
    else
        return localgrid(plan.pen_z, (plan.ξx, plan.ξy, plan.ξz))
    end
end

"""
    last_pencil(plan)

The plan's last pencil: `pen_y` for 2D, `pen_z` for 3D. Spectral buffers of
the model live on this pencil.
"""
last_pencil(plan::PlanFFT2D) = plan.pen_y
last_pencil(plan::PlanFFT3D) = plan.pen_z

function NumModelRK4Imp(f::AbstractField{N,ND,FT,FFT,A},
                        param::NavierStokesParameters,
                        Δt::Real, niter::Integer, freqbckp::Integer) where {N,ND,FT,FFT,A}
    plan = Plan(f)
    writer = WriterVTK(f)
    saver = WriterSave(f)
    writers = WriterCollection([writer, saver])
    npen = last_pencil(plan)
    nvel = length(f.u)
    buf_last() = [PencilArray{FFT}(undef, npen) for _ in 1:nvel]
    buf_x() = [PencilArray{FFT}(undef, plan.pen_x) for _ in 1:nvel]
    return NumModelRK4Imp{typeof(f),typeof(param),typeof(plan)}(
        f, param, Δt, niter, freqbckp, plan, writers,
        buf_last(), buf_last(), buf_last(), buf_last(), buf_last(), buf_last(), buf_last(),
        PencilArray{FFT}(undef, npen),
        buf_x(), buf_x(), buf_x(), PencilArray{FFT}(undef, plan.pen_x))
end

function Base.show(io::IO, n::NumModelRK4Imp)
    return print(io,
                 "Navier-Stokes RK4 (semi-implicit)\n",
                 "  ├───────  time step: $(n.Δt)\n",
                 "  └──────────── solve: number of iterations $(n.niter), backup frequency $(n.freqbckp)")
end

"""
    project!(n, u_hat)

Spectral Helmholtz projection: remove from `u_hat` (spectral, last-pencil
layout) its gradient part so that the projected field is divergence free.
In-place. Dimension-agnostic (2D or 3D).
"""
function project!(n::NumModelRK4Imp, u_hat)
    gridξ = spectral_grid(n.plan)
    # div = i k · û
    d = similar(u_hat[1])
    @. d = gridξ[1] * u_hat[1]
    for i in 2:ndims(gridξ)
        @. d += gridξ[i] * u_hat[i]
    end
    @. n.div_hat = im * d
    # Helmholtz projection: u_s = u - k (k·û)/|k|². With div = i k·û we have
    # k (k·û)/|k|² = -i k div/|k|², so u_s = u + i k div/|k|².
    # Then div(u_s) = i k·u_s = div + i k·(i k div/|k|²) = div - div = 0.
    # ξsquared / ξsquared2 guard the zero mode; called inside `@.` so the
    # 1D wavenumber components are combined element-wise.
    if ndims(gridξ) == 3
        for i in 1:3
            @. u_hat[i] += im * gridξ[i] * n.div_hat /
                           ξsquared(gridξ.x, gridξ.y, gridξ.z)
        end
    else
        for i in 1:2
            @. u_hat[i] += im * gridξ[i] * n.div_hat /
                           ξsquared2(gridξ.x, gridξ.y)
        end
    end
    return u_hat
end

"""
    curl_hat!(n, ω_hat, u_hat)

Spectral vorticity `ω̂ = i k × û`, dimension-specific.
"""
function curl_hat!(n::NumModelRK4Imp, ω_hat, u_hat)
    gridξ = spectral_grid(n.plan)
    if ndims(gridξ) == 3
        # 3D: ω̂ = i k × û
        @. ω_hat[1] = im * (gridξ.y * u_hat[3] - gridξ.z * u_hat[2])
        @. ω_hat[2] = im * (gridξ.z * u_hat[1] - gridξ.x * u_hat[3])
        @. ω_hat[3] = im * (gridξ.x * u_hat[2] - gridξ.y * u_hat[1])
    else
        # 2D: the vorticity is the scalar out-of-plane component
        #   ωz = ∂x v − ∂y u  =>  ω̂z = i (kx v̂ − ky û).
        # The 2D model only allocates two component buffers, so the scalar
        # vorticity is stored in slot [2] (crossadvect! reads it from there).
        @. ω_hat[2] = im * (gridξ.x * u_hat[2] - gridξ.y * u_hat[1])
    end
    return ω_hat
end

"""
    crossadvect!(n, dU, u, ωz_phys)

Physical advective term `dU = u × ω` written in place, dimension-specific.
The other components of `ω` are left untouched (they are zero).
"""
function crossadvect!(n::NumModelRK4Imp, dU, u, ω_phys)
    if length(n.f.g.data) == 3
        cross!(dU, u, ω_phys)
    else
        # 2D: u × (ωz ẑ) = (v ωz, −u ωz); ω_phys[2] holds the scalar vorticity
        cross2!(dU, u, ω_phys[2])
    end
    return dU
end

"""
    dealias_dim!(n, k_hat)

2/3-rule dealiasing of the spectral vector field `k_hat`, dimension-specific.
"""
function dealias_dim!(n::NumModelRK4Imp, k_hat)
    gridξ = spectral_grid(n.plan)
    if length(n.f.g.data) == 3
        dealias!(k_hat, gridξ.x, gridξ.y, gridξ.z)
    else
        dealias2!(k_hat, gridξ.x, gridξ.y)
    end
    return nothing
end

"""
    rhs!(n, u_hat, k_hat)

Nonlinear (advective) part of the incompressible NS RHS, returned in
spectral form. For a divergence-free field the advection term
`-P(u × ∇×u)` equals `-P((u·∇)u)` (vector identity), and the scheme
integrates `∂t u = νΔu - P(u × ω)`. This computes `k_hat = P(u × ω)`
(i.e. the negative of the advective term, matching the Fortran
`calc_nlk_NS`); `timeStep!` then adds `+Δt·k_hat` to `u_hat`. The
conservation of kinetic energy at `ν = 0` (validated in the tests) fixes
the sign: `u · P(u × ω) = 0` for divergence-free `u`.

`u_hat` is the current spectral velocity (last-pencil layout); `k_hat` is
the output spectral RHS (last-pencil layout).
"""
function rhs!(n::NumModelRK4Imp, u_hat, k_hat)
    # u = IFFT(u_hat)
    ldiv_all!(n.uk, n.plan, u_hat)
    # ω̂ = i k × û  (spectral curl)
    curl_hat!(n, n.ω_hat, u_hat)
    # ω = IFFT(ω̂)
    ldiv_all!(n.ω, n.plan, n.ω_hat)
    # dU = u × ω
    crossadvect!(n, n.dU, n.uk, n.ω)
    # k_hat = FFT(dU)
    mul_all!(k_hat, n.plan, n.dU)
    # 2/3-rule dealiasing, then Helmholtz projection
    dealias_dim!(n, k_hat)
    project!(n, k_hat)
    return k_hat
end

"""
    divergence(n)

Divergence of the model's current spectral velocity `n.u_hat` (physical,
pen_x layout). Diagnostic only.
"""
function divergence(n::NumModelRK4Imp)
    gridξ = spectral_grid(n.plan)
    d = similar(n.u_hat[1])
    @. d = gridξ[1] * n.u_hat[1]
    for i in 2:ndims(gridξ)
        @. d += gridξ[i] * n.u_hat[i]
    end
    @. n.div_hat = im * d
    ldiv_all!(n.div, n.plan, n.div_hat)
    return n.div
end

function timeStep!(n::NumModelRK4Imp)
    plan = n.plan
    Δt = n.Δt
    nvel = length(n.f.u)
    # û = FFT(u)
    mul_all!(n.u_hat, plan, n.f.u)
    # k1 = rhs(u)
    rhs!(n, n.u_hat, n.k1)
    # k2 = rhs(u + Δt/2 k1)   (stages kept in spectral layout)
    for i in 1:nvel
        @. n.uk_hat[i] = n.u_hat[i] + Δt / 2 * n.k1[i]
    end
    rhs!(n, n.uk_hat, n.k2)
    # k3 = rhs(u + Δt/2 k2)
    for i in 1:nvel
        @. n.uk_hat[i] = n.u_hat[i] + Δt / 2 * n.k2[i]
    end
    rhs!(n, n.uk_hat, n.k3)
    # k4 = rhs(u + Δt k3)
    for i in 1:nvel
        @. n.uk_hat[i] = n.u_hat[i] + Δt * n.k3[i]
    end
    rhs!(n, n.uk_hat, n.k4)
    # combine + implicit viscosity (exact spectral multiplier)
    gridξ = spectral_grid(plan)
    νdt = n.param.ν * Δt
    if ndims(gridξ) == 3
        for i in 1:nvel
            @. n.u_hat[i] += Δt * (n.k1[i] / 6 + n.k2[i] / 3 + n.k3[i] / 3 +
                                   n.k4[i] / 6)
            @. n.u_hat[i] *= exp(-νdt * (gridξ.x^2 + gridξ.y^2 + gridξ.z^2))
        end
    else
        for i in 1:nvel
            @. n.u_hat[i] += Δt * (n.k1[i] / 6 + n.k2[i] / 3 + n.k3[i] / 3 +
                                   n.k4[i] / 6)
            @. n.u_hat[i] *= exp(-νdt * (gridξ.x^2 + gridξ.y^2))
        end
    end
    # final projection to keep div u = 0 (matches Fortran NS_model: project_uk)
    project!(n, n.u_hat)
    # back to physical
    ldiv_all!(n.f.u, plan, n.u_hat)
    return 0
end
