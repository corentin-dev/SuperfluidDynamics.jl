# CUDA (optional at load time): the GPU compact backend is a Thomas kernel
# compiled by CUDA.jl. `using CUDA` re-exports the CUDACore pieces the kernel
# file needs (@cuda, blockIdx/threadIdx/blockDim, CuArray). When CUDA cannot
# be loaded (no driver, or a build without it), none of it is defined and
# CompactPlan does not offer the GPU kernel backend.
try
    using CUDA
catch
    CUDA = nothing
end
if CUDA !== nothing
    include("compact_gpu.jl")
end

"Abstract supertype for plan type."
abstract type PlanType end
"FFT plan"
struct FFTPlan <: PlanType end
"Finite difference plan"
struct FiniteDifferencePlan <: PlanType end
"Compact finite difference plan (6th-order periodic compact scheme).

Backend selection (only relevant for GPU / non-Array grids):
- `:auto` (default): Thomas on CPU, CUDA Thomas kernel on GPU;
- `:thomas`: force the Thomas line-solve (CPU host loop or CUDA kernel);
- `:spectral`: evaluate the operator as a Fourier multiplier (GPU, requires a
  complex field; the fast periodic-only option).

Boundary conditions (`bcs`): a tuple of one entry per axis giving the
condition on that axis — `0` (or `:periodic`) for the periodic compact
scheme, `1` (or `:dirichlet`) for the homogeneous-Dirichlet non-periodic
scheme (GPS `cdl==2` / Xcompact3d `ncl==2`), `2` (or `:neumann`) for the
homogeneous-Neumann scheme (even-mirror closure, Xcompact3d `ncl==1`).
Defaults to all periodic. Axes may be mixed (e.g. `bcs=(1, 0)` bounds `x`
and keeps `y` periodic)."
struct CompactPlan{B,BCS} <: PlanType
    "GPU backend: `:auto`, `:thomas` (CUDA kernel) or `:spectral` (Fourier multiplier)."
    backend::B
    "per-axis boundary condition: `0`/`:periodic`, `1`/`:dirichlet`, `2`/`:neumann`."
    bcs::BCS
end
function CompactPlan(; backend::Symbol=:auto, bcs = nothing)
    b = bcs === nothing ? () : bcs
    return CompactPlan{typeof(backend), typeof(b)}(backend, b)
end

# Normalise a single BC entry to an integer (0 periodic, 1 Dirichlet, 2 Neumann).
function _bc_int(bc)
    if bc == 0 || bc === :periodic
        return 0
    elseif bc == 1 || bc === :dirichlet
        return 1
    elseif bc == 2 || bc === :neumann
        return 2
    else
        error("CompactPlan: unsupported boundary condition $(repr(bc)) " *
              "(use 0/:periodic, 1/:dirichlet or 2/:neumann).")
    end
end

export Plan, FFTPlan, FiniteDifferencePlan, CompactPlan

"Abstract supertype for plans."
abstract type AbstractPlan{N} end
"Abstract supertype for FFT plans."
abstract type AbstractFFTPlan{N} end
"Abstract supertype for Finite Difference plans."
abstract type AbstractFDPlan{N} end
"Abstract supertype for Compact Finite Difference plans."
abstract type AbstractCompactPlan{N} end

"Abstract supertype for per-axis compact multipliers (periodic or bounded)."
abstract type AbstractCompactAxis{R} end

"""
$(TYPEDEF)

Precomputed multipliers for a compact finite-difference derivative on one axis.

Built by [`compact_setup`](@ref). The cyclic tridiagonal solve (Thomas
multipliers plus the right-hand-side-independent cyclic correction) is
precomputed here, so each evaluation only performs the stencil plus two sweeps.

$(TYPEDFIELDS)
"""
struct CompactAxis{R} <: AbstractCompactAxis{R}
    "number of grid points along the axis."
    n::Int
    "derivative order (1 or 2)."
    order::Int
    "compact coefficient: 1/3 (1st order) or 2/11 (2nd order)."
    alpha::R
    "stencil coefficient a."
    a::R
    "stencil coefficient b."
    b::R
    "Thomas forward multipliers."
    s::Vector{R}
    "reciprocals of the Thomas pivots."
    w::Vector{R}
    "super-diagonal of the (unchanged) tridiagonal."
    f::Vector{R}
    "precomputed cyclic correction vector."
    t::Vector{R}
    "cyclic correction denominator."
    denom::R
end

"""
$(TYPEDEF)

Precomputed multipliers for the **non-periodic (homogeneous Dirichlet)** compact
finite-difference derivative on one axis. Built by [`compact_setup_np`](@ref).

The operator is the 6th-order compact derivative in the interior, with one-sided
(boundary) and relaxed rows at the two ends so that the compact relation is
closed without wrapping. Unlike the periodic case, the LHS is a plain
(non-cyclic) tridiagonal matrix whose sub/super diagonals are row-varying near
the boundary; it is solved with an ordinary Thomas sweep (no rank-1 cyclic
correction). The stencil rows and the LHS diagonals are the GPS `cdl==2`
operators, cross-checked with Xcompact3d (`ncl==2`).

$(TYPEDFIELDS)
"""
struct CompactAxisNP{R} <: AbstractCompactAxis{R}
    "number of grid points along the axis."
    n::Int
    "derivative order (1 or 2)."
    order::Int
    "interior compact coefficient (1/3 or 2/11)."
    alpha::R
    "interior stencil coefficient a."
    a::R
    "interior stencil coefficient b."
    b::R
    "one-sided 1st-derivative boundary weights (rows 1 and n)."
    af1::R
    bf1::R
    cf1::R
    afn::R
    bfn::R
    cfn::R
    "relaxed 1st-derivative weight (rows 2 and n-1)."
    af2::R
    "one-sided 2nd-derivative boundary weights (rows 1 and n)."
    as1::R
    bs1::R
    cs1::R
    ds1::R
    asn::R
    bsn::R
    csn::R
    dsn::R
    "relaxed 2nd-derivative weight (rows 2 and n-1)."
    as2::R
    "tridiagonal sub-diagonal (element (i, i-1), row-varying near the ends)."
    sub::Vector{R}
    "tridiagonal super-diagonal (element (i, i+1), row-varying near the ends)."
    sup::Vector{R}
    "Thomas forward multipliers."
    s::Vector{R}
    "reciprocals of the Thomas pivots."
    w::Vector{R}
end

"""
$(TYPEDEF)

Precomputed multipliers for the **non-periodic (homogeneous Neumann)** compact
finite-difference derivative on one axis. Built by [`compact_setup_neu`](@ref).

The boundary is closed by an even (Neumann, `du/dn = 0`) mirror of the field
about each wall, so the right-hand side is just the interior stencil evaluated
on the mirrored field: only the interior coefficients `a`, `b` and the
row-varying LHS (`sub`/`sup` plus the Thomas factors `s`/`w`) are stored —
no one-sided boundary weights, unlike the Dirichlet path.

$(TYPEDFIELDS)
"""
struct CompactAxisNeu{R} <: AbstractCompactAxis{R}
    "number of grid points along the axis."
    n::Int
    "derivative order (1 or 2)."
    order::Int
    "interior compact coefficient (1/3 or 2/11)."
    alpha::R
    "interior stencil coefficient a."
    a::R
    "interior stencil coefficient b."
    b::R
    "tridiagonal sub-diagonal (element (i, i-1), row-varying near the ends)."
    sub::Vector{R}
    "tridiagonal super-diagonal (element (i, i+1), row-varying near the ends)."
    sup::Vector{R}
    "Thomas forward multipliers."
    s::Vector{R}
    "reciprocals of the Thomas pivots."
    w::Vector{R}
end

"""$(TYPEDSIGNATURES)

Fourier symbol of the 6th-order periodic compact stencil, evaluated
pointwise on the wavenumbers `ξ` (rad/m) at grid spacing `Δ`.

The compact operator is a circulant pentadiagonal matrix (diagonal 1, first
off-diagonal `a`, second `b`, corner `alpha`), so it is diagonalised by the
discrete Fourier transform and is therefore a pure Fourier multiplier. That
makes it trivially GPU-friendly (elementwise multiply after an FFT), in
contrast with the equivalent Thomas line-solve, which needs scalar indexing
and so cannot run on GPU arrays. The result is identical to the Thomas solve
to machine precision (see the compact tests in `test/runtests.jl`).

Symbols (θ = ξ·Δ, the dimensionless phase per bin):
* `order=1`: `(2ia sinθ + 2ib sin2θ) / (1 + 2α cosθ)`, α=1/3, a=(7/9)/Δ, b=(1/36)/Δ
* `order=2`: `(-4a sin²(θ/2) − 4b sin²θ) / (1 + 2α cosθ)`, α=2/11, a=(12/11)/Δ², b=(3/44)/Δ²
"""
function compact_multiplier(ξ, Δ, order)
    θ = ξ .* Δ
    α = order == 1 ? 1/3 : 2/11
    denom = 1 .+ 2α .* cos.(θ)
    if order == 1
        a, b = (7/9)/Δ, (1/36)/Δ
        return (2im * a * sin.(θ) .+ 2im * b * sin.(2θ)) ./ denom
    elseif order == 2
        a, b = (12/11)/Δ^2, (3/44)/Δ^2
        return (-4a * sin.(θ ./ 2) .^ 2 .- 4b * sin.(θ) .^ 2) ./ denom
    else
        error("compact_multiplier: order must be 1 or 2")
    end
end

"""$(TYPEDSIGNATURES)

Build the precomputed multipliers for a compact finite-difference derivative
along an axis of `n` points, spacing `Δ`, derivative `order` (1 or 2).

The operator is the 6th-order compact derivative
`` α·(g(i-1)+g(i+1)) + g(i) = a·(f(i+1)-f(i-1)) + b·(f(i+2)-f(i-2)) ``
(1st order, `α=1/3`, `a=(7/9)/Δ`, `b=(1/36)/Δ`) or the compact second
derivative (2nd order, `α=2/11`, `a=(12/11)/Δ²`, `b=(3/44)/Δ²`), with periodic
wrap.

The periodic wrap makes the coefficient matrix cyclic pentadiagonal. It is
solved without building the matrix: a Thomas factorization of the leading
tridiagonal part plus a cyclic corner correction. Because the corner
perturbation is rank-one (a single `α` coupling at each end), the correction
vector and its denominator depend only on the matrix, not on the right-hand
side; both are precomputed here so that each evaluation costs one stencil plus
two sweeps.
"""
function compact_setup(n::Int, Δ::Real, order::Int)
    if n < 8
        error("compact finite-difference scheme requires at least 8 points along the axis (got $n)")
    end
    R = eltype(Δ)
    if order == 1
        alpha = R(1)/R(3)
        a = (R(7)/R(9))/Δ
        b = (R(1)/R(36))/Δ
    elseif order == 2
        alpha = R(2)/R(11)
        a = (R(12)/R(11))/Δ^2
        b = (R(3)/R(44))/Δ^2
    else
        error("compact_setup: order must be 1 or 2")
    end
    cv = ones(R, n); cv[1] = R(2); cv[n] = R(1) + alpha^2
    bv = fill(alpha, n); bv[n] = zero(R)
    fv = fill(alpha, n); fv[n] = zero(R)
    # Thomas forward pass (non-cyclic part).
    s = zeros(R, n); w = ones(R, n)
    w .= cv
    for i in 2:n
        s[i] = bv[i-1]/w[i-1]
        w[i] = w[i] - fv[i-1]*s[i]
    end
    for i in 1:n
        w[i] = R(1)/w[i]
    end
    # Right-hand-side-independent cyclic correction (swept once, reused).
    t = zeros(R, n); t[1] = -one(R); t[n] = alpha
    for i in 2:n
        t[i] -= t[i-1]*s[i]
    end
    t[n] *= w[n]
    for i in n-1:-1:1
        t[i] = (t[i] - fv[i]*t[i+1])*w[i]
    end
    denom = one(R) + t[1] - alpha*t[n]
    return CompactAxis{R}(n, order, alpha, a, b, s, w, fv, t, denom)
end

"""$(TYPEDSIGNATURES)

Build the precomputed multipliers for the **non-periodic (homogeneous
Dirichlet)** compact derivative along an axis of `n` points, spacing `Δ`,
derivative `order` (1 or 2).

The interior is the same 6th-order compact stencil as [`compact_setup`](@ref).
At each end, the two boundary rows use one-sided stencils and the two adjacent
rows are relaxed (lower-order right-hand side with a modified LHS row), so the
compact relation is closed without wrapping — the GPS `cdl==2` / Xcompact3d
`ncl==2` operators:

- 1st derivative (`α=1/3`, `a=(7/9)/Δ`, `b=(1/36)/Δ`):
  row 1 `(−5u₁+4u₂+u₃)/2Δ`, row 2 `(3/4Δ)(u₃−u₁)`, mirrored rows `n-1`, `n`;
  LHS sub/super diagonals `2, 1/4, α,…, α, 1/4, 2` (row-varying at the ends).
- 2nd derivative (`α=2/11`, `a=(12/11)/Δ²`, `b=(3/44)/Δ²`):
  row 1 `(13u₁−27u₂+15u₃−u₄)/Δ²`, row 2 `(6/5Δ²)(u₃−2u₂+u₁)`, mirrored;
  LHS sub/super diagonals `11, 1/10, α,…, α, 1/10, 11`.

The LHS is a plain tridiagonal matrix (no cyclic corner), so the Thomas
factorization has no cyclic correction: the precomputed `s`, `w` suffice.
"""
function compact_setup_np(n::Int, Δ::Real, order::Int)
    if n < 8
        error("compact finite-difference scheme requires at least 8 points along the axis (got $n)")
    end
    R = eltype(Δ)
    if order == 1
        alpha = R(1)/R(3)
        a = (R(7)/R(9))/Δ
        b = (R(1)/R(36))/Δ
        af1 = -R(5)/R(2)/Δ; bf1 = R(2)/Δ; cf1 = R(1)/R(2)/Δ
        af2 = R(3)/R(4)/Δ
        afn = -R(5)/R(2)/Δ; bfn = R(2)/Δ; cfn = R(1)/R(2)/Δ
        as1 = zero(R); bs1 = zero(R); cs1 = zero(R); ds1 = zero(R)
        asn = zero(R); bsn = zero(R); csn = zero(R); dsn = zero(R)
        as2 = zero(R)
        r1, r2, rn1, rn = R(2), R(1)/R(4), R(1)/R(4), R(2)
    elseif order == 2
        alpha = R(2)/R(11)
        a = (R(12)/R(11))/Δ^2
        b = (R(3)/R(44))/Δ^2
        af1 = zero(R); bf1 = zero(R); cf1 = zero(R)
        af2 = zero(R)
        afn = zero(R); bfn = zero(R); cfn = zero(R)
        as1 = R(13)/Δ^2; bs1 = -R(27)/Δ^2; cs1 = R(15)/Δ^2; ds1 = -R(1)/Δ^2
        as2 = R(6)/R(5)/Δ^2
        asn = R(13)/Δ^2; bsn = -R(27)/Δ^2; csn = R(15)/Δ^2; dsn = -R(1)/Δ^2
        r1, r2, rn1, rn = R(11), R(1)/R(10), R(1)/R(10), R(11)
    else
        error("compact_setup_np: order must be 1 or 2")
    end
    # LHS tridiagonal, row-varying near the ends (homogeneous Dirichlet).
    # sub[i] = element (i, i-1), sup[i] = element (i, i+1); diagonal = 1.
    sub = fill(alpha, n); sup = fill(alpha, n)
    sub[1] = zero(R); sup[1] = r1          # row 1 (one-sided)
    sub[2] = r2;    sup[2] = r2            # row 2 (relaxed)
    sup[n - 1] = rn1; sub[n - 1] = rn1     # row n-1 (relaxed)
    sup[n] = zero(R); sub[n] = rn          # row n (one-sided)
    # Plain (non-cyclic) Thomas factorization, GPS `prepare_compact`.
    w = ones(R, n)                          # w = 1/c, c = 1 (diagonal)
    s = zeros(R, n)
    for i in 2:n
        s[i] = sub[i] / w[i - 1]
        w[i] = w[i] - sup[i - 1] * s[i]
    end
    for i in 1:n
        w[i] = R(1) / w[i]
    end
    return CompactAxisNP{R}(n, order, alpha, a, b, af1, bf1, cf1, afn, bfn, cfn, af2,
                            as1, bs1, cs1, ds1, asn, bsn, csn, dsn, as2,
                            sub, sup, s, w)
end

"""$(TYPEDSIGNATURES)

Build the precomputed multipliers for the **non-periodic (homogeneous
Neumann)** compact derivative along an axis of `n` points, spacing `Δ`,
derivative `order` (1 or 2).

The boundary is closed by an even (Neumann, `du/dn = 0`) mirror of the field
about each wall: the field is even about the boundary node, so the 1st
derivative is odd (it vanishes at the wall) and the 2nd derivative is even.
Evaluating the *interior* 6th-order compact stencil on the mirrored field
closes the system — no one-sided boundary stencils are needed (this is the
Xcompact3d `ncl==1` operator, ported and validated by 6th-order convergence
against an analytic Neumann field).

Consequences for the row-varying LHS (tridiagonal, diagonal 1, off-diagonals
`α` in the interior):
* 1st derivative: the boundary rows are identity (`g_1 = r_1 = 0` and
  `g_n = r_n = 0`), i.e. `sup[1] = sub[n] = 0` — that *is* the Neumann
  condition. All other off-diagonals are `α`.
* 2nd derivative: the boundary rows keep the mirrored coupling,
  `g_1 + 2α g_2 = r_1` and `g_n + 2α g_{n-1} = r_n`, i.e.
  `sup[1] = sub[n] = 2α`; all other off-diagonals are `α`.

Solved by a plain (non-cyclic) Thomas sweep, exactly as the Dirichlet NP path
([`compact_setup_np`](@ref)); the line kernels dispatch on this axis type.
"""
function compact_setup_neu(n::Int, Δ::Real, order::Int)
    if n < 8
        error("compact finite-difference scheme requires at least 8 points along the axis (got $n)")
    end
    R = eltype(Δ)
    if order == 1
        alpha = R(1)/R(3)
        a = (R(7)/R(9))/Δ
        b = (R(1)/R(36))/Δ
    elseif order == 2
        alpha = R(2)/R(11)
        a = (R(12)/R(11))/Δ^2
        b = (R(3)/R(44))/Δ^2
    else
        error("compact_setup_neu: order must be 1 or 2")
    end
    # LHS tridiagonal: α off-diagonals everywhere except the two boundary
    # couplings, which depend on the derivative order (see docstring).
    sub = fill(alpha, n); sup = fill(alpha, n)
    if order == 1
        sup[1] = zero(R); sub[n] = zero(R)   # 1st deriv: boundary rows identity
    else
        sup[1] = 2alpha; sub[n] = 2alpha     # 2nd deriv: mirrored coupling
    end
    # Plain (non-cyclic) Thomas factorization (same as the Dirichlet path).
    w = ones(R, n)
    s = zeros(R, n)
    for i in 2:n
        s[i] = sub[i] / w[i - 1]
        w[i] = w[i] - sup[i - 1] * s[i]
    end
    for i in 1:n
        w[i] = R(1) / w[i]
    end
    return CompactAxisNeu{R}(n, order, alpha, a, b, sub, sup, s, w)
end

"""$(TYPEDSIGNATURES)

Build the per-axis compact multipliers for one axis of `n` points, spacing
`Δ`, derivative `order` (1 or 2) and boundary condition `bc`
(`0`/`:periodic` → [`compact_setup`](@ref), `1`/`:dirichlet` →
[`compact_setup_np`](@ref), `2`/`:neumann` → [`compact_setup_neu`](@ref)).
"""
function compact_axis(n::Int, Δ::Real, order::Int, bc)
    bi = _bc_int(bc)
    return bi == 1 ? compact_setup_np(n, Δ, order) :
           bi == 2 ? compact_setup_neu(n, Δ, order) :
                     compact_setup(n, Δ, order)
end

"""$(TYPEDSIGNATURES)

Resolve the `bcs` of a `CompactPlan` to a tuple of `dim` integers (one per
axis, `0` periodic / `1` Dirichlet); an empty `bcs` means all periodic.
"""
function _compact_bcs(t::CompactPlan, dim::Int)
    bcs = isempty(t.bcs) ? ntuple(i -> 0, Val(dim)) : t.bcs
    length(bcs) == dim ||
        error("CompactPlan: bcs must have one entry per axis ($dim), got $(length(bcs)).")
    return ntuple(i -> _bc_int(bcs[i]), Val(dim))
end

"""
$(TYPEDEF)

Type representing a 2D FFT plan.

It contains the following informations:

$(TYPEDFIELDS)
"""
struct PlanFFT2D{N} <: AbstractFFTPlan{N}
    "reference to a field."
    f::AbstractField2D
    "pencil (or slab) in the ``x`` direction."
    pen_x::Any
    "pencil (or slab) in the ``y`` direction."
    pen_y::Any
    "FFT plan in the ``x`` direction."
    plan_x::AbstractFFTs.Plan
    "FFT plan in the ``y`` direction."
    plan_y::AbstractFFTs.Plan
    "``x`` frequency (non distributed)."
    ξx::AbstractArray
    "``y`` frequency (non distributed)."
    ξy::AbstractArray
    "fields to contain transformed fields distributed along ``x``."
    datax::Vector{AbstractArray}
    "fields to contain transformed fields distributed along ``y``."
    datay::Vector{AbstractArray}
end

"""
$(TYPEDEF)

Type representing a 3D FFT plan.

It contains the following informations:

$(TYPEDFIELDS)
"""
struct PlanFFT3D{N} <: AbstractFFTPlan{N}
    "reference to a field."
    f::AbstractField3D
    "pencil (or slab) in the ``x`` direction."
    pen_x::Any
    "pencil (or slab) in the ``y`` direction."
    pen_y::Any
    "pencil (or slab) in the ``z`` direction."
    pen_z::Any
    "FFT plan in the ``x`` direction."
    plan_x::AbstractFFTs.Plan
    "FFT plan in the ``y`` direction."
    plan_y::AbstractFFTs.Plan
    "FFT plan in the ``z`` direction."
    plan_z::AbstractFFTs.Plan
    "``x`` frequency (non distributed)."
    ξx::AbstractArray
    "``y`` frequency (non distributed)."
    ξy::AbstractArray
    "``z`` frequency (non distributed)."
    ξz::AbstractArray
    "fields to contain transformed fields distributed along ``x``."
    datax::Vector{AbstractArray}
    "fields to contain transformed fields distributed along ``y``."
    datay::Vector{AbstractArray}
    "fields to contain transformed fields distributed along ``z``."
    dataz::Vector{AbstractArray}
end

function ξsquared(ξx::Real, ξy::Real, ξz::Real)
    a = ξx^2 + ξy^2 + ξz^2
    if abs(a) < 1e-8
        return 1
    else
        return a
    end
end

function ξsquared2(ξx::Real, ξy::Real)
    a = ξx^2 + ξy^2
    if abs(a) < 1e-8
        return 1
    else
        return a
    end
end

"""
    ξmax_global(plan)

2/3-rule bound ξmax = (4/9)·min_d max|ξ_d|², from the plan's GLOBAL
wavenumber vectors. The spectral fields are dealiased on the last pencil,
whose grid is distributed over two directions: a rank-local max sees only
part of the spectrum, its bound is smaller than the global one, and the rank
then zeroes modes its peers keep — physics that depends on the rank count
(measured: 1.9e-3 energy drift 1→4 ranks at N=48, benchmarks/
scalability_rankinv.jl). The plan stores the full (undistributed) ξ vectors,
so the global bound costs nothing and needs no MPI reduction. Typed 2D/3D
methods (no `isdefined` probe); models cache the value at construction
(`n.ξmax`) so stepping does not re-reduce the ξ vectors — on GPU each call
would be a device reduction plus a synchronisation.
"""
ξmax_global(plan::PlanFFT2D) = (4 / 9) * min(maximum(abs2, plan.ξx), maximum(abs2, plan.ξy))
function ξmax_global(plan::PlanFFT3D)
    return (4 / 9) * min(maximum(abs2, plan.ξx), maximum(abs2, plan.ξy), maximum(abs2, plan.ξz))
end

"""
$(TYPEDEF)

Type representing a 2D finite difference plan.

It contains the following informations:

$(TYPEDFIELDS)
"""
struct PlanFD2D{N} <: AbstractFDPlan{N}
    "reference to a field."
    f::AbstractField2D
    "pencil (or slab) in the ``x`` direction."
    pen_x::Any
    "pencil (or slab) in the ``y`` direction."
    pen_y::Any
    "discretization step along ``x``."
    Δx::Real
    "discretization step along ``y``."
    Δy::Real
    "temporary fields distributed along ``x``."
    ϕxtmp::AbstractArray
    "temporary fields distributed along ``y``."
    ϕytmp::AbstractArray
end

"""
$(TYPEDEF)

Type representing a 3D finite difference plan.

It contains the following informations:

$(TYPEDFIELDS)
"""
struct PlanFD3D{N} <: AbstractFDPlan{N}
    "reference to a field."
    f::AbstractField3D
    "pencil (or slab) in the ``x`` direction."
    pen_x::Any
    "pencil (or slab) in the ``y`` direction."
    pen_y::Any
    "pencil (or slab) in the ``z`` direction."
    pen_z::Any
    "discretization step along ``x``."
    Δx::Real
    "discretization step along ``y``."
    Δy::Real
    "discretization step along ``z``."
    Δz::Real
    "temporary fields distributed along ``x``."
    ϕxtmp::AbstractArray
    "temporary fields distributed along ``y``."
    ϕytmp::AbstractArray
    "temporary fields distributed along ``z``."
    ϕztmp::AbstractArray
end

"""
$(TYPEDEF)

Type representing a 2D compact finite difference plan.

$(TYPEDFIELDS)
"""
struct PlanCompact2D{N} <: AbstractCompactPlan{N}
    "reference to a field."
    f::AbstractField2D
    "pencil (or slab) in the ``x`` direction."
    pen_x::Any
    "pencil (or slab) in the ``y`` direction."
    pen_y::Any
    "compact multipliers for the first derivative along ``x``."
    ax::AbstractCompactAxis
    "compact multipliers for the second derivative along ``x``."
    a2x::AbstractCompactAxis
    "compact multipliers for the first derivative along ``y``."
    ay::AbstractCompactAxis
    "compact multipliers for the second derivative along ``y``."
    a2y::AbstractCompactAxis
    "temporary field distributed along ``y``."
    ϕytmp::AbstractArray
end

"""
$(TYPEDEF)

Type representing a 3D compact finite difference plan.

$(TYPEDFIELDS)
"""
struct PlanCompact3D{N} <: AbstractCompactPlan{N}
    "reference to a field."
    f::AbstractField3D
    "pencil (or slab) in the ``x`` direction."
    pen_x::Any
    "pencil (or slab) in the ``y`` direction."
    pen_y::Any
    "pencil (or slab) in the ``z`` direction."
    pen_z::Any
    "compact multipliers for the first derivative along ``x``."
    ax::AbstractCompactAxis
    "compact multipliers for the second derivative along ``x``."
    a2x::AbstractCompactAxis
    "compact multipliers for the first derivative along ``y``."
    ay::AbstractCompactAxis
    "compact multipliers for the second derivative along ``y``."
    a2y::AbstractCompactAxis
    "compact multipliers for the first derivative along ``z``."
    az::AbstractCompactAxis
    "compact multipliers for the second derivative along ``z``."
    a2z::AbstractCompactAxis
    "temporary field distributed along ``y``."
    ϕytmp::AbstractArray
    "temporary field distributed along ``z``."
    ϕztmp::AbstractArray
end

"""
$(TYPEDEF)

Spectral-compact (Fourier-multiplier) 2D plan.

On GPU arrays the Thomas line-solve used by `PlanCompact2D` cannot run (it
needs scalar indexing on the local line, which GPU arrays forbid from the
host). This plan instead exploits the fact that the *periodic* compact
operator is a circulant matrix, hence a Fourier multiplier: each derivative
is `IDFT(T(ξ)·FFT(ϕ))` with `T` the compact symbol from
[`compact_multiplier`](@ref). It reuses the standard FFT plan machinery
(`mul_x!`/`ldiv_x!`/`grid_x`/…), which is fully GPU-native, and yields the
same derivatives as the Thomas implementation to machine precision.

Only complex fields are supported, because the FFT scratch buffers are
complex.

# Fields
* `fft`: the underlying FFT plan providing the transforms, wavenumbers and
  scratch buffers.
"""
struct PlanCompactFFT2D{N} <: AbstractCompactPlan{N}
    "underlying FFT plan."
    fft::PlanFFT2D{N}
    "compact first-derivative Fourier symbol vs the ``x`` wavenumber."
    T1x::AbstractVector
    "compact second-derivative Fourier symbol vs the ``x`` wavenumber."
    T2x::AbstractVector
    "compact first-derivative Fourier symbol vs the ``y`` wavenumber."
    T1y::AbstractVector
    "compact second-derivative Fourier symbol vs the ``y`` wavenumber."
    T2y::AbstractVector
end

"""
$(TYPEDEF)

3D analogue of `PlanCompactFFT2D`: a spectral-compact plan built from an
underlying FFT plan (see `PlanCompactFFT2D` for the rationale).

# Fields
* `fft`: the underlying FFT plan providing the transforms, wavenumbers and
  scratch buffers.
"""
struct PlanCompactFFT3D{N} <: AbstractCompactPlan{N}
    "underlying FFT plan."
    fft::PlanFFT3D{N}
    "compact first-derivative Fourier symbol vs the ``x`` wavenumber."
    T1x::AbstractVector
    "compact second-derivative Fourier symbol vs the ``x`` wavenumber."
    T2x::AbstractVector
    "compact first-derivative Fourier symbol vs the ``y`` wavenumber."
    T1y::AbstractVector
    "compact second-derivative Fourier symbol vs the ``y`` wavenumber."
    T2y::AbstractVector
    "compact first-derivative Fourier symbol vs the ``z`` wavenumber."
    T1z::AbstractVector
    "compact second-derivative Fourier symbol vs the ``z`` wavenumber."
    T2z::AbstractVector
end

"""
$(TYPEDEF)

GPU compact plan (CUDA Thomas kernel), 2D.

The periodic compact relation is solved on the device with a dedicated
kernel: one thread per grid line performs the Thomas forward sweep, the
cyclic correction and the back substitution. The banded multipliers are
precomputed per axis (`CompactAxisGPU`, device-side) so the kernel only
runs the per-call elimination. `RealField` works as well as `ComplexField`
(the kernel is dtype-agnostic), unlike the FFT-based plans.

# Fields
* `ax`, `a2x`, `ay`, `a2y`: per-axis compact multipliers (1st and 2nd
  derivative), kept on the device.
* `ϕytmp`, `dytmp`, `ddytmp`: preallocated scratch fields in the ``y``
  layout used by the y-direction dispatch (avoids per-call GPU allocation).
"""
struct PlanCompactGPU2D{N} <: AbstractCompactPlan{N}
    "reference to a field."
    f::AbstractField2D
    "pencil (or slab) in the ``x`` direction."
    pen_x::Any
    "pencil (or slab) in the ``y`` direction."
    pen_y::Any
    "compact multipliers for the first derivative along ``x`` (device)."
    ax::AbstractCompactAxisGPU
    "compact multipliers for the second derivative along ``x`` (device)."
    a2x::AbstractCompactAxisGPU
    "compact multipliers for the first derivative along ``y`` (device)."
    ay::AbstractCompactAxisGPU
    "compact multipliers for the second derivative along ``y`` (device)."
    a2y::AbstractCompactAxisGPU
    "temporary field distributed along ``y`` (input, transposed ``ϕ``)."
    ϕytmp::AbstractArray
    "temporary field distributed along ``y`` (first-derivative output)."
    dytmp::AbstractArray
    "temporary field distributed along ``y`` (second-derivative output)."
    ddytmp::AbstractArray
end

"""
$(TYPEDEF)

3D analogue of `PlanCompactGPU2D` (see it for the rationale). The z
direction is solved in the ``z`` layout and brought back through the ``y``
layout, mirroring the CPU dispatch.
"""
struct PlanCompactGPU3D{N} <: AbstractCompactPlan{N}
    "reference to a field."
    f::AbstractField3D
    "pencil (or slab) in the ``x`` direction."
    pen_x::Any
    "pencil (or slab) in the ``y`` direction."
    pen_y::Any
    "pencil (or slab) in the ``z`` direction."
    pen_z::Any
    "compact multipliers for the first derivative along ``x`` (device)."
    ax::AbstractCompactAxisGPU
    "compact multipliers for the second derivative along ``x`` (device)."
    a2x::AbstractCompactAxisGPU
    "compact multipliers for the first derivative along ``y`` (device)."
    ay::AbstractCompactAxisGPU
    "compact multipliers for the second derivative along ``y`` (device)."
    a2y::AbstractCompactAxisGPU
    "compact multipliers for the first derivative along ``z`` (device)."
    az::AbstractCompactAxisGPU
    "compact multipliers for the second derivative along ``z`` (device)."
    a2z::AbstractCompactAxisGPU
    "temporary field distributed along ``y`` (input, transposed ``ϕ``)."
    ϕytmp::AbstractArray
    "temporary field distributed along ``y`` (derivative output)."
    dytmp::AbstractArray
    "temporary field distributed along ``y`` (second-derivative output)."
    ddytmp::AbstractArray
    "temporary field distributed along ``z`` (input, transposed ``ϕ``)."
    ϕztmp::AbstractArray
    "temporary field distributed along ``z`` (first-derivative output)."
    dztmp::AbstractArray
    "temporary field distributed along ``z`` (second-derivative output)."
    ddztmp::AbstractArray
end

"""
$(TYPEDSIGNATURES)

Returns a 2D plan. By default a FFT plan is returned.

Parameters are:

- `f`: a field
- `t`: a type of plan (either `FFTPlan()`, `FiniteDifferencePlan()`, or `CompactPlan()`)

# Example

```jldoctest
julia> field = Field(grid, RealField());
julia> plan = Plan(field, t=FFTPlan());
```

```jldoctest
julia> field = Field(grid, RealField());
julia> plan = Plan(field, t=FiniteDifferencePlan());
```

```jldoctest
julia> field = Field(grid, ComplexField());
julia> plan = Plan(field, t=CompactPlan());
```

```jldoctest
julia> field = Field(grid, ComplexField());
julia> plan = Plan(field, t=CompactPlan(bcs=(2, 0)));  # Neumann in x, periodic in y
```
"""
function Plan(f::F;
              t::PlanType=FFTPlan()) where {F<:AbstractField2D{N,FT,FFT,A}} where {N,FT,FFT,
                                                                                   A}
    # Pencil decompositions
    pen_x = f.pen
    pen_y = Pencil(pen_x; decomp_dims=(1,), permute=Permutation(2, 1))
    if typeof(t) == FFTPlan
        # frequencies
        ξx = A(fftfreq(f.g.nx, 2π / f.g.Δx))
        ξy = A(fftfreq(f.g.ny, 2π / f.g.Δy))

        # create arrays for FFT
        datax = [PencilArray{FFT}(undef, pen_x), PencilArray{FFT}(undef, pen_x),
                 PencilArray{FFT}(undef, pen_x)]
        datay = [PencilArray{FFT}(undef, pen_y), PencilArray{FFT}(undef, pen_y),
                 PencilArray{FFT}(undef, pen_y)]
        for i in 1:(N - 1)
            push!(datax, PencilArray{FFT}(undef, pen_x))
            push!(datax, PencilArray{FFT}(undef, pen_x))
            push!(datax, PencilArray{FFT}(undef, pen_x))
            push!(datay, PencilArray{FFT}(undef, pen_y))
            push!(datay, PencilArray{FFT}(undef, pen_y))
            push!(datay, PencilArray{FFT}(undef, pen_y))
        end

        # create plans
        plan_x = plan_fft(parent(datax[1]), 1)
        plan_y = plan_fft(parent(datay[1]), 1)

        return PlanFFT2D{N}(f,
                            pen_x, pen_y,
                            plan_x, plan_y,
                            ξx, ξy,
                            datax, datay)
    elseif t isa CompactPlan
        if A === Array
            # CPU: compact Thomas solve (fastest compact path on CPU). Each axis
            # is built for its boundary condition (periodic or Dirichlet).
            bcx, bcy = _compact_bcs(t, 2)
            ax = compact_axis(f.g.nx, f.g.Δx, 1, bcx)
            a2x = compact_axis(f.g.nx, f.g.Δx, 2, bcx)
            ay = compact_axis(f.g.ny, f.g.Δy, 1, bcy)
            a2y = compact_axis(f.g.ny, f.g.Δy, 2, bcy)
            ϕytmp = PencilArray{FFT}(undef, pen_y)
            return PlanCompact2D{N}(f, pen_x, pen_y, ax, a2x, ay, a2y, ϕytmp)
        end
        # GPU (or any non-Array backend).
        if t.backend === :spectral
            # Spectral path: the periodic compact operator is a circulant
            # matrix, hence a Fourier multiplier, evaluated through the
            # standard FFT machinery (see PlanCompactFFT2D). Requires a
            # complex field, like the standard FFTPlan, and periodic axes only
            # (a bounded axis has no Fourier-multiplier form).
            if any(bc -> bc != 0, _compact_bcs(t, 2))
                error("CompactPlan(backend=:spectral) is only defined for periodic axes; " *
                      "this plan has non-periodic (Dirichlet or Neumann) axes, which have no " *
                      "Fourier-multiplier form. Use the default backend (CUDA Thomas " *
                      "kernel) instead.")
            end
            if !(eltype(f.data[1]) <: Complex)
                error("CompactPlan(backend=:spectral) on a $(A) grid requires a complex field " *
                      "(RealField is not supported, like the standard FFTPlan). Build the field " *
                      "with ComplexField(), or use the default backend (CUDA Thomas kernel, " *
                      "which supports RealField), or FFTPlan()/FiniteDifferencePlan().")
            end
            pfft = Plan(f; t=FFTPlan())
            ξx = A(fftfreq(f.g.nx, 2π / f.g.Δx))
            ξy = A(fftfreq(f.g.ny, 2π / f.g.Δy))
            return PlanCompactFFT2D{N}(pfft,
                                       compact_multiplier(ξx, f.g.Δx, 1),
                                       compact_multiplier(ξx, f.g.Δx, 2),
                                       compact_multiplier(ξy, f.g.Δy, 1),
                                       compact_multiplier(ξy, f.g.Δy, 2))
        end
        # Default (:auto / :thomas): dedicated CUDA Thomas kernel (periodic and
        # Dirichlet axes alike). CUDA must be available (it is a hard
        # dependency of this package, but the kernels are only compiled when it
        # loads).
        if CUDA === nothing
            error("CompactPlan(backend=$(t.backend)) on a $(A) grid needs CUDA, " *
                  "which is not available in this build. Use CompactPlan(backend=:spectral) " *
                  "(Fourier multiplier, periodic complex fields only) instead.")
        end
        bcx, bcy = _compact_bcs(t, 2)
        function _gpu_axis(n, Δ, order, bc)
            return bc == 1 ? compact_setup_np_gpu(n, Δ, order) :
                   bc == 2 ? compact_setup_neu_gpu(n, Δ, order) :
                             compact_setup_gpu(n, Δ, order)
        end
        ax = _gpu_axis(f.g.nx, f.g.Δx, 1, bcx)
        a2x = _gpu_axis(f.g.nx, f.g.Δx, 2, bcx)
        ay = _gpu_axis(f.g.ny, f.g.Δy, 1, bcy)
        a2y = _gpu_axis(f.g.ny, f.g.Δy, 2, bcy)
        ϕytmp = PencilArray{FFT}(undef, pen_y)
        dytmp = PencilArray{FFT}(undef, pen_y)
        ddytmp = PencilArray{FFT}(undef, pen_y)
        return PlanCompactGPU2D{N}(f, pen_x, pen_y, ax, a2x, ay, a2y, ϕytmp, dytmp, ddytmp)
    else
        # create arrays for Finite Difference
        ϕxtmp = PencilArray{FFT}(undef, pen_x)
        ϕytmp = PencilArray{FFT}(undef, pen_y)
        return PlanFD2D{N}(f, pen_x, pen_y, f.g.Δx, f.g.Δy, ϕxtmp, ϕytmp)
    end
end

"""
$(TYPEDSIGNATURES)

Returns a 3D plan. By default a FFT plan is returned.

Parameters are:

- `f`: a field
- `t`: a type of plan (either `FFTPlan()`, `FiniteDifferencePlan()`, or `CompactPlan()`)

# Example

```jldoctest
julia> field3 = Field(grid3, RealField());
julia> plan = Plan(field3, t=FFTPlan());
```

```jldoctest
julia> field3 = Field(grid3, RealField());
julia> plan = Plan(field3, t=FiniteDifferencePlan());
```

```jldoctest
julia> field3 = Field(grid3, ComplexField());
julia> plan = Plan(field3, t=CompactPlan());
```

```jldoctest
julia> field3 = Field(grid3, ComplexField());
julia> plan = Plan(field3, t=CompactPlan(bcs=(2, 0, 2)));  # Neumann in x, z
```
"""
function Plan(f::F;
              t::PlanType=FFTPlan()) where {F<:AbstractField3D{N,FT,FFT,A}} where {N,FT,FFT,
                                                                                   A}
    # Pencil decompositions
    pen_x = f.pen
    pen_y = Pencil(pen_x; decomp_dims=(1, 3), permute=Permutation(2, 1, 3))
    pen_z = Pencil(pen_x; decomp_dims=(1, 2), permute=Permutation(3, 1, 2))
    if typeof(t) == FFTPlan
        # frequencies
        ξx = A(fftfreq(f.g.nx, 2π / f.g.Δx))
        ξy = A(fftfreq(f.g.ny, 2π / f.g.Δy))
        ξz = A(fftfreq(f.g.nz, 2π / f.g.Δz))

        # create arrays for FFT
        datax = [PencilArray{FFT}(undef, pen_x), PencilArray{FFT}(undef, pen_x),
                 PencilArray{FFT}(undef, pen_x)]
        datay = [PencilArray{FFT}(undef, pen_y), PencilArray{FFT}(undef, pen_y),
                 PencilArray{FFT}(undef, pen_y)]
        dataz = [PencilArray{FFT}(undef, pen_z), PencilArray{FFT}(undef, pen_z),
                 PencilArray{FFT}(undef, pen_z)]
        for i in 1:(N - 1)
            push!(datax, PencilArray{FFT}(undef, pen_x))
            push!(datax, PencilArray{FFT}(undef, pen_x))
            push!(datax, PencilArray{FFT}(undef, pen_x))
            push!(datay, PencilArray{FFT}(undef, pen_y))
            push!(datay, PencilArray{FFT}(undef, pen_y))
            push!(datay, PencilArray{FFT}(undef, pen_y))
            push!(dataz, PencilArray{FFT}(undef, pen_z))
            push!(dataz, PencilArray{FFT}(undef, pen_z))
            push!(dataz, PencilArray{FFT}(undef, pen_z))
        end

        # create plans
        plan_x = plan_fft(parent(datax[1]), 1)
        plan_y = plan_fft(parent(datay[1]), 1)
        plan_z = plan_fft(parent(dataz[1]), 1)

        return PlanFFT3D{N}(f,
                            pen_x, pen_y, pen_z,
                            plan_x, plan_y, plan_z,
                            ξx, ξy, ξz,
                            datax, datay, dataz)
    elseif t isa CompactPlan
        if A === Array
            # CPU: compact Thomas solve (fastest compact path on CPU). Each axis
            # is built for its boundary condition (periodic or Dirichlet).
            bcx, bcy, bcz = _compact_bcs(t, 3)
            ax = compact_axis(f.g.nx, f.g.Δx, 1, bcx)
            a2x = compact_axis(f.g.nx, f.g.Δx, 2, bcx)
            ay = compact_axis(f.g.ny, f.g.Δy, 1, bcy)
            a2y = compact_axis(f.g.ny, f.g.Δy, 2, bcy)
            az = compact_axis(f.g.nz, f.g.Δz, 1, bcz)
            a2z = compact_axis(f.g.nz, f.g.Δz, 2, bcz)
            ϕytmp = PencilArray{FFT}(undef, pen_y)
            ϕztmp = PencilArray{FFT}(undef, pen_z)
            return PlanCompact3D{N}(f, pen_x, pen_y, pen_z, ax, a2x, ay, a2y, az, a2z, ϕytmp, ϕztmp)
        end
        # GPU (or any non-Array backend).
        if t.backend === :spectral
            # See the 2D dispatch: Fourier multiplier through the standard
            # FFT machinery (complex fields only), periodic axes only.
            if any(bc -> bc != 0, _compact_bcs(t, 3))
                error("CompactPlan(backend=:spectral) is only defined for periodic axes; " *
                      "this plan has non-periodic (Dirichlet or Neumann) axes, which have no " *
                      "Fourier-multiplier form. Use the default backend (CUDA Thomas " *
                      "kernel) instead.")
            end
            if !(eltype(f.data[1]) <: Complex)
                error("CompactPlan(backend=:spectral) on a $(A) grid requires a complex field " *
                      "(RealField is not supported, like the standard FFTPlan). Build the field " *
                      "with ComplexField(), or use the default backend (CUDA Thomas kernel, " *
                      "which supports RealField), or FFTPlan()/FiniteDifferencePlan().")
            end
            pfft = Plan(f; t=FFTPlan())
            ξx = A(fftfreq(f.g.nx, 2π / f.g.Δx))
            ξy = A(fftfreq(f.g.ny, 2π / f.g.Δy))
            ξz = A(fftfreq(f.g.nz, 2π / f.g.Δz))
            return PlanCompactFFT3D{N}(pfft,
                                       compact_multiplier(ξx, f.g.Δx, 1),
                                       compact_multiplier(ξx, f.g.Δx, 2),
                                       compact_multiplier(ξy, f.g.Δy, 1),
                                       compact_multiplier(ξy, f.g.Δy, 2),
                                       compact_multiplier(ξz, f.g.Δz, 1),
                                       compact_multiplier(ξz, f.g.Δz, 2))
        end
        if CUDA === nothing
            error("CompactPlan(backend=$(t.backend)) on a $(A) grid needs CUDA, " *
                  "which is not available in this build. Use CompactPlan(backend=:spectral) " *
                  "(Fourier multiplier, periodic complex fields only) instead.")
        end
        bcx, bcy, bcz = _compact_bcs(t, 3)
        function _gpu_axis(n, Δ, order, bc)
            return bc == 1 ? compact_setup_np_gpu(n, Δ, order) :
                   bc == 2 ? compact_setup_neu_gpu(n, Δ, order) :
                             compact_setup_gpu(n, Δ, order)
        end
        ax = _gpu_axis(f.g.nx, f.g.Δx, 1, bcx)
        a2x = _gpu_axis(f.g.nx, f.g.Δx, 2, bcx)
        ay = _gpu_axis(f.g.ny, f.g.Δy, 1, bcy)
        a2y = _gpu_axis(f.g.ny, f.g.Δy, 2, bcy)
        az = _gpu_axis(f.g.nz, f.g.Δz, 1, bcz)
        a2z = _gpu_axis(f.g.nz, f.g.Δz, 2, bcz)
        ϕytmp = PencilArray{FFT}(undef, pen_y)
        dytmp = PencilArray{FFT}(undef, pen_y)
        ddytmp = PencilArray{FFT}(undef, pen_y)
        ϕztmp = PencilArray{FFT}(undef, pen_z)
        dztmp = PencilArray{FFT}(undef, pen_z)
        ddztmp = PencilArray{FFT}(undef, pen_z)
        return PlanCompactGPU3D{N}(f, pen_x, pen_y, pen_z, ax, a2x, ay, a2y, az, a2z,
                                   ϕytmp, dytmp, ddytmp, ϕztmp, dztmp, ddztmp)
    else
        # create arrays for Finite Difference
        ϕxtmp = PencilArray{FFT}(undef, pen_x)
        ϕytmp = PencilArray{FFT}(undef, pen_y)
        ϕztmp = PencilArray{FFT}(undef, pen_z)
        return PlanFD3D{N}(f, pen_x, pen_y, pen_z, f.g.Δx, f.g.Δy, f.g.Δz, ϕxtmp, ϕytmp,
                           ϕztmp)
    end
end

@inline function Base.getproperty(plan::AbstractFFTPlan{1}, name::Symbol)
    if name === :ϕx_hat
        return plan.datax[1]
    elseif name === :ϕy_hat
        return plan.datay[1]
    elseif name === :ϕz_hat
        return plan.dataz[1]
    elseif name === :a_tmpx
        return plan.datax[2]
    elseif name === :a_tmpy
        return plan.datay[2]
    elseif name === :a_tmpz
        return plan.dataz[2]
    elseif name === :a_tmp2x
        return plan.datax[3]
    elseif name === :a_tmp2y
        return plan.datay[3]
    elseif name === :a_tmp2z
        return plan.dataz[3]
    else
        return getfield(plan, name)
    end
end

@inline function Base.getproperty(plan::AbstractFFTPlan{N}, name::Symbol) where {N}
    if name === :ux_hat
        return plan.datax[1:N]
    elseif name === :uy_hat
        return plan.datay[1:N]
    elseif name === :uz_hat
        return plan.dataz[1:N]
    elseif name === :uxtmp_hat
        return plan.datax[(N + 1):(2 * N)]
    elseif name === :uytmp_hat
        return plan.datay[(N + 1):(2 * N)]
    elseif name === :uztmp_hat
        return plan.dataz[(N + 1):(2 * N)]
    elseif name === :a_tmpx
        return plan.datax[2 * N + 1]
    elseif name === :a_tmpy
        return plan.datay[2 * N + 1]
    elseif name === :a_tmpz
        return plan.dataz[2 * N + 1]
    elseif name === :a_tmp2x
        return plan.datax[2 * N + 2]
    elseif name === :a_tmp2y
        return plan.datay[2 * N + 2]
    elseif name === :a_tmp2z
        return plan.dataz[2 * N + 2]
    else
        return getfield(plan, name)
    end
end

function mul_x!(a_out::PencilArray, plan::AbstractFFTPlan, a_in::PencilArray)
    mul!(parent(a_out), plan.plan_x, parent(a_in))
    return nothing
end

function mul_x!(u_out::Vector{<:AbstractArray}, plan::AbstractFFTPlan{N},
                u_in::Vector{<:AbstractArray}) where {N}
    for i in 1:N
        mul_x!(u_out[i], plan, u_in[i])
    end
    return nothing
end

function mul_y!(a_out::PencilArray, plan::AbstractFFTPlan, a_in::PencilArray)
    a_in_y = plan.a_tmpy
    transpose!(a_in_y, a_in)
    mul!(parent(a_out), plan.plan_y, parent(a_in_y))
    return nothing
end

function mul_y!(u_out::Vector{<:AbstractArray}, plan::AbstractFFTPlan{N},
                u_in::Vector{<:AbstractArray}) where {N}
    for i in 1:N
        mul_y!(u_out[i], plan, u_in[i])
    end
    return nothing
end

function mul_z!(a_out::PencilArray, plan::AbstractFFTPlan, a_in::PencilArray)
    a_in_y = plan.a_tmpy
    a_in_z = plan.a_tmpz
    transpose!(a_in_y, a_in)
    transpose!(a_in_z, a_in_y)
    mul!(parent(a_out), plan.plan_z, parent(a_in_z))
    return nothing
end

function mul_z!(u_out::Vector{<:AbstractArray}, plan::AbstractFFTPlan{N},
                u_in::Vector{<:AbstractArray}) where {N}
    for i in 1:N
        mul_z!(u_out[i], plan, u_in[i])
    end
    return nothing
end

function mul_all!(a_out::PencilArray, plan::PlanFFT2D, a_in::PencilArray)
    a_out_x = plan.a_tmpx
    a_out_y = plan.a_tmpy
    mul!(parent(a_out_x), plan.plan_x, parent(a_in))
    transpose!(a_out_y, a_out_x)
    mul!(parent(a_out), plan.plan_y, parent(a_out_y))
    return nothing
end

function mul_all!(a_out::PencilArray, plan::PlanFFT3D, a_in::PencilArray)
    a_out_x = plan.a_tmpx
    a_out_y = plan.a_tmpy
    a_out_y2 = plan.a_tmp2y
    a_out_z = plan.a_tmpz
    mul!(parent(a_out_x), plan.plan_x, parent(a_in))
    transpose!(a_out_y, a_out_x)
    mul!(parent(a_out_y2), plan.plan_y, parent(a_out_y))
    transpose!(a_out_z, a_out_y2)
    mul!(parent(a_out), plan.plan_z, parent(a_out_z))
    return nothing
end

function mul_all!(u_out::Vector{<:AbstractArray}, plan::AbstractFFTPlan{N},
                  u_in::Vector{<:AbstractArray}) where {N}
    for i in 1:N
        mul_all!(u_out[i], plan, u_in[i])
    end
    return nothing
end

function ldiv_x!(a_out::PencilArray, plan::AbstractFFTPlan, a_in::PencilArray)
    ldiv!(parent(a_out), plan.plan_x, parent(a_in))
    return nothing
end

function ldiv_x!(u_out::Vector{<:AbstractArray}, plan::AbstractFFTPlan{N},
                 u_in::Vector{<:AbstractArray}) where {N}
    for i in 1:N
        ldiv_x!(u_out[i], plan, u_in[i])
    end
    return nothing
end

function ldiv_y!(a_out::PencilArray, plan::AbstractFFTPlan, a_in::PencilArray)
    a_out_y = plan.a_tmpy
    ldiv!(parent(a_out_y), plan.plan_y, parent(a_in))
    transpose!(a_out, a_out_y)
    return nothing
end

function ldiv_y!(u_out::Vector{<:AbstractArray}, plan::AbstractFFTPlan{N},
                 u_in::Vector{<:AbstractArray}) where {N}
    for i in 1:N
        ldiv_y!(u_out[i], plan, u_in[i])
    end
    return nothing
end

function ldiv_z!(a_out::PencilArray, plan::AbstractFFTPlan, a_in::PencilArray)
    a_out_y = plan.a_tmpy
    a_out_z = plan.a_tmpz
    ldiv!(parent(a_out_z), plan.plan_z, parent(a_in))
    transpose!(a_out_y, a_out_z)
    transpose!(a_out, a_out_y)
    return nothing
end

function ldiv_z!(u_out::Vector{<:AbstractArray}, plan::AbstractFFTPlan{N},
                 u_in::Vector{<:AbstractArray}) where {N}
    for i in 1:N
        ldiv_z!(u_out[i], plan, u_in[i])
    end
    return nothing
end

function ldiv_all!(a_out::PencilArray, plan::PlanFFT2D, a_in::PencilArray)
    a_out_x = plan.a_tmpx
    a_out_y = plan.a_tmpy
    ldiv!(parent(a_out_y), plan.plan_y, parent(a_in))
    transpose!(a_out_x, a_out_y)
    ldiv!(parent(a_out), plan.plan_x, parent(a_out_x))
    return nothing
end

function ldiv_all!(a_out::PencilArray, plan::PlanFFT3D, a_in::PencilArray)
    a_out_x = plan.a_tmpx
    a_out_y = plan.a_tmpy
    a_out_y2 = plan.a_tmp2y
    a_out_z = plan.a_tmpz
    ldiv!(parent(a_out_z), plan.plan_z, parent(a_in))
    transpose!(a_out_y, a_out_z)
    ldiv!(parent(a_out_y2), plan.plan_y, parent(a_out_y))
    transpose!(a_out_x, a_out_y2)
    ldiv!(parent(a_out), plan.plan_x, parent(a_out_x))
    return nothing
end

function ldiv_all!(u_out::Vector{<:AbstractArray}, plan::AbstractFFTPlan{N},
                   u_in::Vector{<:AbstractArray}) where {N}
    for i in 1:N
        ldiv_all!(u_out[i], plan, u_in[i])
    end
    return nothing
end

function grid_x(plan::PlanFFT2D)
    grid = localgrid(plan.pen_x, (plan.f.g.x, plan.f.g.y))
    x, y = grid.x, grid.y
    gridξ = localgrid(plan.pen_x, (plan.ξx, plan.ξy))
    ξx, ξy = gridξ.x, gridξ.y
    return x, y, ξx, ξy
end

function grid_y(plan::PlanFFT2D)
    grid = localgrid(plan.pen_y, (plan.f.g.x, plan.f.g.y))
    x, y = grid.x, grid.y
    gridξ = localgrid(plan.pen_y, (plan.ξx, plan.ξy))
    ξx, ξy = gridξ.x, gridξ.y
    return x, y, ξx, ξy
end

function grid_x(plan::PlanFFT3D)
    grid = localgrid(plan.pen_x, (plan.f.g.x, plan.f.g.y, plan.f.g.z))
    x, y, z = grid.x, grid.y, grid.z
    gridξ = localgrid(plan.pen_x, (plan.ξx, plan.ξy, plan.ξz))
    ξx, ξy, ξz = gridξ.x, gridξ.y, gridξ.z
    return x, y, z, ξx, ξy, ξz
end

function grid_y(plan::PlanFFT3D)
    grid = localgrid(plan.pen_y, (plan.f.g.x, plan.f.g.y, plan.f.g.z))
    x, y, z = grid.x, grid.y, grid.z
    gridξ = localgrid(plan.pen_y, (plan.ξx, plan.ξy, plan.ξz))
    ξx, ξy, ξz = gridξ.x, gridξ.y, gridξ.z
    return x, y, z, ξx, ξy, ξz
end

function grid_z(plan::PlanFFT3D)
    grid = localgrid(plan.pen_z, (plan.f.g.x, plan.f.g.y, plan.f.g.z))
    x, y, z = grid.x, grid.y, grid.z
    gridξ = localgrid(plan.pen_z, (plan.ξx, plan.ξy, plan.ξz))
    ξx, ξy, ξz = gridξ.x, gridξ.y, gridξ.z
    return x, y, z, ξx, ξy, ξz
end
