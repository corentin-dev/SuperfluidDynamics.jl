# =============================================================================
# CUDA backend for the 6th-order periodic compact scheme
#
# On GPU the compact relation is solved with a dedicated Thomas kernel instead
# of the spectral (Fourier-multiplier) path: one thread per grid line performs
# the stencil, the Thomas sweep, the cyclic correction and the back
# substitution.
#
# Kernels solve along the leading (fastest) dimension of a 2D `(n, L)` or 3D
# `(n, L2, L3)` raw array; the y/z directions are handled with PencilArray
# transposes that bring the derivative axis to the front, mirroring the CPU
# dispatch in `derivatives.jl`.
#
# Real and complex fields both work here.
#
# This file is included from `Plans.jl` inside `try using CUDA ... catch
# CUDA = nothing end`: when CUDA cannot be loaded, none of it is defined and
# the compact plan does not offer the GPU kernel backend.
# =============================================================================

@inline function _compact_cu_solve!(r, n, j, s, w, f, t, α, denom)
    @inbounds begin
        for i in 2:n
            r[i, j] -= r[i - 1, j] * s[i]
        end
        r[n, j] *= w[n]
        for i in (n - 1):-1:1
            r[i, j] = (r[i, j] - f[i] * r[i + 1, j]) * w[i]
        end
        sx = (r[1, j] - α * r[n, j]) / denom
        for i in 1:n
            r[i, j] -= sx * t[i]
        end
    end
    return
end

@inline function _compact_cu_solve3!(r, n, j, k, s, w, f, t, α, denom)
    @inbounds begin
        for i in 2:n
            r[i, j, k] -= r[i - 1, j, k] * s[i]
        end
        r[n, j, k] *= w[n]
        for i in (n - 1):-1:1
            r[i, j, k] = (r[i, j, k] - f[i] * r[i + 1, j, k]) * w[i]
        end
        sx = (r[1, j, k] - α * r[n, j, k]) / denom
        for i in 1:n
            r[i, j, k] -= sx * t[i]
        end
    end
    return
end

# --- 2D kernels (array layout (n, L), line j is processed by thread j) ------

function _compact_cu_1k!(u, r, n, a, b, s, w, f, t, α, denom, L)
    j = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    j > L && return
    @inbounds begin
        r[1, j] = a * (u[2, j] - u[n, j]) + b * (u[3, j] - u[n - 1, j])
        r[2, j] = a * (u[3, j] - u[1, j]) + b * (u[4, j] - u[n, j])
        for i in 3:(n - 2)
            r[i, j] = a * (u[i + 1, j] - u[i - 1, j]) + b * (u[i + 2, j] - u[i - 2, j])
        end
        r[n - 1, j] = a * (u[n, j] - u[n - 2, j]) + b * (u[1, j] - u[n - 3, j])
        r[n, j] = a * (u[1, j] - u[n - 1, j]) + b * (u[2, j] - u[n - 2, j])
    end
    _compact_cu_solve!(r, n, j, s, w, f, t, α, denom)
    return
end

function _compact_cu_2k!(u, r, n, a, b, s, w, f, t, α, denom, L)
    j = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    j > L && return
    @inbounds begin
        r[1, j] = a * (u[2, j] - 2u[1, j] + u[n, j]) + b * (u[3, j] - 2u[1, j] + u[n - 1, j])
        r[2, j] = a * (u[3, j] - 2u[2, j] + u[1, j]) + b * (u[4, j] - 2u[2, j] + u[n, j])
        for i in 3:(n - 2)
            r[i, j] = a * (u[i + 1, j] - 2u[i, j] + u[i - 1, j]) + b * (u[i + 2, j] - 2u[i, j] + u[i - 2, j])
        end
        r[n - 1, j] = a * (u[n, j] - 2u[n - 1, j] + u[n - 2, j]) + b * (u[1, j] - 2u[n - 1, j] + u[n - 3, j])
        r[n, j] = a * (u[1, j] - 2u[n, j] + u[n - 1, j]) + b * (u[2, j] - 2u[n, j] + u[n - 2, j])
    end
    _compact_cu_solve!(r, n, j, s, w, f, t, α, denom)
    return
end

# --- 3D kernels (array layout (n, L2, L3), one thread per (j, k) line) -------

function _compact_cu_1k3!(u, r, n, a, b, s, w, f, t, α, denom, L2, L3)
    idx = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    L = L2 * L3
    idx > L && return
    j = (idx - 1) % L2 + 1
    k = (idx - 1) ÷ L2 + 1
    @inbounds begin
        r[1, j, k] = a * (u[2, j, k] - u[n, j, k]) + b * (u[3, j, k] - u[n - 1, j, k])
        r[2, j, k] = a * (u[3, j, k] - u[1, j, k]) + b * (u[4, j, k] - u[n, j, k])
        for i in 3:(n - 2)
            r[i, j, k] = a * (u[i + 1, j, k] - u[i - 1, j, k]) + b * (u[i + 2, j, k] - u[i - 2, j, k])
        end
        r[n - 1, j, k] = a * (u[n, j, k] - u[n - 2, j, k]) + b * (u[1, j, k] - u[n - 3, j, k])
        r[n, j, k] = a * (u[1, j, k] - u[n - 1, j, k]) + b * (u[2, j, k] - u[n - 2, j, k])
    end
    _compact_cu_solve3!(r, n, j, k, s, w, f, t, α, denom)
    return
end

function _compact_cu_2k3!(u, r, n, a, b, s, w, f, t, α, denom, L2, L3)
    idx = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    L = L2 * L3
    idx > L && return
    j = (idx - 1) % L2 + 1
    k = (idx - 1) ÷ L2 + 1
    @inbounds begin
        r[1, j, k] = a * (u[2, j, k] - 2u[1, j, k] + u[n, j, k]) + b * (u[3, j, k] - 2u[1, j, k] + u[n - 1, j, k])
        r[2, j, k] = a * (u[3, j, k] - 2u[2, j, k] + u[1, j, k]) + b * (u[4, j, k] - 2u[2, j, k] + u[n, j, k])
        for i in 3:(n - 2)
            r[i, j, k] = a * (u[i + 1, j, k] - 2u[i, j, k] + u[i - 1, j, k]) + b * (u[i + 2, j, k] - 2u[i, j, k] + u[i - 2, j, k])
        end
        r[n - 1, j, k] = a * (u[n, j, k] - 2u[n - 1, j, k] + u[n - 2, j, k]) + b * (u[1, j, k] - 2u[n - 1, j, k] + u[n - 3, j, k])
        r[n, j, k] = a * (u[1, j, k] - 2u[n, j, k] + u[n - 1, j, k]) + b * (u[2, j, k] - 2u[n, j, k] + u[n - 2, j, k])
    end
    _compact_cu_solve3!(r, n, j, k, s, w, f, t, α, denom)
    return
end

# --- line drivers -------------------------------------------------------------
# One thread per grid line.

function _compact_cu_1line_per!(r, u, n, c)
    L = size(u, 2)
    threads = 128
    @cuda threads=threads blocks=(L + threads - 1) ÷ threads _compact_cu_1k!(u, r, n, c.a, c.b, c.s, c.w, c.f, c.t, c.alpha, c.denom, L)
    return
end

function _compact_cu_2line_per!(r, u, n, c)
    L = size(u, 2)
    threads = 128
    @cuda threads=threads blocks=(L + threads - 1) ÷ threads _compact_cu_2k!(u, r, n, c.a, c.b, c.s, c.w, c.f, c.t, c.alpha, c.denom, L)
    return
end

function _compact_cu_1line3_per!(r, u, n, c)
    L2, L3 = size(u, 2), size(u, 3)
    L = L2 * L3
    threads = 128
    @cuda threads=threads blocks=(L + threads - 1) ÷ threads _compact_cu_1k3!(u, r, n, c.a, c.b, c.s, c.w, c.f, c.t, c.alpha, c.denom, L2, L3)
    return
end

function _compact_cu_2line3_per!(r, u, n, c)
    L2, L3 = size(u, 2), size(u, 3)
    L = L2 * L3
    threads = 128
    @cuda threads=threads blocks=(L + threads - 1) ÷ threads _compact_cu_2k3!(u, r, n, c.a, c.b, c.s, c.w, c.f, c.t, c.alpha, c.denom, L2, L3)
    return
end

# Device-side per-axis compact multipliers: the same precomputed Thomas data
# as the CPU `CompactAxis`, with the banded vectors uploaded to the device so
# they can be read directly from the kernels.
# Both the periodic (`CompactAxisGPU`) and the non-periodic / Dirichlet
# (`CompactAxisGPU_np`) variants share this abstract type so a GPU plan can
# mix periodic and bounded axes, and the line drivers dispatch on it.
abstract type AbstractCompactAxisGPU end

struct CompactAxisGPU <: AbstractCompactAxisGPU
    "number of grid points along the axis."
    n::Int
    "compact coefficient: 1/3 (1st order) or 2/11 (2nd order)."
    alpha::Float64
    "stencil coefficient a."
    a::Float64
    "stencil coefficient b."
    b::Float64
    "Thomas forward multipliers (device)."
    s::CuArray{Float64, 1}
    "reciprocals of the Thomas pivots (device)."
    w::CuArray{Float64, 1}
    "super-diagonal of the (unchanged) tridiagonal (device)."
    f::CuArray{Float64, 1}
    "precomputed cyclic correction vector (device)."
    t::CuArray{Float64, 1}
    "cyclic correction denominator."
    denom::Float64
end

"""$(TYPEDSIGNATURES)

Build the device-side compact multipliers for one axis by reusing the CPU
[`compact_setup`](@ref) (guaranteeing identical coefficients) and uploading
the banded vectors to the GPU.
"""
function compact_setup_gpu(n::Int, Δ::Real, order::Int)
    c = compact_setup(n, Δ, order)
    return CompactAxisGPU(n, c.alpha, c.a, c.b,
                          CuArray(c.s), CuArray(c.w), CuArray(c.f), CuArray(c.t), c.denom)
end

# =============================================================================
# Non-periodic (homogeneous Dirichlet) CUDA kernels
#
# Plain tridiagonal Thomas (no cyclic correction), one thread per line, with the
# one-sided boundary stencils and the relaxed rows (i=2, i=n-1) taken from the
# GPS cdl==2 operator. The row-varying sub/super diagonals and the Thomas
# factors are uploaded once; the kernel does the per-call stencil + the two
# Thomas sweeps.
# =============================================================================

@inline function _compact_cu_solve_np2!(r, n, j, s, w, sup)
    @inbounds begin
        for i in 2:n
            r[i, j] -= r[i - 1, j] * s[i]
        end
        r[n, j] *= w[n]
        for i in (n - 1):-1:1
            r[i, j] = (r[i, j] - sup[i] * r[i + 1, j]) * w[i]
        end
    end
    return
end

@inline function _compact_cu_solve_np3!(r, n, j, k, s, w, sup)
    @inbounds begin
        for i in 2:n
            r[i, j, k] -= r[i - 1, j, k] * s[i]
        end
        r[n, j, k] *= w[n]
        for i in (n - 1):-1:1
            r[i, j, k] = (r[i, j, k] - sup[i] * r[i + 1, k]) * w[i]
        end
    end
    return
end

function _compact_cu_1k_np!(u, r, n, a, b, s, w, sup, af1, bf1, cf1, af2, afn, bfn, cfn, L)
    j = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    j > L && return
    @inbounds begin
        r[1, j] = af1 * u[1, j] + bf1 * u[2, j] + cf1 * u[3, j]
        r[2, j] = af2 * (u[3, j] - u[1, j])
        for i in 3:(n - 2)
            r[i, j] = a * (u[i + 1, j] - u[i - 1, j]) + b * (u[i + 2, j] - u[i - 2, j])
        end
        r[n - 1, j] = af2 * (u[n, j] - u[n - 2, j])
        r[n, j] = -afn * u[n, j] - bfn * u[n - 1, j] - cfn * u[n - 2, j]
    end
    _compact_cu_solve_np2!(r, n, j, s, w, sup)
    return
end

function _compact_cu_2k_np!(u, r, n, a, b, s, w, sup, as1, bs1, cs1, ds1, as2, asn, bsn, csn, dsn, L)
    j = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    j > L && return
    @inbounds begin
        r[1, j] = as1 * u[1, j] + bs1 * u[2, j] + cs1 * u[3, j] + ds1 * u[4, j]
        r[2, j] = as2 * (u[3, j] - 2u[2, j] + u[1, j])
        for i in 3:(n - 2)
            r[i, j] = a * (u[i + 1, j] - 2u[i, j] + u[i - 1, j]) + b * (u[i + 2, j] - 2u[i, j] + u[i - 2, j])
        end
        r[n - 1, j] = as2 * (u[n, j] - 2u[n - 1, j] + u[n - 2, j])
        r[n, j] = asn * u[n, j] + bsn * u[n - 1, j] + csn * u[n - 2, j] + dsn * u[n - 3, j]
    end
    _compact_cu_solve_np2!(r, n, j, s, w, sup)
    return
end

# --- Neumann (even-mirror) kernels ------------------------------------------
# The right-hand side is the interior stencil on the even-mirrored field (no
# one-sided weights); the solve is the plain Thomas sweep with the Neumann
# LHS factors. Only `a`, `b` and the banded `s`/`w`/`sup` are per axis.

function _compact_cu_1k_neu!(u, r, n, a, b, s, w, sup, L)
    j = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    j > L && return
    @inbounds begin
        r[1, j] = zero(eltype(r))
        r[2, j] = a * (u[3, j] - u[1, j]) + b * (u[4, j] - u[2, j])
        for i in 3:(n - 2)
            r[i, j] = a * (u[i + 1, j] - u[i - 1, j]) + b * (u[i + 2, j] - u[i - 2, j])
        end
        r[n - 1, j] = a * (u[n, j] - u[n - 2, j]) + b * (u[n - 1, j] - u[n - 3, j])
        r[n, j] = zero(eltype(r))
    end
    _compact_cu_solve_np2!(r, n, j, s, w, sup)
    return
end

function _compact_cu_2k_neu!(u, r, n, a, b, s, w, sup, L)
    j = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    j > L && return
    @inbounds begin
        r[1, j] = 2a * (u[2, j] - u[1, j]) + 2b * (u[3, j] - u[1, j])
        r[2, j] = a * (u[1, j] - 2u[2, j] + u[3, j]) + b * (u[4, j] - u[2, j])
        for i in 3:(n - 2)
            r[i, j] = a * (u[i + 1, j] - 2u[i, j] + u[i - 1, j]) + b * (u[i + 2, j] - 2u[i, j] + u[i - 2, j])
        end
        r[n - 1, j] = a * (u[n, j] - 2u[n - 1, j] + u[n - 2, j]) + b * (u[n - 3, j] - u[n - 1, j])
        r[n, j] = 2a * (u[n - 1, j] - u[n, j]) + 2b * (u[n - 2, j] - u[n, j])
    end
    _compact_cu_solve_np2!(r, n, j, s, w, sup)
    return
end

function _compact_cu_1k3_neu!(u, r, n, a, b, s, w, sup, L2, L3)
    idx = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    L = L2 * L3
    idx > L && return
    j = (idx - 1) % L2 + 1
    k = (idx - 1) ÷ L2 + 1
    @inbounds begin
        r[1, j, k] = zero(eltype(r))
        r[2, j, k] = a * (u[3, j, k] - u[1, j, k]) + b * (u[4, j, k] - u[2, j, k])
        for i in 3:(n - 2)
            r[i, j, k] = a * (u[i + 1, j, k] - u[i - 1, j, k]) + b * (u[i + 2, j, k] - u[i - 2, j, k])
        end
        r[n - 1, j, k] = a * (u[n, j, k] - u[n - 2, j, k]) + b * (u[n - 1, j, k] - u[n - 3, j, k])
        r[n, j, k] = zero(eltype(r))
    end
    _compact_cu_solve_np3!(r, n, j, k, s, w, sup)
    return
end

function _compact_cu_2k3_neu!(u, r, n, a, b, s, w, sup, L2, L3)
    idx = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    L = L2 * L3
    idx > L && return
    j = (idx - 1) % L2 + 1
    k = (idx - 1) ÷ L2 + 1
    @inbounds begin
        r[1, j, k] = 2a * (u[2, j, k] - u[1, j, k]) + 2b * (u[3, j, k] - u[1, j, k])
        r[2, j, k] = a * (u[1, j, k] - 2u[2, j, k] + u[3, j, k]) + b * (u[4, j, k] - u[2, j, k])
        for i in 3:(n - 2)
            r[i, j, k] = a * (u[i + 1, j, k] - 2u[i, j, k] + u[i - 1, j, k]) + b * (u[i + 2, j, k] - 2u[i, j, k] + u[i - 2, j, k])
        end
        r[n - 1, j, k] = a * (u[n, j, k] - 2u[n - 1, j, k] + u[n - 2, j, k]) + b * (u[n - 3, j, k] - u[n - 1, j, k])
        r[n, j, k] = 2a * (u[n - 1, j, k] - u[n, j, k]) + 2b * (u[n - 2, j, k] - u[n, j, k])
    end
    _compact_cu_solve_np3!(r, n, j, k, s, w, sup)
    return
end

function _compact_cu_1k3_np!(u, r, n, a, b, s, w, sup, af1, bf1, cf1, af2, afn, bfn, cfn, L2, L3)
    idx = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    L = L2 * L3
    idx > L && return
    j = (idx - 1) % L2 + 1
    k = (idx - 1) ÷ L2 + 1
    @inbounds begin
        r[1, j, k] = af1 * u[1, j, k] + bf1 * u[2, j, k] + cf1 * u[3, j, k]
        r[2, j, k] = af2 * (u[3, j, k] - u[1, j, k])
        for i in 3:(n - 2)
            r[i, j, k] = a * (u[i + 1, j, k] - u[i - 1, j, k]) + b * (u[i + 2, j, k] - u[i - 2, j, k])
        end
        r[n - 1, j, k] = af2 * (u[n, j, k] - u[n - 2, j, k])
        r[n, j, k] = -afn * u[n, j, k] - bfn * u[n - 1, j, k] - cfn * u[n - 2, j, k]
    end
    _compact_cu_solve_np3!(r, n, j, k, s, w, sup)
    return
end

function _compact_cu_2k3_np!(u, r, n, a, b, s, w, sup, as1, bs1, cs1, ds1, as2, asn, bsn, csn, dsn, L2, L3)
    idx = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    L = L2 * L3
    idx > L && return
    j = (idx - 1) % L2 + 1
    k = (idx - 1) ÷ L2 + 1
    @inbounds begin
        r[1, j, k] = as1 * u[1, j, k] + bs1 * u[2, j, k] + cs1 * u[3, j, k] + ds1 * u[4, j, k]
        r[2, j, k] = as2 * (u[3, j, k] - 2u[2, j, k] + u[1, j, k])
        for i in 3:(n - 2)
            r[i, j, k] = a * (u[i + 1, j, k] - 2u[i, j, k] + u[i - 1, j, k]) + b * (u[i + 2, j, k] - 2u[i, j, k] + u[i - 2, j, k])
        end
        r[n - 1, j, k] = as2 * (u[n, j, k] - 2u[n - 1, j, k] + u[n - 2, j, k])
        r[n, j, k] = asn * u[n, j, k] + bsn * u[n - 1, j, k] + csn * u[n - 2, j, k] + dsn * u[n - 3, j, k]
    end
    _compact_cu_solve_np3!(r, n, j, k, s, w, sup)
    return
end

# --- NP line drivers (dispatch on the axis type: NP vs periodic) ------------

# Device-side per-axis non-periodic compact multipliers (Dirichlet).
struct CompactAxisGPU_np <: AbstractCompactAxisGPU    "number of grid points along the axis."
    n::Int
    "interior stencil coefficient a."
    a::Float64
    "interior stencil coefficient b."
    b::Float64
    "Thomas forward multipliers (device)."
    s::CuArray{Float64, 1}
    "reciprocals of the Thomas pivots (device)."
    w::CuArray{Float64, 1}
    "tridiagonal super-diagonal (device)."
    sup::CuArray{Float64, 1}
    "1st-derivative one-sided / relaxed closure weights."
    af1::Float64
    bf1::Float64
    cf1::Float64
    af2::Float64
    afn::Float64
    bfn::Float64
    cfn::Float64
    "2nd-derivative one-sided / relaxed closure weights."
    as1::Float64
    bs1::Float64
    cs1::Float64
    ds1::Float64
    as2::Float64
    asn::Float64
    bsn::Float64
    csn::Float64
    dsn::Float64
end

@inline function _compact_cu_1line_np!(r, u, n, c::CompactAxisGPU_np)
    L = size(u, 2)
    threads = 128
    @cuda threads=threads blocks=(L + threads - 1) ÷ threads _compact_cu_1k_np!(u, r, n, c.a, c.b, c.s, c.w, c.sup, c.af1, c.bf1, c.cf1, c.af2, c.afn, c.bfn, c.cfn, L)
    return
end

@inline function _compact_cu_2line_np!(r, u, n, c::CompactAxisGPU_np)
    L = size(u, 2)
    threads = 128
    @cuda threads=threads blocks=(L + threads - 1) ÷ threads _compact_cu_2k_np!(u, r, n, c.a, c.b, c.s, c.w, c.sup, c.as1, c.bs1, c.cs1, c.ds1, c.as2, c.asn, c.bsn, c.csn, c.dsn, L)
    return
end

@inline function _compact_cu_1line3_np!(r, u, n, c::CompactAxisGPU_np)
    L2, L3 = size(u, 2), size(u, 3)
    L = L2 * L3
    threads = 128
    @cuda threads=threads blocks=(L + threads - 1) ÷ threads _compact_cu_1k3_np!(u, r, n, c.a, c.b, c.s, c.w, c.sup, c.af1, c.bf1, c.cf1, c.af2, c.afn, c.bfn, c.cfn, L2, L3)
    return
end

@inline function _compact_cu_2line3_np!(r, u, n, c::CompactAxisGPU_np)
    L2, L3 = size(u, 2), size(u, 3)
    L = L2 * L3
    threads = 128
    @cuda threads=threads blocks=(L + threads - 1) ÷ threads _compact_cu_2k3_np!(u, r, n, c.a, c.b, c.s, c.w, c.sup, c.as1, c.bs1, c.cs1, c.ds1, c.as2, c.asn, c.bsn, c.csn, c.dsn, L2, L3)
    return
end

"""$(TYPEDSIGNATURES)

Build the device-side non-periodic (Dirichlet) compact multipliers for one axis
by reusing the CPU [`compact_setup_np`](@ref) (identical coefficients) and
uploading the banded vectors to the GPU.
"""
function compact_setup_np_gpu(n::Int, Δ::Real, order::Int)
    c = compact_setup_np(n, Δ, order)
    return CompactAxisGPU_np(n, c.a, c.b,
                             CuArray(c.s), CuArray(c.w), CuArray(c.sup),
                             c.af1, c.bf1, c.cf1, c.af2, c.afn, c.bfn, c.cfn,
                             c.as1, c.bs1, c.cs1, c.ds1, c.as2, c.asn, c.bsn, c.csn, c.dsn)
end

# Device-side per-axis non-periodic compact multipliers (Neumann, even mirror).
struct CompactAxisGPU_neu <: AbstractCompactAxisGPU
    "number of grid points along the axis."
    n::Int
    "interior stencil coefficient a."
    a::Float64
    "interior stencil coefficient b."
    b::Float64
    "Thomas forward multipliers (device)."
    s::CuArray{Float64, 1}
    "reciprocals of the Thomas pivots (device)."
    w::CuArray{Float64, 1}
    "tridiagonal super-diagonal (device)."
    sup::CuArray{Float64, 1}
end

@inline function _compact_cu_1line_neu!(r, u, n, c::CompactAxisGPU_neu)
    L = size(u, 2)
    threads = 128
    @cuda threads=threads blocks=(L + threads - 1) ÷ threads _compact_cu_1k_neu!(u, r, n, c.a, c.b, c.s, c.w, c.sup, L)
    return
end

@inline function _compact_cu_2line_neu!(r, u, n, c::CompactAxisGPU_neu)
    L = size(u, 2)
    threads = 128
    @cuda threads=threads blocks=(L + threads - 1) ÷ threads _compact_cu_2k_neu!(u, r, n, c.a, c.b, c.s, c.w, c.sup, L)
    return
end

@inline function _compact_cu_1line3_neu!(r, u, n, c::CompactAxisGPU_neu)
    L2, L3 = size(u, 2), size(u, 3)
    L = L2 * L3
    threads = 128
    @cuda threads=threads blocks=(L + threads - 1) ÷ threads _compact_cu_1k3_neu!(u, r, n, c.a, c.b, c.s, c.w, c.sup, L2, L3)
    return
end

@inline function _compact_cu_2line3_neu!(r, u, n, c::CompactAxisGPU_neu)
    L2, L3 = size(u, 2), size(u, 3)
    L = L2 * L3
    threads = 128
    @cuda threads=threads blocks=(L + threads - 1) ÷ threads _compact_cu_2k3_neu!(u, r, n, c.a, c.b, c.s, c.w, c.sup, L2, L3)
    return
end

"""$(TYPEDSIGNATURES)

Build the device-side non-periodic (Neumann) compact multipliers for one axis
by reusing the CPU [`compact_setup_neu`](@ref) (identical coefficients) and
uploading the banded vectors to the GPU.
"""
function compact_setup_neu_gpu(n::Int, Δ::Real, order::Int)
    c = compact_setup_neu(n, Δ, order)
    return CompactAxisGPU_neu(n, c.a, c.b,
                              CuArray(c.s), CuArray(c.w), CuArray(c.sup))
end

# --- generic line drivers: dispatch on the axis type (periodic vs Dirichlet) --
# The GPU `computeDerivatives!` dispatch calls these with a per-axis object, so
# a single plan can mix periodic and bounded axes; the right kernel (periodic
# Thomas + cyclic correction, or plain Thomas) is selected by the axis type.
function _compact_cu_1line!(r, u, n, c::AbstractCompactAxisGPU)
    if c isa CompactAxisGPU
        return _compact_cu_1line_per!(r, u, n, c)
    elseif c isa CompactAxisGPU_neu
        return _compact_cu_1line_neu!(r, u, n, c)
    else
        return _compact_cu_1line_np!(r, u, n, c)
    end
end

function _compact_cu_2line!(r, u, n, c::AbstractCompactAxisGPU)
    if c isa CompactAxisGPU
        return _compact_cu_2line_per!(r, u, n, c)
    elseif c isa CompactAxisGPU_neu
        return _compact_cu_2line_neu!(r, u, n, c)
    else
        return _compact_cu_2line_np!(r, u, n, c)
    end
end

function _compact_cu_1line3!(r, u, n, c::AbstractCompactAxisGPU)
    if c isa CompactAxisGPU
        return _compact_cu_1line3_per!(r, u, n, c)
    elseif c isa CompactAxisGPU_neu
        return _compact_cu_1line3_neu!(r, u, n, c)
    else
        return _compact_cu_1line3_np!(r, u, n, c)
    end
end

function _compact_cu_2line3!(r, u, n, c::AbstractCompactAxisGPU)
    if c isa CompactAxisGPU
        return _compact_cu_2line3_per!(r, u, n, c)
    elseif c isa CompactAxisGPU_neu
        return _compact_cu_2line3_neu!(r, u, n, c)
    else
        return _compact_cu_2line3_np!(r, u, n, c)
    end
end
