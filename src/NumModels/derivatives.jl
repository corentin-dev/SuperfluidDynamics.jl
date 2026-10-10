# FFT

## 2D
function computeDerivatives!(gf::GradientField2D, p::AbstractFFTPlan, ϕt::AbstractArray)
    # FFT x
    # frequencies
    x, y, ξx, ξy = grid_x(p)
    mul_x!(p.ϕx_hat, p, ϕt)
    # compute dx
    p.a_tmp2x .= im .* ξx .* p.ϕx_hat
    ldiv_x!(gf.dx, p, p.a_tmp2x)
    # compute ddx
    p.a_tmp2x .= -ξx .^ 2 .* p.ϕx_hat
    ldiv_x!(gf.ddx, p, p.a_tmp2x)
    # FFT y
    x, y, ξx, ξy = grid_y(p)
    mul_y!(p.ϕy_hat, p, ϕt)
    # compute dy
    p.a_tmp2y .= im .* ξy .* p.ϕy_hat
    ldiv_y!(gf.dy, p, p.a_tmp2y)
    # compute ddy
    p.a_tmp2y .= -ξy .^ 2 .* p.ϕy_hat
    ldiv_y!(gf.ddy, p, p.a_tmp2y)
    return nothing
end

function computeDerivatives!(gf::GradientRotField2D, p::AbstractFFTPlan, ϕt::AbstractArray)
    # FFT x
    # frequencies
    x, y, ξx, ξy = grid_x(p)
    mul_x!(p.ϕx_hat, p, ϕt)
    # compute dx
    p.a_tmp2x .= im .* ξx .* p.ϕx_hat
    ldiv_x!(gf.dx, p, p.a_tmp2x)
    # compute rx
    p.a_tmp2x .= im .* y .* ξx .* p.ϕx_hat
    ldiv_x!(gf.rx, p, p.a_tmp2x)
    # compute ddx
    p.a_tmp2x .= -ξx .^ 2 .* p.ϕx_hat
    ldiv_x!(gf.ddx, p, p.a_tmp2x)
    # FFT y
    x, y, ξx, ξy = grid_y(p)
    mul_y!(p.ϕy_hat, p, ϕt)
    # compute dy
    p.a_tmp2y .= im .* ξy .* p.ϕy_hat
    ldiv_y!(gf.dy, p, p.a_tmp2y)
    # compute ry
    p.a_tmp2y .= -im .* x .* ξy .* p.ϕy_hat
    ldiv_y!(gf.ry, p, p.a_tmp2y)
    # compute ddy
    p.a_tmp2y .= -ξy .^ 2 .* p.ϕy_hat
    ldiv_y!(gf.ddy, p, p.a_tmp2y)
    return nothing
end

## 3D

function computeDerivatives!(gf::GradientField3D, p::AbstractFFTPlan, ϕt::AbstractArray)
    # FFT x
    # frequencies
    x, y, z, ξx, ξy, ξz = grid_x(p)
    mul_x!(p.ϕx_hat, p, ϕt)
    # compute dx
    p.a_tmp2x .= im .* ξx .* p.ϕx_hat
    ldiv_x!(gf.dx, p, p.a_tmp2x)
    # compute ddx
    p.a_tmp2x .= -ξx .^ 2 .* p.ϕx_hat
    ldiv_x!(gf.ddx, p, p.a_tmp2x)
    # FFT y
    x, y, z, ξx, ξy, ξz = grid_y(p)
    mul_y!(p.ϕy_hat, p, ϕt)
    # compute dy
    p.a_tmp2y .= im .* ξy .* p.ϕy_hat
    ldiv_y!(gf.dy, p, p.a_tmp2y)
    # compute ddy
    p.a_tmp2y .= -ξy .^ 2 .* p.ϕy_hat
    ldiv_y!(gf.ddy, p, p.a_tmp2y)
    # FFT z
    x, y, z, ξx, ξy, ξz = grid_z(p)
    mul_z!(p.ϕz_hat, p, ϕt)
    # compute dz
    p.a_tmp2z .= im .* ξz .* p.ϕz_hat
    ldiv_z!(gf.dz, p, p.a_tmp2z)
    # compute ddz
    p.a_tmp2z .= -ξz .^ 2 .* p.ϕz_hat
    ldiv_z!(gf.ddz, p, p.a_tmp2z)
    return nothing
end

function computeDerivatives!(gf::GradientRotField3D, p::AbstractFFTPlan, ϕt::AbstractArray)
    # FFT x
    # frequencies
    x, y, z, ξx, ξy, ξz = grid_x(p)
    mul_x!(p.ϕx_hat, p, ϕt)
    # compute dx
    p.a_tmp2x .= im .* ξx .* p.ϕx_hat
    ldiv_x!(gf.dx, p, p.a_tmp2x)
    # compute rx
    p.a_tmp2x .= im .* y .* ξx .* p.ϕx_hat
    ldiv_x!(gf.rx, p, p.a_tmp2x)
    # compute ddx
    p.a_tmp2x .= -ξx .^ 2 .* p.ϕx_hat
    ldiv_x!(gf.ddx, p, p.a_tmp2x)
    # FFT y
    x, y, z, ξx, ξy, ξz = grid_y(p)
    mul_y!(p.ϕy_hat, p, ϕt)
    # compute dy
    p.a_tmp2y .= im .* ξy .* p.ϕy_hat
    ldiv_y!(gf.dy, p, p.a_tmp2y)
    # compute ry
    p.a_tmp2y .= -im .* x .* ξy .* p.ϕy_hat
    ldiv_y!(gf.ry, p, p.a_tmp2y)
    # compute ddy
    p.a_tmp2y .= -ξy .^ 2 .* p.ϕy_hat
    ldiv_y!(gf.ddy, p, p.a_tmp2y)
    # FFT z
    x, y, z, ξx, ξy, ξz = grid_z(p)
    mul_z!(p.ϕz_hat, p, ϕt)
    # compute dz
    p.a_tmp2z .= im .* ξz .* p.ϕz_hat
    ldiv_z!(gf.dz, p, p.a_tmp2z)
    # compute ddz
    p.a_tmp2z .= -ξz .^ 2 .* p.ϕz_hat
    ldiv_z!(gf.ddz, p, p.a_tmp2z)
    return nothing
end

function computeDerivatives!(gf::GradientCurlField3D, p::AbstractFFTPlan, ut::AbstractArray)
    # get ̂u
    mul_all!(p.uz_hat, p, ut)
    # new field for ̂ω
    ω_hat = similar_data(p.uz_hat)
    x, y, z, ξx, ξy, ξz = grid_z(p)
    # ̂ω  = ∇ × ̂u
    @. ω_hat[1] = im * (ξy * p.uz_hat[3] - ξz * p.uz_hat[2])
    @. ω_hat[2] = im * (ξz * p.uz_hat[1] - ξx * p.uz_hat[3])
    @. ω_hat[3] = im * (ξx * p.uz_hat[2] - ξy * p.uz_hat[1])
    # get ω with iFFT
    ldiv_all!(gf.ω, p, ω_hat)
    return nothing
end

# Finite Difference

## 2D
function computeDerivatives!(gf::GradientField2D, p::AbstractFDPlan, ϕt::AbstractArray)
    # x direction
    computedxddx!(parent(ϕt), parent(gf.dx), parent(gf.ddx), p.Δx; order=6)
    # y direction
    transpose!(p.ϕytmp, ϕt)
    dϕytmp = similar(p.ϕytmp)
    ddϕytmp = similar(p.ϕytmp)
    computedyddy!(parent(p.ϕytmp), parent(dϕytmp), parent(ddϕytmp), p.Δy; order=6)
    transpose!(gf.dy, dϕytmp)
    transpose!(gf.ddy, ddϕytmp)
    return nothing
end

function computeDerivatives!(gf::GradientRotField2D, p::AbstractFDPlan, ϕt::AbstractArray)
    grid = localgrid(p.pen_x, (p.f.g.x, p.f.g.y))
    x, y = grid.x, grid.y
    # x direction
    computedxddx!(parent(ϕt), parent(gf.dx), parent(gf.ddx), p.Δx; order=6)
    gf.rx .= y .* gf.dx
    # y direction
    transpose!(p.ϕytmp, ϕt)
    dϕytmp = similar(p.ϕytmp)
    ddϕytmp = similar(p.ϕytmp)
    computedyddy!(parent(p.ϕytmp), parent(dϕytmp), parent(ddϕytmp), p.Δy; order=6)
    transpose!(gf.dy, dϕytmp)
    transpose!(gf.ddy, ddϕytmp)
    dϕytmp = nothing
    ddϕytmp = nothing
    # compute (in pen_x)
    gf.ry .= -x .* gf.dy
    return nothing
end

## 3D

function computeDerivatives!(gf::GradientField3D, p::AbstractFDPlan, ϕt::AbstractArray)
    # x direction
    computedxddx!(parent(ϕt), parent(gf.dx), parent(gf.ddx), p.Δx; order=6)
    # y direction
    transpose!(p.ϕytmp, ϕt)
    dϕytmp = similar(p.ϕytmp)
    ddϕytmp = similar(p.ϕytmp)
    computedyddy!(parent(p.ϕytmp), parent(dϕytmp), parent(ddϕytmp), p.Δy; order=6)
    transpose!(gf.dy, dϕytmp)
    transpose!(gf.ddy, ddϕytmp)
    # z direction
    transpose!(p.ϕztmp, p.ϕytmp)
    dϕztmp = similar(p.ϕztmp)
    ddϕztmp = similar(p.ϕztmp)
    computedzddz!(parent(p.ϕztmp), parent(dϕztmp), parent(ddϕztmp), p.Δz; order=6)
    transpose!(dϕytmp, dϕztmp)
    transpose!(ddϕytmp, ddϕztmp)
    transpose!(gf.dz, dϕytmp)
    transpose!(gf.ddz, ddϕytmp)
    # deallocate
    dϕztmp = nothing
    ddϕztmp = nothing
    dϕytmp = nothing
    ddϕytmp = nothing
    return nothing
end

function computeDerivatives!(gf::GradientRotField3D, p::AbstractFDPlan, ϕt::AbstractArray)
    # x direction
    grid = localgrid(p.pen_x, (p.f.g.x, p.f.g.y, p.f.g.z))
    x, y = grid.x, grid.y
    computedxddx!(parent(ϕt), parent(gf.dx), parent(gf.ddx), p.Δx; order=6)
    gf.rx .= y .* gf.dx
    # y direction
    grid = localgrid(p.pen_y, (p.f.g.x, p.f.g.y, p.f.g.z))
    x, y = grid.x, grid.y
    transpose!(p.ϕytmp, ϕt)
    dϕytmp = similar(p.ϕytmp)
    ddϕytmp = similar(p.ϕytmp)
    computedyddy!(parent(p.ϕytmp), parent(dϕytmp), parent(ddϕytmp), p.Δy; order=6)
    transpose!(gf.dy, dϕytmp)
    transpose!(gf.ddy, ddϕytmp)
    gf.ry .= -x .* gf.dy
    # z direction
    transpose!(p.ϕztmp, p.ϕytmp)
    dϕztmp = similar(p.ϕztmp)
    ddϕztmp = similar(p.ϕztmp)
    computedzddz!(parent(p.ϕztmp), parent(dϕztmp), parent(ddϕztmp), p.Δz; order=6)
    transpose!(dϕytmp, dϕztmp)
    transpose!(ddϕytmp, ddϕztmp)
    transpose!(gf.dz, dϕytmp)
    transpose!(gf.ddz, ddϕytmp)
    # deallocate
    dϕztmp = nothing
    ddϕztmp = nothing
    dϕytmp = nothing
    ddϕytmp = nothing
    return nothing
end

# Compact Finite Difference
#
# 6th-order periodic compact finite-difference derivatives. Each line along the
# derivative axis is solved as a cyclic tridiagonal system: a Thomas
# factorization of the interior part plus a rank-one cyclic corner correction.
# The correction vector and its denominator depend only on the matrix, not on
# the right-hand side, so they are precomputed in `CompactAxis` and reused on
# every call.
#
# The derivative axis is always the leading (local) dimension, so the kernels
# loop over the remaining index (axes(a_in, 2) for 2D, or a nested pair for 3D)
# and solve each line independently. For `y`/`z` the field is first transposed
# so that the target axis becomes the leading one.

function compact1line!(a_out, a_in, n, c::CompactAxis)
    a, b = c.a, c.b
    α = c.alpha
    for j in axes(a_in, 2)
        r = @view(a_out[:, j])
        u = @view(a_in[:, j])
        @inbounds begin
            r[1] = a * (u[2] - u[n]) + b * (u[3] - u[n - 1])
            r[2] = a * (u[3] - u[1]) + b * (u[4] - u[n])
            for i in 3:(n - 2)
                r[i] = a * (u[i + 1] - u[i - 1]) + b * (u[i + 2] - u[i - 2])
            end
            r[n - 1] = a * (u[n] - u[n - 2]) + b * (u[1] - u[n - 3])
            r[n] = a * (u[1] - u[n - 1]) + b * (u[2] - u[n - 2])
        end
        _compact_solve!(r, n, c)
    end
    return nothing
end

function compact2line!(a_out, a_in, n, c::CompactAxis)
    a, b = c.a, c.b
    for j in axes(a_in, 2)
        r = @view(a_out[:, j])
        u = @view(a_in[:, j])
        @inbounds begin
            r[1] = a * (u[2] - 2u[1] + u[n]) + b * (u[3] - 2u[1] + u[n - 1])
            r[2] = a * (u[3] - 2u[2] + u[1]) + b * (u[4] - 2u[2] + u[n])
            for i in 3:(n - 2)
                r[i] = a * (u[i + 1] - 2u[i] + u[i - 1]) + b * (u[i + 2] - 2u[i] + u[i - 2])
            end
            r[n - 1] = a * (u[n] - 2u[n - 1] + u[n - 2]) + b * (u[1] - 2u[n - 1] + u[n - 3])
            r[n] = a * (u[1] - 2u[n] + u[n - 1]) + b * (u[2] - 2u[n] + u[n - 2])
        end
        _compact_solve!(r, n, c)
    end
    return nothing
end

function compact1line3!(a_out, a_in, n, c::CompactAxis)
    a, b = c.a, c.b
    for k in axes(a_in, 3), j in axes(a_in, 2)
        r = @view(a_out[:, j, k])
        u = @view(a_in[:, j, k])
        @inbounds begin
            r[1] = a * (u[2] - u[n]) + b * (u[3] - u[n - 1])
            r[2] = a * (u[3] - u[1]) + b * (u[4] - u[n])
            for i in 3:(n - 2)
                r[i] = a * (u[i + 1] - u[i - 1]) + b * (u[i + 2] - u[i - 2])
            end
            r[n - 1] = a * (u[n] - u[n - 2]) + b * (u[1] - u[n - 3])
            r[n] = a * (u[1] - u[n - 1]) + b * (u[2] - u[n - 2])
        end
        _compact_solve!(r, n, c)
    end
    return nothing
end

function compact2line3!(a_out, a_in, n, c::CompactAxis)
    a, b = c.a, c.b
    for k in axes(a_in, 3), j in axes(a_in, 2)
        r = @view(a_out[:, j, k])
        u = @view(a_in[:, j, k])
        @inbounds begin
            r[1] = a * (u[2] - 2u[1] + u[n]) + b * (u[3] - 2u[1] + u[n - 1])
            r[2] = a * (u[3] - 2u[2] + u[1]) + b * (u[4] - 2u[2] + u[n])
            for i in 3:(n - 2)
                r[i] = a * (u[i + 1] - 2u[i] + u[i - 1]) + b * (u[i + 2] - 2u[i] + u[i - 2])
            end
            r[n - 1] = a * (u[n] - 2u[n - 1] + u[n - 2]) + b * (u[1] - 2u[n - 1] + u[n - 3])
            r[n] = a * (u[1] - 2u[n] + u[n - 1]) + b * (u[2] - 2u[n] + u[n - 2])
        end
        _compact_solve!(r, n, c)
    end
    return nothing
end

"""$(TYPEDSIGNATURES)

In-place cyclic tridiagonal solve for one compact line `r` (already holding the
stencil right-hand side), using the precomputed multipliers in `c`.
"""
function _compact_solve!(r, n, c::CompactAxis)
    s, w, f = c.s, c.w, c.f
    α = c.alpha
    @inbounds begin
        for i in 2:n
            r[i] -= r[i - 1] * s[i]
        end
        r[n] *= w[n]
        for i in (n - 1):-1:1
            r[i] = (r[i] - f[i] * r[i + 1]) * w[i]
        end
        sx = (r[1] - α * r[n]) / c.denom
        for i in 1:n
            r[i] -= sx * c.t[i]
        end
    end
    return r
end

"""$(TYPEDSIGNATURES)

In-place plain tridiagonal solve for one **non-periodic** compact line `r`
(already holding the stencil right-hand side), using the precomputed Thomas
multipliers in `c`. The LHS is a plain tridiagonal matrix (no cyclic correction).
"""
function _compact_solve_np!(r, n, c::CompactAxisNP)
    s, w, sup = c.s, c.w, c.sup
    @inbounds begin
        for i in 2:n
            r[i] -= r[i - 1] * s[i]
        end
        r[n] *= w[n]
        for i in (n - 1):-1:1
            r[i] = (r[i] - sup[i] * r[i + 1]) * w[i]
        end
    end
    return r
end

# --- Non-periodic (homogeneous Dirichlet) compact line kernels --------------
# Overloads of the `compact*line!` functions, selected by the per-axis type
# (`CompactAxisNP`). Same structure as the periodic kernels, but the two end
# rows use one-sided stencils and the adjacent rows are relaxed (see
# `compact_setup_np`), and the solve is a plain tridiagonal Thomas (no cyclic
# correction). The stencil is dtype-agnostic (real or complex fields). This is
# what makes mixed axes (e.g. x periodic, y bounded) work: the plan dispatch
# calls `compact1line!(…, p.ax)` and Julia picks the periodic or NP method
# from the type of `p.ax`.

function compact1line!(a_out, a_in, n, c::CompactAxisNP)
    a, b = c.a, c.b
    for j in axes(a_in, 2)
        r = @view(a_out[:, j])
        u = @view(a_in[:, j])
        @inbounds begin
            r[1] = c.af1 * u[1] + c.bf1 * u[2] + c.cf1 * u[3]
            r[2] = c.af2 * (u[3] - u[1])
            for i in 3:(n - 2)
                r[i] = a * (u[i + 1] - u[i - 1]) + b * (u[i + 2] - u[i - 2])
            end
            r[n - 1] = c.af2 * (u[n] - u[n - 2])
            r[n] = -c.afn * u[n] - c.bfn * u[n - 1] - c.cfn * u[n - 2]
        end
        _compact_solve_np!(r, n, c)
    end
    return nothing
end

function compact2line!(a_out, a_in, n, c::CompactAxisNP)
    a, b = c.a, c.b
    for j in axes(a_in, 2)
        r = @view(a_out[:, j])
        u = @view(a_in[:, j])
        @inbounds begin
            r[1] = c.as1 * u[1] + c.bs1 * u[2] + c.cs1 * u[3] + c.ds1 * u[4]
            r[2] = c.as2 * (u[3] - 2u[2] + u[1])
            for i in 3:(n - 2)
                r[i] = a * (u[i + 1] - 2u[i] + u[i - 1]) + b * (u[i + 2] - 2u[i] + u[i - 2])
            end
            r[n - 1] = c.as2 * (u[n] - 2u[n - 1] + u[n - 2])
            r[n] = c.asn * u[n] + c.bsn * u[n - 1] + c.csn * u[n - 2] + c.dsn * u[n - 3]
        end
        _compact_solve_np!(r, n, c)
    end
    return nothing
end

function compact1line3!(a_out, a_in, n, c::CompactAxisNP)
    a, b = c.a, c.b
    for k in axes(a_in, 3), j in axes(a_in, 2)
        r = @view(a_out[:, j, k])
        u = @view(a_in[:, j, k])
        @inbounds begin
            r[1] = c.af1 * u[1] + c.bf1 * u[2] + c.cf1 * u[3]
            r[2] = c.af2 * (u[3] - u[1])
            for i in 3:(n - 2)
                r[i] = a * (u[i + 1] - u[i - 1]) + b * (u[i + 2] - u[i - 2])
            end
            r[n - 1] = c.af2 * (u[n] - u[n - 2])
            r[n] = -c.afn * u[n] - c.bfn * u[n - 1] - c.cfn * u[n - 2]
        end
        _compact_solve_np!(r, n, c)
    end
    return nothing
end

function compact2line3!(a_out, a_in, n, c::CompactAxisNP)
    a, b = c.a, c.b
    for k in axes(a_in, 3), j in axes(a_in, 2)
        r = @view(a_out[:, j, k])
        u = @view(a_in[:, j, k])
        @inbounds begin
            r[1] = c.as1 * u[1] + c.bs1 * u[2] + c.cs1 * u[3] + c.ds1 * u[4]
            r[2] = c.as2 * (u[3] - 2u[2] + u[1])
            for i in 3:(n - 2)
                r[i] = a * (u[i + 1] - 2u[i] + u[i - 1]) + b * (u[i + 2] - 2u[i] + u[i - 2])
            end
            r[n - 1] = c.as2 * (u[n] - 2u[n - 1] + u[n - 2])
            r[n] = c.asn * u[n] + c.bsn * u[n - 1] + c.csn * u[n - 2] + c.dsn * u[n - 3]
        end
        _compact_solve_np!(r, n, c)
    end
    return nothing
end

# --- Non-periodic (homogeneous Neumann) compact line kernels ----------------
# Overloads on `CompactAxisNeu`. The boundary is closed by an even (Neumann)
# mirror of the field about each wall, so the right-hand side is the interior
# compact stencil evaluated on the mirrored field: no one-sided boundary
# weights, only `a`/`b`. The 1st derivative is odd about the wall (it vanishes
# there -> identity boundary rows, that is the Neumann condition); the 2nd is
# even (mirrored coupling). The solve is the plain tridiagonal Thomas sweep.

"""$(TYPEDSIGNATURES)

In-place plain tridiagonal solve for one **non-periodic Neumann** compact line
`r`.
"""
function _compact_solve_neu!(r, n, c::CompactAxisNeu)
    s, w, sup = c.s, c.w, c.sup
    @inbounds begin
        for i in 2:n
            r[i] -= r[i - 1] * s[i]
        end
        r[n] *= w[n]
        for i in (n - 1):-1:1
            r[i] = (r[i] - sup[i] * r[i + 1]) * w[i]
        end
    end
    return r
end

function compact1line!(a_out, a_in, n, c::CompactAxisNeu)
    a, b = c.a, c.b
    for j in axes(a_in, 2)
        r = @view(a_out[:, j])
        u = @view(a_in[:, j])
        @inbounds begin
            r[1] = zero(eltype(r))
            r[2] = a * (u[3] - u[1]) + b * (u[4] - u[2])
            for i in 3:(n - 2)
                r[i] = a * (u[i + 1] - u[i - 1]) + b * (u[i + 2] - u[i - 2])
            end
            r[n - 1] = a * (u[n] - u[n - 2]) + b * (u[n - 1] - u[n - 3])
            r[n] = zero(eltype(r))
        end
        _compact_solve_neu!(r, n, c)
    end
    return nothing
end

function compact2line!(a_out, a_in, n, c::CompactAxisNeu)
    a, b = c.a, c.b
    for j in axes(a_in, 2)
        r = @view(a_out[:, j])
        u = @view(a_in[:, j])
        @inbounds begin
            r[1] = 2a * (u[2] - u[1]) + 2b * (u[3] - u[1])
            r[2] = a * (u[1] - 2u[2] + u[3]) + b * (u[4] - u[2])
            for i in 3:(n - 2)
                r[i] = a * (u[i + 1] - 2u[i] + u[i - 1]) + b * (u[i + 2] - 2u[i] + u[i - 2])
            end
            r[n - 1] = a * (u[n] - 2u[n - 1] + u[n - 2]) + b * (u[n - 3] - u[n - 1])
            r[n] = 2a * (u[n - 1] - u[n]) + 2b * (u[n - 2] - u[n])
        end
        _compact_solve_neu!(r, n, c)
    end
    return nothing
end

function compact1line3!(a_out, a_in, n, c::CompactAxisNeu)
    a, b = c.a, c.b
    for k in axes(a_in, 3), j in axes(a_in, 2)
        r = @view(a_out[:, j, k])
        u = @view(a_in[:, j, k])
        @inbounds begin
            r[1] = zero(eltype(r))
            r[2] = a * (u[3] - u[1]) + b * (u[4] - u[2])
            for i in 3:(n - 2)
                r[i] = a * (u[i + 1] - u[i - 1]) + b * (u[i + 2] - u[i - 2])
            end
            r[n - 1] = a * (u[n] - u[n - 2]) + b * (u[n - 1] - u[n - 3])
            r[n] = zero(eltype(r))
        end
        _compact_solve_neu!(r, n, c)
    end
    return nothing
end

function compact2line3!(a_out, a_in, n, c::CompactAxisNeu)
    a, b = c.a, c.b
    for k in axes(a_in, 3), j in axes(a_in, 2)
        r = @view(a_out[:, j, k])
        u = @view(a_in[:, j, k])
        @inbounds begin
            r[1] = 2a * (u[2] - u[1]) + 2b * (u[3] - u[1])
            r[2] = a * (u[1] - 2u[2] + u[3]) + b * (u[4] - u[2])
            for i in 3:(n - 2)
                r[i] = a * (u[i + 1] - 2u[i] + u[i - 1]) + b * (u[i + 2] - 2u[i] + u[i - 2])
            end
            r[n - 1] = a * (u[n] - 2u[n - 1] + u[n - 2]) + b * (u[n - 3] - u[n - 1])
            r[n] = 2a * (u[n - 1] - u[n]) + 2b * (u[n - 2] - u[n])
        end
        _compact_solve_neu!(r, n, c)
    end
    return nothing
end

# Spectral-compact (Fourier-multiplier) derivatives.
#
# These dispatch on the PlanCompactFFT* plan types (built for GPU / non-Array
# backends). Each derivative is IDFT(T(ξ)·FFT(ϕ)) with T the compact symbol
# precomputed in the plan (p.T1x, p.T2x, …). The compact symbol already
# contains the factor ``i`` of the derivative, so it replaces the ``im*ξ``
# factor used by the standard spectral path.

## 2D
function computeDerivatives!(gf::GradientField2D, p::PlanCompactFFT2D, ϕt::AbstractArray)
    pfft = p.fft
    # x direction
    mul_x!(pfft.ϕx_hat, pfft, ϕt)
    pfft.a_tmp2x .= p.T1x .* pfft.ϕx_hat
    ldiv_x!(gf.dx, pfft, pfft.a_tmp2x)
    pfft.a_tmp2x .= p.T2x .* pfft.ϕx_hat
    ldiv_x!(gf.ddx, pfft, pfft.a_tmp2x)
    # y direction
    mul_y!(pfft.ϕy_hat, pfft, ϕt)
    pfft.a_tmp2y .= p.T1y .* pfft.ϕy_hat
    ldiv_y!(gf.dy, pfft, pfft.a_tmp2y)
    pfft.a_tmp2y .= p.T2y .* pfft.ϕy_hat
    ldiv_y!(gf.ddy, pfft, pfft.a_tmp2y)
    return nothing
end

function computeDerivatives!(gf::GradientRotField2D, p::PlanCompactFFT2D, ϕt::AbstractArray)
    pfft = p.fft
    grid = localgrid(pfft.pen_x, (pfft.f.g.x, pfft.f.g.y))
    x, y = grid.x, grid.y
    # x direction
    mul_x!(pfft.ϕx_hat, pfft, ϕt)
    pfft.a_tmp2x .= p.T1x .* pfft.ϕx_hat
    ldiv_x!(gf.dx, pfft, pfft.a_tmp2x)
    gf.rx .= y .* gf.dx
    pfft.a_tmp2x .= p.T2x .* pfft.ϕx_hat
    ldiv_x!(gf.ddx, pfft, pfft.a_tmp2x)
    # y direction
    mul_y!(pfft.ϕy_hat, pfft, ϕt)
    pfft.a_tmp2y .= p.T1y .* pfft.ϕy_hat
    ldiv_y!(gf.dy, pfft, pfft.a_tmp2y)
    gf.ry .= -x .* gf.dy
    pfft.a_tmp2y .= p.T2y .* pfft.ϕy_hat
    ldiv_y!(gf.ddy, pfft, pfft.a_tmp2y)
    return nothing
end

## 3D
function computeDerivatives!(gf::GradientField3D, p::PlanCompactFFT3D, ϕt::AbstractArray)
    pfft = p.fft
    # x direction
    mul_x!(pfft.ϕx_hat, pfft, ϕt)
    pfft.a_tmp2x .= p.T1x .* pfft.ϕx_hat
    ldiv_x!(gf.dx, pfft, pfft.a_tmp2x)
    pfft.a_tmp2x .= p.T2x .* pfft.ϕx_hat
    ldiv_x!(gf.ddx, pfft, pfft.a_tmp2x)
    # y direction
    mul_y!(pfft.ϕy_hat, pfft, ϕt)
    pfft.a_tmp2y .= p.T1y .* pfft.ϕy_hat
    ldiv_y!(gf.dy, pfft, pfft.a_tmp2y)
    pfft.a_tmp2y .= p.T2y .* pfft.ϕy_hat
    ldiv_y!(gf.ddy, pfft, pfft.a_tmp2y)
    # z direction
    mul_z!(pfft.ϕz_hat, pfft, ϕt)
    pfft.a_tmp2z .= p.T1z .* pfft.ϕz_hat
    ldiv_z!(gf.dz, pfft, pfft.a_tmp2z)
    pfft.a_tmp2z .= p.T2z .* pfft.ϕz_hat
    ldiv_z!(gf.ddz, pfft, pfft.a_tmp2z)
    return nothing
end

function computeDerivatives!(gf::GradientRotField3D, p::PlanCompactFFT3D, ϕt::AbstractArray)
    pfft = p.fft
    grid = localgrid(pfft.pen_x, (pfft.f.g.x, pfft.f.g.y, pfft.f.g.z))
    x, y = grid.x, grid.y
    # x direction
    mul_x!(pfft.ϕx_hat, pfft, ϕt)
    pfft.a_tmp2x .= p.T1x .* pfft.ϕx_hat
    ldiv_x!(gf.dx, pfft, pfft.a_tmp2x)
    gf.rx .= y .* gf.dx
    pfft.a_tmp2x .= p.T2x .* pfft.ϕx_hat
    ldiv_x!(gf.ddx, pfft, pfft.a_tmp2x)
    # y direction
    mul_y!(pfft.ϕy_hat, pfft, ϕt)
    pfft.a_tmp2y .= p.T1y .* pfft.ϕy_hat
    ldiv_y!(gf.dy, pfft, pfft.a_tmp2y)
    gf.ry .= -x .* gf.dy
    pfft.a_tmp2y .= p.T2y .* pfft.ϕy_hat
    ldiv_y!(gf.ddy, pfft, pfft.a_tmp2y)
    # z direction
    mul_z!(pfft.ϕz_hat, pfft, ϕt)
    pfft.a_tmp2z .= p.T1z .* pfft.ϕz_hat
    ldiv_z!(gf.dz, pfft, pfft.a_tmp2z)
    pfft.a_tmp2z .= p.T2z .* pfft.ϕz_hat
    ldiv_z!(gf.ddz, pfft, pfft.a_tmp2z)
    return nothing
end

## 2D
function computeDerivatives!(gf::GradientField2D, p::AbstractCompactPlan, ϕt::AbstractArray)
    nx = p.ax.n
    # x direction
    compact1line!(parent(gf.dx), parent(ϕt), nx, p.ax)
    compact2line!(parent(gf.ddx), parent(ϕt), nx, p.a2x)
    # y direction
    transpose!(p.ϕytmp, ϕt)
    ny = p.ay.n
    dy = similar(p.ϕytmp)
    ddy = similar(p.ϕytmp)
    compact1line!(parent(dy), parent(p.ϕytmp), ny, p.ay)
    compact2line!(parent(ddy), parent(p.ϕytmp), ny, p.a2y)
    transpose!(gf.dy, dy)
    transpose!(gf.ddy, ddy)
    return nothing
end

function computeDerivatives!(gf::GradientRotField2D, p::AbstractCompactPlan, ϕt::AbstractArray)
    nx = p.ax.n
    grid = localgrid(p.pen_x, (p.f.g.x, p.f.g.y))
    x, y = grid.x, grid.y
    # x direction
    compact1line!(parent(gf.dx), parent(ϕt), nx, p.ax)
    gf.rx .= y .* gf.dx
    compact2line!(parent(gf.ddx), parent(ϕt), nx, p.a2x)
    # y direction
    transpose!(p.ϕytmp, ϕt)
    ny = p.ay.n
    dy = similar(p.ϕytmp)
    ddy = similar(p.ϕytmp)
    compact1line!(parent(dy), parent(p.ϕytmp), ny, p.ay)
    compact2line!(parent(ddy), parent(p.ϕytmp), ny, p.a2y)
    transpose!(gf.dy, dy)
    transpose!(gf.ddy, ddy)
    gf.ry .= -x .* gf.dy
    return nothing
end

## 3D
function computeDerivatives!(gf::GradientField3D, p::AbstractCompactPlan, ϕt::AbstractArray)
    nx = p.ax.n
    # x direction
    compact1line3!(parent(gf.dx), parent(ϕt), nx, p.ax)
    compact2line3!(parent(gf.ddx), parent(ϕt), nx, p.a2x)
    # y direction
    transpose!(p.ϕytmp, ϕt)
    ny = p.ay.n
    dy = similar(p.ϕytmp)
    ddy = similar(p.ϕytmp)
    compact1line3!(parent(dy), parent(p.ϕytmp), ny, p.ay)
    compact2line3!(parent(ddy), parent(p.ϕytmp), ny, p.a2y)
    transpose!(gf.dy, dy)
    transpose!(gf.ddy, ddy)
    # z direction (z-layout -> y-layout -> x-layout, one permutation each)
    transpose!(p.ϕztmp, p.ϕytmp)
    nz = p.az.n
    dz = similar(p.ϕztmp)
    ddz = similar(p.ϕztmp)
    compact1line3!(parent(dz), parent(p.ϕztmp), nz, p.az)
    compact2line3!(parent(ddz), parent(p.ϕztmp), nz, p.a2z)
    transpose!(dy, dz)
    transpose!(ddy, ddz)
    transpose!(gf.dz, dy)
    transpose!(gf.ddz, ddy)
    return nothing
end

function computeDerivatives!(gf::GradientRotField3D, p::AbstractCompactPlan, ϕt::AbstractArray)
    nx = p.ax.n
    grid = localgrid(p.pen_x, (p.f.g.x, p.f.g.y, p.f.g.z))
    x, y = grid.x, grid.y
    # x direction
    compact1line3!(parent(gf.dx), parent(ϕt), nx, p.ax)
    gf.rx .= y .* gf.dx
    compact2line3!(parent(gf.ddx), parent(ϕt), nx, p.a2x)
    # y direction
    transpose!(p.ϕytmp, ϕt)
    ny = p.ay.n
    dy = similar(p.ϕytmp)
    ddy = similar(p.ϕytmp)
    compact1line3!(parent(dy), parent(p.ϕytmp), ny, p.ay)
    compact2line3!(parent(ddy), parent(p.ϕytmp), ny, p.a2y)
    transpose!(gf.dy, dy)
    transpose!(gf.ddy, ddy)
    gf.ry .= -x .* gf.dy
    # z direction (z-layout -> y-layout -> x-layout, one permutation each)
    transpose!(p.ϕztmp, p.ϕytmp)
    nz = p.az.n
    dz = similar(p.ϕztmp)
    ddz = similar(p.ϕztmp)
    compact1line3!(parent(dz), parent(p.ϕztmp), nz, p.az)
    compact2line3!(parent(ddz), parent(p.ϕztmp), nz, p.a2z)
    transpose!(dy, dz)
    transpose!(ddy, ddz)
    transpose!(gf.dz, dy)
    transpose!(gf.ddz, ddy)
    return nothing
end

# ==========================================================================
# GPU compact (CUDA Thomas kernel) derivatives.
#
# These dispatch on the PlanCompactGPU* plan types. The kernel solves the
# periodic compact relation along the leading dimension of a raw CuArray
# (one thread per line); the y/z directions are handled with PencilArray
# transposes that bring the derivative axis to the front, exactly like the
# CPU compact dispatch. Scratch buffers are preallocated in the plan.
# ==========================================================================

## 2D
function computeDerivatives!(gf::GradientField2D, p::PlanCompactGPU2D, ϕt::AbstractArray)
    nx = p.ax.n
    # x direction
    _compact_cu_1line!(parent(gf.dx), parent(ϕt), nx, p.ax)
    _compact_cu_2line!(parent(gf.ddx), parent(ϕt), nx, p.a2x)
    # y direction
    transpose!(p.ϕytmp, ϕt)
    ny = p.ay.n
    _compact_cu_1line!(parent(p.dytmp), parent(p.ϕytmp), ny, p.ay)
    _compact_cu_2line!(parent(p.ddytmp), parent(p.ϕytmp), ny, p.a2y)
    transpose!(gf.dy, p.dytmp)
    transpose!(gf.ddy, p.ddytmp)
    return nothing
end

function computeDerivatives!(gf::GradientRotField2D, p::PlanCompactGPU2D, ϕt::AbstractArray)
    nx = p.ax.n
    grid = localgrid(p.pen_x, (p.f.g.x, p.f.g.y))
    x, y = grid.x, grid.y
    # x direction
    _compact_cu_1line!(parent(gf.dx), parent(ϕt), nx, p.ax)
    gf.rx .= y .* gf.dx
    _compact_cu_2line!(parent(gf.ddx), parent(ϕt), nx, p.a2x)
    # y direction
    transpose!(p.ϕytmp, ϕt)
    ny = p.ay.n
    _compact_cu_1line!(parent(p.dytmp), parent(p.ϕytmp), ny, p.ay)
    _compact_cu_2line!(parent(p.ddytmp), parent(p.ϕytmp), ny, p.a2y)
    transpose!(gf.dy, p.dytmp)
    transpose!(gf.ddy, p.ddytmp)
    gf.ry .= -x .* gf.dy
    return nothing
end

## 3D
function computeDerivatives!(gf::GradientField3D, p::PlanCompactGPU3D, ϕt::AbstractArray)
    nx = p.ax.n
    # x direction
    _compact_cu_1line3!(parent(gf.dx), parent(ϕt), nx, p.ax)
    _compact_cu_2line3!(parent(gf.ddx), parent(ϕt), nx, p.a2x)
    # y direction
    transpose!(p.ϕytmp, ϕt)
    ny = p.ay.n
    _compact_cu_1line3!(parent(p.dytmp), parent(p.ϕytmp), ny, p.ay)
    _compact_cu_2line3!(parent(p.ddytmp), parent(p.ϕytmp), ny, p.a2y)
    transpose!(gf.dy, p.dytmp)
    transpose!(gf.ddy, p.ddytmp)
    # z direction (z-layout <- y-layout, then back x-layout <- y-layout)
    transpose!(p.ϕztmp, p.ϕytmp)
    nz = p.az.n
    _compact_cu_1line3!(parent(p.dztmp), parent(p.ϕztmp), nz, p.az)
    _compact_cu_2line3!(parent(p.ddztmp), parent(p.ϕztmp), nz, p.a2z)
    transpose!(p.dytmp, p.dztmp)
    transpose!(p.ddytmp, p.ddztmp)
    transpose!(gf.dz, p.dytmp)
    transpose!(gf.ddz, p.ddytmp)
    return nothing
end

function computeDerivatives!(gf::GradientRotField3D, p::PlanCompactGPU3D, ϕt::AbstractArray)
    nx = p.ax.n
    grid = localgrid(p.pen_x, (p.f.g.x, p.f.g.y, p.f.g.z))
    x, y = grid.x, grid.y
    # x direction
    _compact_cu_1line3!(parent(gf.dx), parent(ϕt), nx, p.ax)
    gf.rx .= y .* gf.dx
    _compact_cu_2line3!(parent(gf.ddx), parent(ϕt), nx, p.a2x)
    # y direction
    transpose!(p.ϕytmp, ϕt)
    ny = p.ay.n
    _compact_cu_1line3!(parent(p.dytmp), parent(p.ϕytmp), ny, p.ay)
    _compact_cu_2line3!(parent(p.ddytmp), parent(p.ϕytmp), ny, p.a2y)
    transpose!(gf.dy, p.dytmp)
    transpose!(gf.ddy, p.ddytmp)
    gf.ry .= -x .* gf.dy
    # z direction (z-layout <- y-layout, then back x-layout <- y-layout)
    transpose!(p.ϕztmp, p.ϕytmp)
    nz = p.az.n
    _compact_cu_1line3!(parent(p.dztmp), parent(p.ϕztmp), nz, p.az)
    _compact_cu_2line3!(parent(p.ddztmp), parent(p.ϕztmp), nz, p.a2z)
    transpose!(p.dytmp, p.dztmp)
    transpose!(p.ddytmp, p.ddztmp)
    transpose!(gf.dz, p.dytmp)
    transpose!(gf.ddz, p.ddytmp)
    return nothing
end

# Finite difference helper functions
function computedxddx!(ϕt::AbstractArray{A,2}, dx::AbstractArray{A,2},
                       ddx::AbstractArray{A,2}, Δx::Real; order::Integer=2) where {A}
    N = size(ϕt)
    nx = N[1]
    if order == 2
        for i in 1:nx
            dx[i, :] = (ϕt[i % nx + 1, :] - ϕt[mod(i % nx - 2, nx) + 1, :]) / (2 * Δx)
            ddx[i, :] = (-2 * ϕt[i, :] + ϕt[i % nx + 1, :] + ϕt[mod(i % nx - 2, nx) + 1, :]) /
                        Δx^2
        end
    elseif order == 4
        for i in 1:nx
            dx[i, :] = (-ϕt[(i + 1) % nx + 1, :] + 8 * ϕt[i % nx + 1, :] -
                        8 * ϕt[mod(i % nx - 2, nx) + 1, :] + ϕt[mod(i % nx - 3, nx) + 1, :]) /
                       (12 * Δx)
            ddx[i, :] = (-30 * ϕt[i, :] - ϕt[(i + 1) % nx + 1, :] + 16 * ϕt[i % nx + 1, :] +
                         16 * ϕt[mod(i % nx - 2, nx) + 1, :] -
                         ϕt[mod(i % nx - 3, nx) + 1, :]) / (12 * Δx^2)
        end
    elseif order == 6
        for i in 1:nx
            dx[i, :] = (ϕt[(i + 2) % nx + 1, :] - 9 * ϕt[(i + 1) % nx + 1, :] +
                        45 * ϕt[i % nx + 1, :] - 45 * ϕt[mod(i % nx - 2, nx) + 1, :] +
                        9 * ϕt[mod(i % nx - 3, nx) + 1, :] - ϕt[mod(i % nx - 4, nx) + 1, :]) /
                       (60 * Δx)
            ddx[i, :] = (-490 * ϕt[i, :] + 2 * ϕt[(i + 2) % nx + 1, :] -
                         27 * ϕt[(i + 1) % nx + 1, :] + 270 * ϕt[i % nx + 1, :] +
                         270 * ϕt[mod(i % nx - 2, nx) + 1, :] -
                         27 * ϕt[mod(i % nx - 3, nx) + 1, :] +
                         2 * ϕt[mod(i % nx - 4, nx) + 1, :]) / (180 * Δx^2)
        end
    elseif order == 8
        for i in 1:nx
            dx[i, :] = (-3 * ϕt[(i + 3) % nx + 1, :] + 32 * ϕt[(i + 2) % nx + 1, :] -
                        168 * ϕt[(i + 1) % nx + 1, :] + 672 * ϕt[i % nx + 1, :]
                        -
                        672 * ϕt[mod(i % nx - 2, nx) + 1, :] +
                        168 * ϕt[mod(i % nx - 3, nx) + 1, :] -
                        32 * ϕt[mod(i % nx - 4, nx) + 1, :] +
                        3 * ϕt[mod(i % nx - 5, nx) + 1, :]) / (840 * Δx)
            ddx[i, :] = (-14350 * ϕt[i, :] - 9 * ϕt[(i + 3) % nx + 1, :] +
                         128 * ϕt[(i + 2) % nx + 1, :] - 1008 * ϕt[(i + 1) % nx + 1, :] +
                         8064 * ϕt[i % nx + 1, :]
                         + 8064 * ϕt[mod(i % nx - 2, nx) + 1, :] -
                         1008 * ϕt[mod(i % nx - 3, nx) + 1, :] +
                         128 * ϕt[mod(i % nx - 4, nx) + 1, :] -
                         9 * ϕt[mod(i % nx - 5, nx) + 1, :]) / (5040 * Δx^2)
        end
    end
    return nothing
end

function computedxddx!(ϕt::AbstractArray{A,3}, dx::AbstractArray{A,3},
                       ddx::AbstractArray{A,3}, Δx::Real; order::Integer=2) where {A}
    N = size(ϕt)
    nx = N[1]
    if order == 2
        for i in 1:nx
            dx[i, :, :] = (ϕt[i % nx + 1, :, :] - ϕt[mod(i % nx - 2, nx) + 1, :, :]) /
                          (2 * Δx)
            ddx[i, :, :] = (-2 * ϕt[i, :, :] + ϕt[i % nx + 1, :, :] +
                            ϕt[mod(i % nx - 2, nx) + 1, :, :]) / Δx^2
        end
    elseif order == 4
        for i in 1:nx
            dx[i, :, :] = (-ϕt[(i + 1) % nx + 1, :, :] + 8 * ϕt[i % nx + 1, :, :] -
                           8 * ϕt[mod(i % nx - 2, nx) + 1, :, :] +
                           ϕt[mod(i % nx - 3, nx) + 1, :, :]) / (12 * Δx)
            ddx[i, :, :] = (-30 * ϕt[i, :, :] - ϕt[(i + 1) % nx + 1, :, :] +
                            16 * ϕt[i % nx + 1, :, :] +
                            16 * ϕt[mod(i % nx - 2, nx) + 1, :, :] -
                            ϕt[mod(i % nx - 3, nx) + 1, :, :]) / (12 * Δx^2)
        end
    elseif order == 6
        for i in 1:nx
            dx[i, :, :] = (ϕt[(i + 2) % nx + 1, :, :] - 9 * ϕt[(i + 1) % nx + 1, :, :] +
                           45 * ϕt[i % nx + 1, :, :] -
                           45 * ϕt[mod(i % nx - 2, nx) + 1, :, :] +
                           9 * ϕt[mod(i % nx - 3, nx) + 1, :, :] -
                           ϕt[mod(i % nx - 4, nx) + 1, :, :]) / (60 * Δx)
            ddx[i, :, :] = (-490 * ϕt[i, :, :] + 2 * ϕt[(i + 2) % nx + 1, :, :] -
                            27 * ϕt[(i + 1) % nx + 1, :, :] + 270 * ϕt[i % nx + 1, :, :] +
                            270 * ϕt[mod(i % nx - 2, nx) + 1, :, :] -
                            27 * ϕt[mod(i % nx - 3, nx) + 1, :, :] +
                            2 * ϕt[mod(i % nx - 4, nx) + 1, :, :]) / (180 * Δx^2)
        end
    elseif order == 8
        for i in 1:nx
            dx[i, :, :] = (-3 * ϕt[(i + 3) % nx + 1, :, :] +
                           32 * ϕt[(i + 2) % nx + 1, :, :] -
                           168 * ϕt[(i + 1) % nx + 1, :, :] + 672 * ϕt[i % nx + 1, :, :]
                           -
                           672 * ϕt[mod(i % nx - 2, nx) + 1, :, :] +
                           168 * ϕt[mod(i % nx - 3, nx) + 1, :, :] -
                           32 * ϕt[mod(i % nx - 4, nx) + 1, :, :] +
                           3 * ϕt[mod(i % nx - 5, nx) + 1, :, :]) / (840 * Δx)
            ddx[i, :, :] = (-14350 * ϕt[i, :, :] - 9 * ϕt[(i + 3) % nx + 1, :, :] +
                            128 * ϕt[(i + 2) % nx + 1, :, :] -
                            1008 * ϕt[(i + 1) % nx + 1, :, :] + 8064 * ϕt[i % nx + 1, :, :]
                            + 8064 * ϕt[mod(i % nx - 2, nx) + 1, :, :] -
                            1008 * ϕt[mod(i % nx - 3, nx) + 1, :, :] +
                            128 * ϕt[mod(i % nx - 4, nx) + 1, :, :] -
                            9 * ϕt[mod(i % nx - 5, nx) + 1, :, :]) / (5040 * Δx^2)
        end
    end
    return nothing
end

function computedyddy!(ϕt::AbstractArray{A,2}, dy::AbstractArray{A,2},
                       ddy::AbstractArray{A,2}, Δy::Real; order::Integer=2) where {A}
    N = size(ϕt)
    # after transpose!, the y axis is the leading (1st) dimension
    ny = N[1]
    if order == 2
        for j in 1:ny
            dy[j, :] = (ϕt[j % ny + 1, :] - ϕt[mod(j % ny - 2, ny) + 1, :]) / (2 * Δy)
            ddy[j, :] = (-2 * ϕt[j, :] + ϕt[j % ny + 1, :] + ϕt[mod(j % ny - 2, ny) + 1, :]) /
                        Δy^2
        end
    elseif order == 4
        for j in 1:ny
            dy[j, :] = (-ϕt[(j + 1) % ny + 1, :] + 8 * ϕt[j % ny + 1, :] -
                        8 * ϕt[mod(j % ny - 2, ny) + 1, :] + ϕt[mod(j % ny - 3, ny) + 1, :]) /
                       (12 * Δy)
            ddy[j, :] = (-30 * ϕt[j, :] - ϕt[(j + 1) % ny + 1] + 16 * ϕt[:, j % ny + 1, :] +
                         16 * ϕt[mod(j % ny - 2, ny) + 1, :] -
                         ϕt[mod(j % ny - 3, ny) + 1, :]) / (12 * Δy^2)
        end
    elseif order == 6
        for j in 1:ny
            dy[j, :] = (ϕt[(j + 2) % ny + 1, :] - 9 * ϕt[(j + 1) % ny + 1, :] +
                        45 * ϕt[j % ny + 1, :] - 45 * ϕt[mod(j % ny - 2, ny) + 1, :] +
                        9 * ϕt[mod(j % ny - 3, ny) + 1, :] - ϕt[mod(j % ny - 4, ny) + 1, :]) /
                       (60 * Δy)
            ddy[j, :] = (-490 * ϕt[j, :] + 2 * ϕt[(j + 2) % ny + 1, :] -
                         27 * ϕt[(j + 1) % ny + 1, :] + 270 * ϕt[j % ny + 1, :] +
                         270 * ϕt[mod(j % ny - 2, ny) + 1, :] -
                         27 * ϕt[mod(j % ny - 3, ny) + 1, :] +
                         2 * ϕt[mod(j % ny - 4, ny) + 1, :]) / (180 * Δy^2)
        end
    elseif order == 8
        for j in 1:ny
            dy[j, :] = (-3 * ϕt[(j + 3) % ny + 1, :] + 32 * ϕt[(j + 2) % ny + 1, :] -
                        168 * ϕt[(j + 1) % ny + 1, :] + 672 * ϕt[j % ny + 1, :]
                        -
                        672 * ϕt[mod(j % ny - 2, ny) + 1, :] +
                        168 * ϕt[mod(j % ny - 3, ny) + 1, :] -
                        32 * ϕt[mod(j % ny - 4, ny) + 1, :] +
                        3 * ϕt[mod(j % ny - 5, ny) + 1, :]) / (840 * Δy)
            ddy[j, :] = (-14350 * ϕt[j, :] - 9 * ϕt[(j + 3) % ny + 1, :] +
                         128 * ϕt[(j + 2) % ny + 1, :] - 1008 * ϕt[(j + 1) % ny + 1, :] +
                         8064 * ϕt[j % ny + 1, :]
                         + 8064 * ϕt[mod(j % ny - 2, ny) + 1, :] -
                         1008 * ϕt[mod(j % ny - 3, ny) + 1, :] +
                         128 * ϕt[mod(j % ny - 4, ny) + 1, :] -
                         9 * ϕt[mod(j % ny - 5, ny) + 1, :]) / (5040 * Δy^2)
        end
    end
    return nothing
end

function computedyddy!(ϕt::AbstractArray{A,3}, dy::AbstractArray{A,3},
                       ddy::AbstractArray{A,3}, Δy::Real; order::Integer=2) where {A}
    N = size(ϕt)
    # after transpose!, the y axis is the leading (1st) dimension
    ny = N[1]
    if order == 2
        for j in 1:ny
            dy[j, :, :] = (ϕt[j % ny + 1, :, :] - ϕt[mod(j % ny - 2, ny) + 1, :, :]) /
                          (2 * Δy)
            ddy[j, :, :] = (-2 * ϕt[j, :, :] + ϕt[j % ny + 1, :, :] +
                            ϕt[mod(j % ny - 2, ny) + 1, :, :]) / Δy^2
        end
    elseif order == 4
        for j in 1:ny
            dy[j, :, :] = (-ϕt[(j + 1) % ny + 1, :, :] + 8 * ϕt[j % ny + 1, :, :] -
                           8 * ϕt[mod(j % ny - 2, ny) + 1, :, :] +
                           ϕt[mod(j % ny - 3, ny) + 1, :, :]) / (12 * Δy)
            ddy[j, :, :] = (-30 * ϕt[:, j, :] - ϕt[(j + 1) % ny + 1, :, :] +
                            16 * ϕt[j % ny + 1, :, :] +
                            16 * ϕt[mod(j % ny - 2, ny) + 1, :, :] -
                            ϕt[mod(j % ny - 3, ny) + 1, :, :]) / (12 * Δy^2)
        end
    elseif order == 6
        for j in 1:ny
            dy[j, :, :] = (ϕt[(j + 2) % ny + 1, :, :] - 9 * ϕt[(j + 1) % ny + 1, :, :] +
                           45 * ϕt[j % ny + 1, :, :] -
                           45 * ϕt[mod(j % ny - 2, ny) + 1, :, :] +
                           9 * ϕt[mod(j % ny - 3, ny) + 1, :, :] -
                           ϕt[mod(j % ny - 4, ny) + 1, :, :]) / (60 * Δy)
            ddy[j, :, :] = (-490 * ϕt[j, :, :] + 2 * ϕt[(j + 2) % ny + 1, :, :] -
                            27 * ϕt[(j + 1) % ny + 1, :, :] + 270 * ϕt[j % ny + 1, :, :] +
                            270 * ϕt[mod(j % ny - 2, ny) + 1, :, :] -
                            27 * ϕt[mod(j % ny - 3, ny) + 1, :, :] +
                            2 * ϕt[mod(j % ny - 4, ny) + 1, :, :]) / (180 * Δy^2)
        end
    elseif order == 8
        for
            dy[j, :, :] = (-3 * ϕt[(j + 3) % ny + 1, :, :] +
                           32 * ϕt[(j + 2) % ny + 1, :, :] -
                           168 * ϕt[(j + 1) % ny + 1, :, :] + 672 * ϕt[j % ny + 1, :, :]
                           -
                           672 * ϕt[mod(j % ny - 2, ny) + 1, :, :] +
                           168 * ϕt[mod(j % ny - 3, ny) + 1, :, :] -
                           32 * ϕt[mod(j % ny - 4, ny) + 1, :, :] +
                           3 * ϕt[mod(j % ny - 5, ny) + 1, :, :]) / (840 * Δy)
            ddy[j, :, :] = (-14350 * ϕt[j, :, :] - 9 * ϕt[(j + 3) % ny + 1, :, :] +
                            128 * ϕt[(j + 2) % ny + 1, :, :] -
                            1008 * ϕt[(j + 1) % ny + 1, :] + 8064 * ϕt[:, j % ny + 1, :, :]
                            + 8064 * ϕt[mod(j % ny - 2, ny) + 1, :, :] -
                            1008 * ϕt[mod(j % ny - 3, ny) + 1, :, :] +
                            128 * ϕt[mod(j % ny - 4, ny) + 1, :, :] -
                            9 * ϕt[mod(j % ny - 5, ny) + 1, :, :]) / (5040 * Δy^2)
        end
    end
    return nothing
end

function computedzddz!(ϕt::AbstractArray{A,3}, dz::AbstractArray{A,3},
                       ddz::AbstractArray{A,3}, Δz::Real; order::Integer=2) where {A}
    N = size(ϕt)
    # after transpose!, the z axis is the leading (1st) dimension
    nz = N[1]
    if order == 2
        for k in 1:nz
            dz[k, :, :] = (ϕt[k % nz + 1, :, :] - ϕt[mod(k % nz - 2, nz) + 1, :, :]) /
                          (2 * Δz)
            ddz[k, :, :] = (-2 * ϕt[k, :, :] + ϕt[k % nz + 1, :, :] +
                            ϕt[mod(k % nz - 2, nz) + 1, :, :]) / Δz^2
        end
    elseif order == 4
        for k in 1:nz
            dz[k, :, :] = (-ϕt[(k + 1) % nz + 1, :, :] + 8 * ϕt[k % nz + 1, :, :] -
                           8 * ϕt[mod(k % nz - 2, nz) + 1, :, :] +
                           ϕt[mod(k % nz - 3, nz) + 1, :, :]) / (12 * Δz)
            ddz[k, :, :] = (-30 * ϕt[k, :, :] - ϕt[(k + 1) % nz + 1, :, :] +
                            16 * ϕt[k % nz + 1, :, :] +
                            16 * ϕt[mod(k % nz - 2, nz) + 1, :, :] -
                            ϕt[mod(k % nz - 3, nz) + 1, :, :]) / (12 * Δz^2)
        end
    elseif order == 6
        for k in 1:nz
            dz[k, :, :] = (ϕt[(k + 2) % nz + 1, :, :] - 9 * ϕt[(k + 1) % nz + 1, :, :] +
                           45 * ϕt[k % nz + 1, :, :] -
                           45 * ϕt[mod(k % nz - 2, nz) + 1, :, :] +
                           9 * ϕt[mod(k % nz - 3, nz) + 1, :, :] -
                           ϕt[mod(k % nz - 4, nz) + 1, :, :]) / (60 * Δz)
            ddz[k, :, :] = (-490 * ϕt[k, :, :] + 2 * ϕt[(k + 2) % nz + 1, :, :] -
                            27 * ϕt[(k + 1) % nz + 1, :, :] + 270 * ϕt[k % nz + 1, :, :] +
                            270 * ϕt[mod(k % nz - 2, nz) + 1, :, :] -
                            27 * ϕt[mod(k % nz - 3, nz) + 1, :, :] +
                            2 * ϕt[mod(k % nz - 4, nz) + 1, :, :]) / (180 * Δz^2)
        end
    elseif order == 8
        for k in 1:nz
            dz[k, :, :] = (-3 * ϕt[(k + 3) % nz + 1, :, :] +
                           32 * ϕt[(k + 2) % nz + 1, :, :] -
                           168 * ϕt[(k + 1) % nz + 1, :, :] + 672 * ϕt[k % nz + 1, :, :]
                           -
                           672 * ϕt[mod(k % nz - 2, nz) + 1, :, :] +
                           168 * ϕt[mod(k % nz - 3, nz) + 1, :, :] -
                           32 * ϕt[mod(k % nz - 4, nz) + 1, :, :] +
                           3 * ϕt[mod(k % nz - 5, nz) + 1, :, :]) / (840 * Δz)
            ddz[k, :, :] = (-14350 * ϕt[k, :, :] - 9 * ϕt[(k + 3) % nz + 1, :, :] +
                            128 * ϕt[(k + 2) % nz + 1, :, :] -
                            1008 * ϕt[(k + 1) % nz + 1, :, :] + 8064 * ϕt[k % nz + 1, :, :]
                            + 8064 * ϕt[mod(k % nz - 2, nz) + 1, :, :] -
                            1008 * ϕt[mod(k % nz - 3, nz) + 1, :, :] +
                            128 * ϕt[mod(k % nz - 4, nz) + 1, :, :] -
                            9 * ϕt[mod(k % nz - 5, nz) + 1, :, :]) / (5040 * Δz^2)
        end
    end
    return nothing
end
