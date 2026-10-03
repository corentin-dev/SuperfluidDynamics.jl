"""
    cross(c, a, b)

Compute the cross product of two vector fields `c = a × b` and return it as a
new field.

    cross!(c, a, b)

Same, but writes in-place into the (already allocated) vector field `c`.
"""
function cross(a, b)
    c = similar_data(a)
    cross!(c, a, b)
    return c
end

function cross!(c, a, b)
    @. c[1] = a[2] * b[3] - a[3] * b[2]
    @. c[2] = a[3] * b[1] - a[1] * b[3]
    @. c[3] = a[1] * b[2] - a[2] * b[1]
    return c
end

# 2/3-rule bound ξmax = (4/9)·min_d max_d' |ξ_d'|², from the plan's GLOBAL
# wavenumber vectors. The spectral fields are dealiased on the last pencil,
# whose grid is distributed over two directions: a rank-local max sees only
# part of the spectrum, its bound is smaller than the global one, and the rank
# then zeroes modes its peers keep — physics that depends on the rank count
# (measured: 1.9e-3 energy drift 1→4 ranks at N=48, benchmarks/
# scalability_rankinv.jl). The plan stores the full (undistributed) ξ vectors,
# so the global bound costs nothing and needs no MPI reduction.
function ξmax_global(plan::AbstractFFTPlan)
    sq = x -> x^2
    # dimension test on the plan's own fields (the {N} parameter counts data
    # components, not dimensions): 3D plans carry ξz.
    ξs = isdefined(plan, :ξz) ? (plan.ξx, plan.ξy, plan.ξz) : (plan.ξx, plan.ξy)
    return 4 / 9 * minimum(maximum(sq, ξ) for ξ in ξs)
end

"""
    dealias!(u_hat, ξx, ξy, ξz, ξmax)

2/3-rule dealiasing: zero out the modes of the spectral vector field `u_hat`
whose |k|² exceeds `ξmax` (the GLOBAL bound from `ξmax_global`; passing a
rank-local bound makes the filtered modes depend on the rank count).
"""
function dealias!(u_hat, ξx, ξy, ξz, ξmax)
    @. u_hat[1] *= (ξx^2 + ξy^2 + ξz^2) < ξmax
    @. u_hat[2] *= (ξx^2 + ξy^2 + ξz^2) < ξmax
    @. u_hat[3] *= (ξx^2 + ξy^2 + ξz^2) < ξmax
    return nothing
end

"""
    dealias2!(u_hat, ξx, ξy, ξmax)

2/3-rule dealiasing for a 2-component (2D) spectral velocity field `u_hat`
(`ξmax` must be the global bound from `ξmax_global`).
"""
function dealias2!(u_hat, ξx, ξy, ξmax)
    @. u_hat[1] *= (ξx^2 + ξy^2) < ξmax
    @. u_hat[2] *= (ξx^2 + ξy^2) < ξmax
    return nothing
end

"""
    cross2!(c, a, ωz)

2D advective term `c = u × (ωz ẑ)` written in place. For `u = (a[1], a[2])`
and vorticity `ω = ωz ẑ`, the cross product is
`c[1] = -a[2]*ωz`, `c[2] = a[1]*ωz` (the z-component is 0).
"""
function cross2!(c, a, ωz)
    @. c[1] = a[2] * ωz
    @. c[2] = -a[1] * ωz
    return c
end
