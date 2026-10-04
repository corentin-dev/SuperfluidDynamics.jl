# Bogoliubov-de Gennes eigenproblem, matrix-free.
#
# Given a stationary GP order parameter psi0 (with chemical potential mu), the
# excitations (u, v) satisfy
#
#     H  u - K v =  omega u
#     K* u - H v = -omega v
#
# with H = coeffΔ*Lap + V + 2*beta*|psi0|^2 - mu - i*Omega*Lz and K = beta*psi0^2
# (coeffΔ < 0, so the kinetic term is the positive operator -|coeffΔ| ∇²).
# The spectrum is symmetric (omega <-> -omega); each pair (u, v) and (v*, u*) is
# normalized to |u|^2 - |v|^2 = 1 (symplectic norm) and the positive-norm mode
# of each pair is kept.
#
# The operator is applied through the package's own derivative machinery
# (computeDerivatives! with the plan carried by the model), so the memory
# footprint is O(N) and no dense 2N x 2N matrix is ever formed.
#
# The omega = 0 mode (u, v) = (psi0, psi0*) sits at the origin of the spectrum,
# where smallest-|omega| Arnoldi converges slowest. It is NOT deflated (a
# rank-1 deflation of a zero eigenvalue is a no-op) but filtered out after the
# solve, by identifying it with the known vector (psi0, psi0*): the positive-
# symplectic-norm member of each omega <-> -omega pair is kept, except when its
# overlap with (psi0, psi0*) is large (in which case it is the zero mode, not a
# physical excitation). This structure-based rejection stays robust even when
# psi0 is not an exact eigenstate of the discrete operator, in which case the
# zero mode appears as a small-residual pair at omega ≈ ±eps.

using LinearMaps
import Arpack

"""
$(TYPEDEF)

Matrix-free Bogoliubov-de Gennes eigensolver for the Gross-Pitaevskii equation
linearized about a stationary order parameter ``\\psi_0`` (`f.ϕ`), 2D or 3D:

```math
\\begin{pmatrix} H & -K \\\\ K^* & -H \\end{pmatrix}
\\begin{pmatrix} u \\\\ v \\end{pmatrix} = \\omega
\\begin{pmatrix} u \\\\ v \\end{pmatrix},
```

```math
H = \\text{coeffΔ}\\,\\nabla^2 + V + 2\\beta|\\psi_0|^2 - \\mu - i\\Omega\\mathcal{L}_z,
\\qquad K = \\beta\\psi_0^2,
```

(with ``\\text{coeffΔ} < 0`` so the kinetic term is the positive operator
``-|\\text{coeffΔ}|\\,\\nabla^2``),

with the chemical potential
``\\mu = \\langle\\psi_0, (\\text{coeffΔ}\\nabla^2 + V + \\beta|\\psi_0|^2)\\psi_0\\rangle``
of the stationary state. The eigenpairs are returned with the symplectic
normalization ``\\|u\\|^2-\\|v\\|^2 = 1`` and the positive-norm member of each
``\\omega\\leftrightarrow-\\omega`` pair is kept.

`timeStep!` runs the whole eigensolve (this is an eigensolver, not a time
stepper), so the standard driver `solve!(n)` works with `n.niter = 1`.

It contains the following informations:

$(TYPEDFIELDS)
"""
mutable struct NumModelBdG{F,P,Plan} <: AbstractNumModel{F,P,Plan}
    f::F
    gf::Any
    param::P
    Δt::Real
    niter::Integer
    freqbckp::Integer
    plan::Plan
    writers::AbstractWriterCollection{F}
    "chemical potential of the stationary state ψ₀ (set by `timeStep!`)."
    mu::Float64
    "number of eigenvalues requested (per sign of ω)."
    nev::Integer
    "which eigenvalues: `:SM` (default, smallest |ω|) or `:LM`."
    which::Symbol
    "eigenproblem tolerance (0 = Arpack default)."
    tol::Real
    "Krylov restarts."
    restarts::Integer
    "Krylov basis size handed to Arpack (`ncv`); `0` = the automatic choice"
    " `min(max(2nev+7, 20), 2N-2)`. A small trapped problem can make Arpack "
    "report \"all possible eigenvalues of OP have been found\" at the default "
    "basis size; raising `ncv` is then the knob (exposed, not guessed)."
    ncv::Integer
    "eigenvalues found (filled by `timeStep!`)."
    ωs::Vector{Float64}
    "eigenvectors u (one per eigenvalue, same layout as f.ϕ)."
    us::Vector{Any}
    "eigenvectors v (one per eigenvalue, same layout as f.ϕ)."
    vs::Vector{Any}
    "scratch arrays with the layout of f.ϕ (operator application)."
    utmp::PencilArray
    vtmp::PencilArray
end

"""
$(TYPEDSIGNATURES)

Builds a `NumModelBdG`. The field `f` must contain the stationary state ψ₀
(see e.g. `NumModelGPRK` with a large `niter` or the implicit GP models to
obtain one). `param` carries the potential, the interaction coefficient `β`
and the rotation `Ω`.

`nev` eigenvalues are requested (per sign of ω), `which` selects the Arpack
`which` (`:SM` = smallest |ω|, `:LM` = largest), `tol` is the eigenproblem
tolerance (0 = Arpack default), `restarts` is the maximum number of Krylov
restarts and `ncv` the Krylov basis size (0 = automatic, see the field doc).
"""
function NumModelBdG(f::AbstractField,
                     param::BdGParameters,
                     niter::Integer, freqbckp::Integer;
                     nev::Integer=6, which::Symbol=:SM, tol::Real=0.0,
                     restarts::Integer=20, ncv::Integer=0)
    which in (:SM, :LM) || throw(ArgumentError("which must be :SM or :LM, got $which"))
    nev > 0 || throw(ArgumentError("nev must be positive, got $nev"))
    ncv >= 0 || throw(ArgumentError("ncv must be 0 (automatic) or positive, got $ncv"))
    gf = GradientField(f; rotation=true)
    plan = Plan(f)
    writer = WriterVTK(f); saver = WriterSave(f)
    writers = WriterCollection([writer, saver])
    bfun() = similar(f.ϕ)
    return NumModelBdG{typeof(f),typeof(param),typeof(plan)}(
        f, gf, param, 0.0, niter, freqbckp, plan, writers,
        0.0, nev, which, tol, restarts, ncv,
        Float64[], Any[], Any[], bfun(), bfun())
end

function Base.show(io::IO, n::NumModelBdG)
    return print(io,
                 "Bogoliubov-de Gennes (matrix-free)\n",
                 "  ├──────  chemical potential: $(round(n.mu; digits=4))\n",
                 "  └──────────── eigen: nev = $(n.nev), which = $(n.which)")
end

# The standard driver `solve!` reports the energy of the field at every step.
# For the BdG model the "energy" is just the GP energy of the stationary
# state (constant), so the `GrossPitaevskii` energy dispatch (typed for
# GrossPitaevskiiParameters) does not apply; this overload returns the same
# 4-tuple (EΩ, EΔ, Eβ, E_total) so `solve!` works unchanged.
function energy(n::NumModelBdG, showEnergy=false)
    ϕ = n.f.ϕ
    coeffΔ, Ω, β = n.param.coeffΔ, n.param.Ω, n.param.β
    V = n.param.pot.V
    computeDerivatives!(n.gf, n.plan, ϕ)
    dv = n.f.g.Δx * n.f.g.Δy * (ndims(parent(ϕ)) == 3 ? n.f.g.Δz : 1.0)
    # sums over PencilArrays reduce across MPI ranks (parent() would sum the
    # local rank data only)
    EΩ = real(sum(im .* conj.(ϕ) .* (Ω .* (n.gf.rx .+ n.gf.ry)))) * dv
    lap = n.gf.ddx + n.gf.ddy
    if ndims(parent(ϕ)) == 3
        lap = lap + n.gf.ddz
    end
    EΔ = real(sum(conj.(ϕ) .* (n.param.coeffΔ * lap + V .* ϕ))) * dv
    Eβ = 0.5 * β * real(sum(abs2.(ϕ) .^ 2)) * dv
    E = -EΩ + EΔ + Eβ
    if showEnergy
        println_parallel("BdG stationary-state energy: $(E)")
    end
    return EΩ, EΔ, Eβ, E
end

# ---------------------------------------------------------------------------
# matrix-free operator
# ---------------------------------------------------------------------------

# Dispersive part of H: coeffΔ ∇² - iΩ L_z (single computeDerivatives! call).
# n.gf must be a GradientField with rotation=true.
function _lapRot(n::NumModelBdG, u)
    computeDerivatives!(n.gf, n.plan, u)
    tmp = similar(u)
    @. tmp = n.param.coeffΔ * (n.gf.ddx + n.gf.ddy) -
              n.param.Ω * im * (n.gf.rx + n.gf.ry)
    if ndims(parent(u)) == 3
        @. tmp += n.param.coeffΔ * n.gf.ddz
    end
    return tmp
end

"""
    bdg_mu(n::NumModelBdG)

Chemical potential of the stationary state,
``\\mu = \\langle\\psi, (\\text{coeffΔ}\\,\\nabla^2 + V + \\beta|\\psi|^2)\\,\\psi\\rangle``
(a real number, with the grid quadrature weights).
"""
function bdg_mu(n::NumModelBdG)
    ψ = n.f.ϕ
    computeDerivatives!(n.gf, n.plan, ψ)
    V, β, coeffΔ = n.param.pot.V, n.param.β, n.param.coeffΔ
    # the kinetic part is coeffΔ ∇² = coeffΔ (ddx + ddy [+ ddz]) (coeffΔ < 0)
    hψ = @. coeffΔ * (n.gf.ddx + n.gf.ddy) + V * ψ + β * abs2(ψ) * ψ
    if ndims(parent(ψ)) == 3
        hψ = @. hψ + coeffΔ * n.gf.ddz
    end
    if n.param.Ω != 0
        hψ = @. hψ - n.param.Ω * im * (n.gf.rx + n.gf.ry)
    end
    dv = n.f.g.Δx * n.f.g.Δy * (ndims(parent(ψ)) == 3 ? n.f.g.Δz : 1.0)
    return real(sum(conj.(ψ) .* hψ)) * dv / (sum(abs2.(ψ)) * dv)
end

"""
    bdg_apply!(n, out, u, v)

Apply the matrix-free BdG operator to `(u, v)`, storing the result in
`out = (Hu - Kv, K*u - Hv)` (two arrays with the layout of `f.ϕ`).

`H` is applied through the package's derivative machinery, so the bounded
(Dirichlet/Neumann) compact schemes are supported exactly like the periodic
ones.
"""
function bdg_apply!(n::NumModelBdG, out, u, v)
    ψ = n.f.ϕ
    V, β, μ = n.param.pot.V, n.param.β, n.mu
    # H u = (coeffΔ ∇² - iΩ L_z) u + (V + 2β|ψ|² - μ) u
    hu = _lapRot(n, u)
    @. hu += (V + 2β * abs2(ψ) - μ) * u
    # H v (H is real, so H* = H)
    hv = _lapRot(n, v)
    @. hv += (V + 2β * abs2(ψ) - μ) * v
    # out = (H u - K v, K* u - H v) with K = β ψ²
    @. out[1] = hu - β * ψ^2 * v
    @. out[2] = β * conj(ψ)^2 * u - hv
    return out
end

# Apply the operator to a stacked vector [u; v] (as used by the LinearMap).
# The package derivative machinery works on PencilArrays, so the views are
# copied into the scratch arrays `n.utmp`/`n.vtmp` (same layout as f.ϕ).
function _bdg_apply!(n::NumModelBdG, out, x)
    ψp = n.f.ϕ
    N = length(parent(ψp))
    copyto!(parent(n.utmp), view(x, 1:N))
    copyto!(parent(n.vtmp), view(x, N+1:2N))
    bdg_apply!(n, out, n.utmp, n.vtmp)
    return vcat(vec(parent(out[1])), vec(parent(out[2])))
end

# ---------------------------------------------------------------------------
# eigensolve
# ---------------------------------------------------------------------------

"""
    timeStep!(n::NumModelBdG)

Run the Bogoliubov-de Gennes eigensolve on the stationary state stored in
`n.f.ϕ`. Fills `n.mu`, `n.ωs`, `n.us` and `n.vs` and returns 0 for
compatibility with the standard solver loop.

The full 2N×2N operator is applied matrix-free (`_bdg_apply!`) through an
`Arpack.eigs` Krylov solve with `which = :SM` (smallest |ω|). `2·nev + 2`
eigenvalues are requested so that, in addition to the desired lowest modes,
the ω = 0 modes and at least one member of each ω ↔ −ω pair are captured:
the zero eigenspace is two-dimensional when `β = 0` (spanned by
`(ψ, ψ*)` and `(ψ, −ψ*)`), so two slots are reserved for it.
`bdg_output` then keeps the `n.nev` lowest positive-frequency physical modes
(the positive-symplectic-norm member of each pair, minus the zero mode).
"""
function timeStep!(n::NumModelBdG)
    n.mu = bdg_mu(n)
    if MPI.Comm_size(PencilArrays.get_comm(n.f.ϕ)) > 1
        return _timeStep_distributed!(n)
    end
    N = length(parent(n.f.ϕ))
    out = [similar(n.f.ϕ), similar(n.f.ϕ)]

    # matrix-free operator on the stacked vector [u; v]
    lmap = LinearMap{Complex{Float64}}(2N; issymmetric=false) do x
        _bdg_apply!(n, out, x)
    end

    nev2 = 2 * n.nev + 2
    ncv = n.ncv > 0 ? n.ncv : min(max(nev2 + 5, 20), 2N - 2)
    ew, ev = Arpack.eigs(lmap; nev=nev2, which=n.which,
                         tol=Float64(n.tol), ncv=ncv,
                         maxiter=n.restarts * 1000)
    ord = sortperm(abs.(real(ew)))
    n.ωs, n.us, n.vs = bdg_output(n, real(ew[ord]), ev[:, ord])
    return 0
end


# ---------------------------------------------------------------------------
# distributed (MPI) eigensolve
# ---------------------------------------------------------------------------
#
# Arpack is a serial library. Run independently on each rank it would build a
# different Krylov basis from the rank-local data (local inner products, local
# convergence tests) while the operator itself is a collective: the ranks drift
# apart and block in different collectives. Instead, rank 0 alone drives Arpack
# on the *global* 2N vector, and every other rank serves the mat-vec:
#
#     rank 0:  Bcast(command = apply); Bcast(x); scatter x; apply; gather y
#     others:  loop { Bcast(command); if stop -> break; Bcast(x); scatter; apply; gather }
#
# so the control flow is decided in one place by construction (no reliance on
# bit-identical floating point across ranks). Memory: the global vectors live on
# rank 0 only. The cost is an O(N_global) gather per mat-vec, which is fine for
# the few hundred mat-vecs of a BdG solve but makes this path a correctness
# feature, not a scalable one. A fully distributed Krylov solver would require
# global inner products (e.g. KrylovKit with PencilArray-aware vectors).
#
# The modes `n.us`/`n.vs` are returned as *global* arrays on every rank.
#
# Assumes the pencil has no index permutation (always the case for `Field`),
# i.e. memory order == logical order, and CPU arrays (Arpack is CPU-only).

function _bdg_replicate(x::PencilArray)
    comm = PencilArrays.get_comm(x)
    g = PencilArrays.gather(x, 0)
    G = g === nothing ? Array{eltype(x)}(undef, PencilArrays.size_global(x)) : g
    MPI.Bcast!(G, 0, comm)
    return G
end

# Collective: scatter the global stacked vector `xg` into the local `utmp`/`vtmp`,
# apply the operator, and gather the result. Returns the stacked global result
# on rank 0 and `nothing` elsewhere.
function _bdg_apply_global!(n::NumModelBdG, out, xg::AbstractVector)
    dims = PencilArrays.size_global(n.f.ϕ)
    Ng = prod(dims)
    r = PencilArrays.range_local(n.utmp, PencilArrays.LogicalOrder())
    parent(n.utmp) .= view(reshape(view(xg, 1:Ng), dims), r...)
    parent(n.vtmp) .= view(reshape(view(xg, (Ng + 1):(2Ng)), dims), r...)
    bdg_apply!(n, out, n.utmp, n.vtmp)
    gu = PencilArrays.gather(out[1], 0)
    gv = PencilArrays.gather(out[2], 0)
    return gu === nothing ? nothing : vcat(vec(gu), vec(gv))
end

function _timeStep_distributed!(n::NumModelBdG)
    T = Complex{Float64}
    comm = PencilArrays.get_comm(n.f.ϕ)
    rank = MPI.Comm_rank(comm)
    dims = PencilArrays.size_global(n.f.ϕ)
    Ng = prod(dims)
    out = [similar(n.f.ϕ), similar(n.f.ϕ)]
    nev2 = 2 * n.nev + 2
    ncv = n.ncv > 0 ? n.ncv : min(max(nev2 + 5, 20), 2Ng - 2)

    ew = Vector{T}(undef, 0)
    ev = Matrix{T}(undef, 2Ng, 0)
    cmd = Ref(Int32(0))      # 1 = apply the operator, 0 = stop
    ok = Ref(Int32(1))       # 1 = rank 0 finished without error
    if rank == 0
        try
            lmap = LinearMap{T}(2Ng; issymmetric=false) do x
                cmd[] = 1
                MPI.Bcast!(cmd, 0, comm)
                xb = Vector{T}(x)
                MPI.Bcast!(xb, 0, comm)
                return _bdg_apply_global!(n, out, xb)
            end
            ew, ev = Arpack.eigs(lmap; nev=nev2, which=n.which,
                                 tol=Float64(n.tol), ncv=ncv,
                                 maxiter=n.restarts * 1000)
        catch
            ok[] = 0
            rethrow()
        finally
            # always release the other ranks, also when Arpack threw
            cmd[] = 0
            MPI.Bcast!(cmd, 0, comm)
            MPI.Bcast!(ok, 0, comm)
        end
    else
        while true
            MPI.Bcast!(cmd, 0, comm)
            cmd[] == 0 && break
            xb = Vector{T}(undef, 2Ng)
            MPI.Bcast!(xb, 0, comm)
            _bdg_apply_global!(n, out, xb)
        end
        MPI.Bcast!(ok, 0, comm)
        ok[] == 1 || error("BdG eigensolve failed on rank 0 (see its error)")
    end

    # share the converged eigenpairs with every rank
    m = Ref(Int32(length(ew)))
    MPI.Bcast!(m, 0, comm)
    if rank != 0
        ew = Vector{T}(undef, m[])
        ev = Matrix{T}(undef, 2Ng, m[])
    end
    MPI.Bcast!(ew, 0, comm)
    MPI.Bcast!(ev, 0, comm)

    ψg = _bdg_replicate(n.f.ϕ)
    ord = sortperm(abs.(real(ew)))
    n.ωs, n.us, n.vs = bdg_output(n, real(ew[ord]), ev[:, ord]; ψ=ψg)
    return 0
end

"""
    bdg_output(n, ωs, ev; zero_ov=0.15)

Select and normalize the physical BdG modes from the raw eigenvectors.
`ev` holds one 2N eigenvector per column (`u` in the first half, `v` in the
second). A column is kept when it belongs to the ω > 0 branch, which is
detected by a positive symplectic norm s = ‖u‖² - ‖v‖² (the ω < 0 partner has
s < 0). The ω = 0 mode (u, v) = (ψ, ψ*) is dropped by its *structure*, not its
frequency: it is (dominantly) the vector (ψ, ψ*), so any column whose overlap
with that vector exceeds `zero_ov` is rejected. Using overlap (rather than
|ω| ≈ 0) makes this robust when the supplied stationary state is not an exact
eigenstate of the discrete operator — in that case the zero mode appears as a
small-residual pair at ω ≈ ±ε and its frequency is not reliably zero. Kept
modes are renormalized to s = 1 and returned sorted by increasing ω.
"""
function bdg_output(n::NumModelBdG, ωs, ev; zero_ov=0.9, ψ=parent(n.f.ϕ))
    # `ψ` is the stationary state matching the layout of `ev`: the local data on one
    # rank (default), or the full replicated array when the eigenvectors are global
    # (distributed solve).
    ψp = ψ
    N = length(ψp)
    ψn = sqrt(sum(abs2, ψp))
    dv = n.f.g.Δx * n.f.g.Δy * (ndims(ψp) == 3 ? n.f.g.Δz : 1.0)
    m = size(ev, 2)
    us, vs = Any[], Any[]
    w = Float64[]
    for j in 1:m
        col = ev[:, j]
        s = (sum(abs2, col[1:N]) - sum(abs2, col[N+1:2N])) * dv
        s > 0 || continue              # ω < 0 branch (s < 0)
        un, vn = sqrt(sum(abs2, col[1:N])), sqrt(sum(abs2, col[N+1:2N]))
        (un > 0 && vn > 0) || continue
        # Structural zero-mode test: the gauge mode is (u, v) ∝ (ψ, ψ*), i.e.
        # BOTH components are parallel to ψ. Measured (β=200, dense spectrum):
        # the zero mode gives ov_u = ov_v = 1.0, while the breathing mode —
        # the physical mode most collinear with ψ — reaches only ov_u = 0.43,
        # ov_v = 0.68. An earlier version thresholded the overlap of the
        # *stacked* vector with (ψ, ψ*); at 0.15 that silently deleted the
        # breathing mode (ov = 0.52), which is an exact ω = 2 eigenvalue
        # (hidden symmetry) and the package's own reference case.
        ov_u = abs(sum(conj.(vec(ψp)) .* col[1:N])) / (un * ψn)
        ov_v = abs(sum(vec(ψp) .* col[N+1:2N])) / (vn * ψn)
        max(ov_u, ov_v) < zero_ov || continue             # zero mode
        u = reshape(col[1:N], size(ψp))
        v = conj.(reshape(col[N+1:2N], size(ψp)))
        push!(w, ωs[j])
        push!(us, u ./ sqrt(s))        # renormalize to ‖u‖² - ‖v‖² = 1
        push!(vs, v ./ sqrt(s))
    end
    ord = sortperm(w)
    # keep only the n.nev lowest modes (a degenerate cluster can yield more)
    k = min(n.nev, length(ord))
    return w[ord[1:k]], us[ord[1:k]], vs[ord[1:k]]
end
