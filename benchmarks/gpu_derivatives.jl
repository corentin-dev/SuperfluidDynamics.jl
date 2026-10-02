# GPU benchmark: wall time of computeDerivatives! for the compact scheme and
# the spectral (FFT) scheme, CPU arrays vs CuArray, 2D and 3D. Single process
# (no MPI). This is the performance counterpart of the "CompactPlan on GPU
# arrays" correctness test set.
#
#   julia --project=. -t8 -O3 benchmarks/gpu_derivatives.jl
#
# Sizes can be overridden: GPX_N2D=1024 GPX_N3D=128 julia ... (3D needs
# enough device memory: 128^3 complex128 ~ 256 MB per buffer).

using SuperfluidDynamics
using Printf: @printf

HAVE_GPU = try
    using CUDA
    CUDA.functional()
catch
    false
end

n2d = parse(Int, get(ENV, "GPX_N2D", "1024"))
n3d = parse(Int, get(ENV, "GPX_N3D", "128"))
reps = parse(Int, get(ENV, "GPX_REPS", "20"))

"Seconds per computeDerivatives! call, after 3 untimed warm-up calls."
function time_derivatives(f, plan, gf)
    sync() = HAVE_GPU && CUDA.synchronize()
    for _ in 1:3
        SuperfluidDynamics.computeDerivatives!(gf, plan, f.ϕ)
    end
    sync()
    t0 = time_ns()
    for _ in 1:reps
        SuperfluidDynamics.computeDerivatives!(gf, plan, f.ϕ)
    end
    sync()
    return (time_ns() - t0) / 1e9 / reps
end

function fill!(f)
    # a smooth non-separable field, so no derivative is trivially zero (same
    # idiom as the GPU test set: axes broadcast through the field layout)
    z = length(f.g.data) == 3 ? 0.9 .* f.z : 0.0
    f.ϕ .= exp.(1im .* (0.5 .* f.x .+ 0.7 .* f.y .+ z)) .*
           (1.0 .+ 0.3 .* sin.(0.31 .* f.x) .* cos.(0.42 .* f.y))
end

function trial(label, gridsize, bounds; gpu::Bool)
    AT = gpu ? CuArray : Array
    g = Grid(gridsize, bounds; array_type=AT)
    f = Field(g, ComplexField())
    fill!(f)
    pc = Plan(f; t=SuperfluidDynamics.CompactPlan())
    gf = GradientField(f; rotation=false)
    t_compact = time_derivatives(f, pc, gf)
    pf = Plan(f)
    t_fft = try
        time_derivatives(f, pf, gf)
    catch e
        NaN
    end
    @printf("%-28s compact=%8.4f ms   fft=%8.4f ms\n", label, 1e3 * t_compact, 1e3 * t_fft)
end

@printf("gpu_derivatives: have_gpu=%s reps=%d\n", HAVE_GPU, reps)
trial("2D $(n2d)^2 CPU", (n2d, n2d), ((-4, 4), (-4, 4)); gpu=false)
HAVE_GPU && trial("2D $(n2d)^2 GPU", (n2d, n2d), ((-4, 4), (-4, 4)); gpu=true)
trial("3D $(n3d)^3 CPU", (n3d, n3d, n3d), ((-4, 4), (-4, 4), (-4, 4)); gpu=false)
if HAVE_GPU
    try
        trial("3D $(n3d)^3 GPU", (n3d, n3d, n3d), ((-4, 4), (-4, 4), (-4, 4)); gpu=true)
    catch e
        @printf("%-28s skipped: %s\n", "3D GPU", first(split(sprint(showerror, e), "\n")))
    end
end
