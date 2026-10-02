# Full-backtrace reproduction of the NS-on-GPU failure seen by gpu_mpi.jl:
# one NumModelRK4Imp step on a CuArray field, error printed with its stack
# trace (gpu_mpi.jl truncates the message to keep its log compact).
#   julia --project=. -t1 -O3 benchmarks/gpu_ns_trace.jl
using SuperfluidDynamics
using CUDA

N = parse(Int, get(ENV, "GPU_MPI_N", "64"))
grid = Grid((N, N, N), ((-4, 4), (-4, 4), (-4, 4)); array_type=CuArray)
f = Field(grid, ComplexField(); ndims=3)
taylor_green!(f, f.x, f.y, f.z)
model = NumModelRK4Imp(f, NavierStokesParameters(; ν=0.0), 0.01, 1, 1)
try
    SuperfluidDynamics.timeStep!(model)
    CUDA.synchronize()
    println("ns-gpu: OK")
catch e
    println("ns-gpu: ECHEC")
    showerror(stdout, e, catch_backtrace())
    println()
end
