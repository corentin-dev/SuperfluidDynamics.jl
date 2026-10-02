# Does the GPU path work under MPI? The docs sell the GPU as single-process;
# this probe answers whether multi-rank CuArray fields are usable — removing
# that limitation if it works, documenting a real bug if it does not.
#
#   mpiexec -n 2 julia --project=. -t1 -O3 benchmarks/gpu_mpi.jl     (2 GPUs)
#
# Each rank pins one GPU via CUDA_VISIBLE_DEVICES (set from SLURM_LOCALID
# BEFORE importing CUDA — CUDA.jl reads it at first import). Probes, per rank:
#   1. spectral derivatives of a scalar CuArray field under MPI decomposition
#   2. one NumModelRK4Imp step on CuArray (the full NS solver on GPU)
# Environment: GPU_MPI_N (default 32).

using MPI
MPI.Init()

comm = MPI.COMM_WORLD
rank = MPI.Comm_rank(comm)
nranks = MPI.Comm_size(comm)

# Pin one device per local rank before CUDA is imported (the node has two GPUs
# per node here; without this both ranks share device 0).
# True local rank via MPI-3 shared-node split: SLURM_LOCALID is not exported
# to tasks when HYDRA (MPICH_jll's mpiexec) is the launcher, and using the
# global rank pinned BOTH ranks to device 0.
local_comm = MPI.Comm_split_type(comm, MPI.COMM_TYPE_SHARED, rank)
local_id = MPI.Comm_rank(local_comm)
vis = get(ENV, "CUDA_VISIBLE_DEVICES", "")
gpu_list = isempty(vis) ? nothing : split(vis, ',')
if gpu_list !== nothing
    ENV["CUDA_VISIBLE_DEVICES"] = gpu_list[(local_id % length(gpu_list)) + 1]
end

HAVE_GPU = try
    using CUDA
    CUDA.functional()
catch
    false
end

if !HAVE_GPU
    println("rank $rank/$nranks gpu-mpi: ABSENT (no CUDA device visible)")
    MPI.Finalize()
    return
end

using SuperfluidDynamics
using Printf: @printf

N = parse(Int, get(ENV, "GPU_MPI_N", "32"))

say(msg) = (@printf("rank %d/%d %s\n", rank, nranks, msg); flush(stdout))

# 1. spectral derivatives on an MPI-decomposed CuArray field
try
    grid = Grid((N, N, N), ((-4, 4), (-4, 4), (-4, 4)); array_type=CuArray)
    f = Field(grid, ComplexField())
    f.ϕ .= exp.(1im .* (0.3 .* f.x .+ 0.5 .* f.y .+ 0.7 .* f.z))
    gf = GradientField(f; rotation=false)
    SuperfluidDynamics.computeDerivatives!(gf, Plan(f), f.ϕ)
    CUDA.synchronize()
    MPI.Barrier(comm)
    say("gpu-mpi[deriv]: OK local_size=" * string(size(parent(f.ϕ))))
catch e
    MPI.Barrier(comm)
    msg = first(split(sprint(showerror, e), "\n"))
    say("gpu-mpi[deriv]: ECHEC " * string(typeof(e)) * ": " * msg[1:min(lastindex(msg), 140)])
end

# 2. the NS solver itself on CuArray: one correctness step, then a timed
# window when STEPS>0 (multi-rank GPU scaling point for results/scaling.md).
STEPS = parse(Int, get(ENV, "GPU_MPI_STEPS", "0"))
try
    grid = Grid((N, N, N), ((-4, 4), (-4, 4), (-4, 4)); array_type=CuArray)
    f = Field(grid, ComplexField(); ndims=3)
    taylor_green!(f, f.x, f.y, f.z)
    model = NumModelRK4Imp(f, NavierStokesParameters(; ν=0.0), 0.01, 1, 1)
    SuperfluidDynamics.timeStep!(model)
    CUDA.synchronize()
    MPI.Barrier(comm)
    say("gpu-mpi[ns-step]: OK")
    if STEPS > 0
        t0 = time_ns()
        for _ in 1:STEPS
            SuperfluidDynamics.timeStep!(model)
        end
        CUDA.synchronize()
        MPI.Barrier(comm)
        dt = (time_ns() - t0) / 1e9 / STEPS
        dt_max = MPI.Allreduce(dt, max, comm)
        if rank == 0
            @printf("nranks=%d  bench=gpu  N=%d  steps=%d  s/step=%.5f  Mpts/s=%.2f\n",
                    nranks, N, STEPS, dt_max, N^3 / 1e6 / dt_max)
        end
    end
catch e
    MPI.Barrier(comm)
    msg = first(split(sprint(showerror, e), "\n"))
    say("gpu-mpi[ns-step]: ECHEC " * string(typeof(e)) * ": " * msg[1:min(lastindex(msg), 140)])
end
MPI.Finalize()
