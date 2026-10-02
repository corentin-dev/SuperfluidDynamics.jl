# Staged probe: where exactly does GPU + MPI break? Each stage prints a marker
# before running, so the last marker in the log localises the fault. Controls
# (CPU at the same stage) separate "MPI decomposition" from "CUDA arrays".
#
#   GPU_MPI_CPU=1  mpiexec -n 2 julia --project=. -t1 -O3 benchmarks/gpu_mpi_stages.jl
#   mpiexec -n 2 julia --project=. -t1 -O3 benchmarks/gpu_mpi_stages.jl
#
# Environment: GPU_MPI_N (default 32), GPU_MPI_CPU=1 forces Array (CPU control).

using MPI
MPI.Init()

comm = MPI.COMM_WORLD
rank = MPI.Comm_rank(comm)
nranks = MPI.Comm_size(comm)

local_id = parse(Int, get(ENV, "SLURM_LOCALID", string(rank)))
vis = get(ENV, "CUDA_VISIBLE_DEVICES", "")
if !isempty(vis)
    gpu_list = split(vis, ',')
    ENV["CUDA_VISIBLE_DEVICES"] = gpu_list[(local_id % length(gpu_list)) + 1]
end

use_gpu = get(ENV, "GPU_MPI_CPU", "0") != "1"
HAVE_GPU = if use_gpu
    try
        using CUDA
        CUDA.functional()
    catch
        false
    end
else
    false
end

using SuperfluidDynamics
N = parse(Int, get(ENV, "GPU_MPI_N", "32"))
AT = HAVE_GPU ? CuArray : Array

step(msg) = (println("rank $rank/$nranks [$msg]"); flush(stdout); MPI.Barrier(comm))

step("0 start: at=$(AT) gpu=$HAVE_GPU local_id=$local_id")
grid = Grid((N, N, N), ((-4, 4), (-4, 4), (-4, 4)); array_type=AT)
step("1 Grid ok")

f = Field(grid, ComplexField())
step("2 Field ok")

f.ϕ .= exp.(1im .* (0.3 .* f.x .+ 0.5 .* f.y))
step("3 fill ok")

gf = GradientField(f; rotation=false)
step("4 GradientField ok")

p = Plan(f)
step("5 Plan ok")

SuperfluidDynamics.computeDerivatives!(gf, p, f.ϕ)
step("6 computeDerivatives ok")

HAVE_GPU && CUDA.synchronize()
step("7 synchronize ok")

E = sum(abs2.(parent(f.ϕ)))
step("8 reduction ok (E=$E)")
MPI.Finalize()
