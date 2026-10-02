# Decisive test for the GPU+MPI fault: does MPI accept device pointers
# (CUDA-aware MPI)? The pencil transposes inside Plan/FFT exchange PencilArrays
# between ranks; if those buffers live on the GPU and the MPI build is not
# CUDA-aware, the exchange faults — exactly where the staged probe dies.
#
# Structured so a failure on one rank cannot hang the other: the buffer
# construction (where MPI.jl rejects unsupported array types) is tested and
# synchronised BEFORE any blocking point-to-point call.
#
#   mpiexec -n 2 julia --project=. -t1 -O3 benchmarks/gpu_mpi_send.jl

using MPI
MPI.Init()

comm = MPI.COMM_WORLD
rank = MPI.Comm_rank(comm)
nranks = MPI.Comm_size(comm)

report(msg) = (println("rank $rank/$nranks $msg"); flush(stdout))

if nranks != 2
    rank == 0 && report("2 ranks required")
    MPI.Finalize()
    return
end

HAVE_GPU = try
    using CUDA
    CUDA.functional()
catch
    false
end

# 1. Buffer construction — where MPI.jl rejects unsupported array types.
host_ok = try
    MPI.Buffer(fill(1.0, 64))
    true
catch e
    report("buffer-host: ECHEC $(typeof(e))")
    false
end
MPI.Barrier(comm)

dev_ok = false
if HAVE_GPU
    dev_ok = try
        MPI.Buffer(CUDA.CuArray(fill(1.0, 64)))
        true
    catch e
        report("buffer-device: ECHEC $(typeof(e)): " *
               first(split(sprint(showerror, e), "\n"))[1:min(end, 120)])
        false
    end
end
MPI.Barrier(comm)

# 2. Only exchange if BOTH ranks could build the buffers, so no rank waits on a
# peer that dropped out.
ok = MPI.Allreduce([host_ok ? 1 : 0], &, comm)[1] == 1
if ok
    if rank == 0
        MPI.Send(fill(Float64(rank + 1), 64), 1, 0, comm)
    else
        MPI.Recv!(zeros(64), 0, 0, comm)
    end
    report("send-host: OK")
end
MPI.Barrier(comm)

if HAVE_GPU
    okdev = MPI.Allreduce([dev_ok ? 1 : 0], &, comm)[1] == 1
    if okdev
        if rank == 0
            MPI.Send(CUDA.CuArray(fill(Float64(rank + 1), 64)), 1, 0, comm)
        else
            MPI.Recv!(CUDA.CuArray(zeros(64)), 0, 0, comm)
            CUDA.synchronize()
        end
        report("send-device: OK")
    else
        report("send-device: NON TESTABLE (buffer refusé)")
    end
end
MPI.Barrier(comm)
report("fin")
MPI.Finalize()
