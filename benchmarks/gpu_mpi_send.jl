# Decisive test for the GPU+MPI fault: does MPI_Send/MPI_Recv accept device
# pointers (CUDA-aware MPI)? The pencil transposes inside Plan/FFT exchange
# PencilArrays between ranks; if those buffers live on the GPU and the MPI
# build is not CUDA-aware, the send dereferences a device pointer as host and
# faults — which is exactly where the staged probe (gpu_mpi_stages.jl) dies.
#
#   mpiexec -n 2 julia --project=. -t1 -O3 benchmarks/gpu_mpi_send.jl

using MPI
MPI.Init()

comm = MPI.COMM_WORLD
rank = MPI.Comm_rank(comm)
nranks = MPI.Comm_size(comm)
nranks == 2 || (println("2 ranks required"); MPI.Finalize(); return)

HAVE_GPU = try
    using CUDA
    CUDA.functional()
catch
    false
end

using SuperfluidDynamics   # for the PencilArray layout
N = 16

probe(label, sendbuf, recvbuf) = try
    if rank == 0
        MPI.Send(sendbuf, 1, 0, comm)
    else
        MPI.Recv!(recvbuf, 0, 1, comm)
    end
    MPI.Barrier(comm)
    println("rank $rank $label: OK")
catch e
    MPI.Barrier(comm)
    println("rank $rank $label: ECHEC ", typeof(e))
end
flush(stdout)

# host control
h = fill(Float64(rank + 1), 64)
hr = fill(0.0, 64)
probe("send-host", h, hr)

if HAVE_GPU
    g = CUDA.CuArray(fill(Float64(rank + 1), 64))
    gr = CUDA.CuArray(zeros(Float64, 64))
    CUDA.synchronize()
    MPI.Barrier(comm)
    probe("send-device", g, gr)
end
MPI.Finalize()
