# CI image for the SuperfluidDynamics pipelines (unit tests + documentation).
#
# Bakes in, once: system libraries (OpenMPI + HDF5), Julia 1.12.7, the
# project dependencies (instantiated from the committed Manifest and built
# against the system HDF5/OpenMPI), the test environment, and the
# documentation stack (Documenter/Literate/Makie). Jobs then only pay for
# pulling the image and a fast `Pkg.instantiate()` against the warm
# ~/.julia cache, instead of ~10 minutes of apt + download + precompilation
# per job.
#
# Rebuild when this file, Project.toml or Manifest.toml change (the CI job
# hashes the three and skips when the image is already on the registry).
FROM ubuntu:26.04

ARG JULIA_VERSION=1.12.7
# Floor version advertised in [compat]; baked so the test-1.10 job does not
# depend on julialang-s3 reachability at job time (reviewer point: behind a
# proxy the download would fail for reasons unrelated to the code). 1.10.12 =
# the patch level where the suite is green (1.10.9 fails GPUCompiler
# precompilation, verified in the CI trace).
ARG JULIA_FLOOR_VERSION=1.10.12

RUN apt-get update -qq \
    && apt-get install -y --no-install-recommends \
        git curl wget ca-certificates \
        git-lfs \
        libhdf5-openmpi-dev libopenmpi-dev \
    && rm -rf /var/lib/apt/lists/*

# The doc build reads checkpoints from docs/save/ (git-LFS), so make sure the
# checkout materialises LFS objects instead of leaving 131-byte pointers.
RUN git lfs install --system

RUN wget -q https://julialang-s3.julialang.org/bin/linux/x64/1.12/julia-${JULIA_VERSION}-linux-x86_64.tar.gz \
    && tar -xzf julia-${JULIA_VERSION}-linux-x86_64.tar.gz -C /opt/ \
    && ln -s /opt/julia-${JULIA_VERSION}/bin/julia /usr/local/bin/julia \
    && rm julia-${JULIA_VERSION}-linux-x86_64.tar.gz

RUN wget -q https://julialang-s3.julialang.org/bin/linux/x64/1.10/julia-${JULIA_FLOOR_VERSION}-linux-x86_64.tar.gz \
    && tar -xzf julia-${JULIA_FLOOR_VERSION}-linux-x86_64.tar.gz -C /opt/ \
    && ln -s /opt/julia-${JULIA_FLOOR_VERSION} /opt/julia-1.10 \
    && rm julia-${JULIA_FLOOR_VERSION}-linux-x86_64.tar.gz

# --- main environment ---------------------------------------------------------
# Instantiate from the committed Manifest (deterministic) and build the
# MPI/HDF5 bindings against the system libraries. `src` is needed because the
# test environment path-depends on the root project.
WORKDIR /builds/project
COPY Project.toml Manifest.toml ./
COPY src/ src/
# Kept for continuity; recent MPI.jl/HDF5.jl resolve through Preferences.jl.
ENV JULIA_MPI_BINARY=system \
    JULIA_HDF5_PATH=/usr/lib/x86_64-linux-gnu/hdf5/openmpi/
RUN julia -e 'using Pkg; Pkg.activate("."); Pkg.instantiate(); Pkg.build()'

# --- test environment --------------------------------------------------------
# Test dependencies live in [extras]/[targets] of the root Project.toml, so the
# test environment is the root environment: precompiling the package is enough
# for both `julia --project=. test/runtests.jl` and `Pkg.test()`.
RUN julia -e 'using Pkg; Pkg.activate("."); Pkg.precompile()'

# --- documentation stack (installed into the root environment, as CI does) ---
RUN julia -e 'using Pkg; Pkg.activate("."); \
               Pkg.add("Documenter"); Pkg.add("Literate"); Pkg.add("WGLMakie"); \
               Pkg.add("JSServe"); Pkg.add("CairoMakie")'
