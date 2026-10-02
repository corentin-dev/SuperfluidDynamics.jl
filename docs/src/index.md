# SuperfluidDynamics.jl

**SuperfluidDynamics.jl** is a Julia package for simulating superfluids. It started as
a Gross-Pitaevskii (GP) solver for Bose-Einstein condensates and has grown into
a general library for quantum and classical fluids. It currently supports:

- **Gross-Pitaevskii** — imaginary-time solvers for stationary states and
  real-time solvers for dynamical evolutions,
- **Bogoliubov-de Gennes** — a matrix-free eigensolver for the linearized
  excitations about a stationary GP state,
- **Navier-Stokes** — the incompressible equations in 2D and 3D,
- **Two-fluid models** — the coupled Gross-Pitaevskii / Navier-Stokes model
  (NSGP) and the linear Hall-Vinen-Bekarevich-Khalatnikov model (HVBK).

Derivatives are computed with Fourier transforms or finite differences. The
package is parallel (distributed) on top of [`PencilArrays`](https://github.com/PencilArrays/PencilArrays.jl); most of it is written with broadcasting and
works on both CPU arrays (`Array`) and CUDA arrays (`CuArray`).

This package is authored by Corentin Lothodé, and is largely inspired by
**GPS**, a Fortran code by Philippe Parnaudeau.

## Installation

Once registered in the General registry:

```
pkg> add SuperfluidDynamics
```

Until then (and to work on the sources), add it from the repository — HTTPS
works without an SSH key:

```
pkg> add https://plmlab.math.cnrs.fr/lothode/SuperfluidDynamics.jl
```

or clone it and work from a local project:

```bash
git clone https://plmlab.math.cnrs.fr/lothode/SuperfluidDynamics.jl.git
cd SuperfluidDynamics.jl
julia --project=.
```

```julia-repl
julia> using SuperfluidDynamics
```

See the [Quick start](@ref) for a minimal Gross-Pitaevskii simulation, and the
**Physic models** section for each model in detail:
[Gross-Pitaevskii](grosspitaevskii/grosspitaevskii.md),
[Bogoliubov-de Gennes](bdg.md) and
[Navier-Stokes / two-fluid models](navierstokes.md).

## Building the documentation

```bash
julia --project --color=yes docs/make.jl
```

The generated site is written under `docs/build/`.
