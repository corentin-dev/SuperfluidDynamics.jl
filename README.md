# SuperfluidDynamics.jl

[![Pipeline Status](https://plmlab.math.cnrs.fr/lothode/SuperfluidDynamics.jl/badges/master/pipeline.svg)](https://plmlab.math.cnrs.fr/lothode/SuperfluidDynamics.jl/-/pipelines)
[![Latest Release](https://plmlab.math.cnrs.fr/lothode/SuperfluidDynamics.jl/-/badges/release.svg)](https://plmlab.math.cnrs.fr/lothode/SuperfluidDynamics.jl/-/releases)
[![Documentation](https://img.shields.io/badge/documentation-online-blue.svg)](https://lothode.pages.math.cnrs.fr/SuperfluidDynamics.jl)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

> **The official repository is hosted on [plmlab.math.cnrs.fr](https://plmlab.math.cnrs.fr/lothode/SuperfluidDynamics.jl)** (this repository is also mirrored on GitHub). Please open issues and merge requests on the official repository.

This is a package allowing simulation of superfluids. The first intention of this package is to solve the Gross-Pitaevskii equation to simulation Bose-Einstein Condensates. It evolved into a more advance package in order to solve Quantum-Turbulence. It now also solves the incompressible Navier-Stokes equations, the coupled Gross-Pitaevskii / Navier-Stokes two-fluid model of Parnaudeau et al. (NSGP), and the linear Hall-Vinen-Bekarevich-Khalatnikov (HVBK) two-fluid model. Derivatives are estimated through Fourier transformations or finite differences.

In order to be parallel (distributed), this package exploits intensively `PencilArrays`. Most of the package is written using broadcast, and is compatible with both CPU arrays (`Array`) and CUDA arrays (`CuArray`). It was not tested for other array type, yet. Every array creation is inferred from the `Grid` array type.

This package is authored by Corentin Lothodé, and largely inspired by GPS, a Fortran program by Philippe Parnaudeau (see [Acknowledgements](#acknowledgements)).

## Models

- **Gross-Pitaevskii**: imaginary time (backward Euler, Crank-Nicolson, ADI) and real time (time-dependent Crank-Nicolson), plus an external-velocity solver. See `NumModelBackwardEuler`, `NumModelCrankNicolson`, `NumModelADI1`, `NumModelCrankNicolsonT`, `NumModelExternalVelocity`.
- **Bogoliubov-de Gennes** (`NumModelBdG`): matrix-free eigensolver for the linearized excitations about a stationary GP state. The `2N x 2N` operator is applied through the derivative machinery (no dense matrix); the zero mode is rejected by overlap with `(ψ₀, ψ₀*)` and the modes are returned symplectically normalized.
- **Navier-Stokes** (incompressible, 2D and 3D): semi-implicit RK4 solver `NumModelRK4Imp` (vorticity-advection form, exact implicit viscous multiplier, spectral Helmholtz projection).
- **NSGP** (coupled GP/Navier-Stokes two-fluid model of Parnaudeau et al., `NumModelNSGP`): a non-stationary Gross-Pitaevskii equation for the superfluid wavefunction coupled, through the Coste coupling, to a forced Navier-Stokes equation for the normal fluid (one-way or two-way).
- **HVBK** (linear two-fluid model, `NumModelHBVK`): two incompressible velocity fields coupled by the linear mutual friction `F = -1/2 rb |∇×u_s| (u_n - u_s)`, total momentum conserved.

## Get package

```
git clone git@plmlab.math.cnrs.fr:lothode/SuperfluidDynamics.jl.git
```

Start Julia :
```
julia --project=.
```

Import package :
```
using SuperfluidDynamics
```

## `MPI` and `HDF5`

This project uses  `MPIPreferences.jl` to setup `MPI.jl`. In order to use it, you can create a file named `LocalPreferences.toml` containing:

```
[MPIPreferences]
_format = "1.0"
abi = "OpenMPI"
binary = "system"
libmpi = "libmpi"
mpiexec = "mpiexec"
```

To the same `LocalPreferences.toml`, you can add the following informations:

```
[HDF5_jll]
libhdf5_path = "/usr/lib/libhdf5.so"
libhdf5_hl_path = "/usr/lib/libhdf5_hl.so"
```

## Acknowledgements

This package is largely inspired by **GPS**, a Fortran program by Philippe Parnaudeau, and by the work it enabled:

- P. Parnaudeau, A. Suzuki, and J.-M. Sac-Epée, "GPS: An efficient & spectrally accurate code for computing Gross-Pitaevskii equation", *ISC-2015, Research Posters Session*, 2015.
- M. Brachet, G. Sadaka, Z. Zhang, V. Kalt, and I. Danaila, "Coupling Navier-Stokes and Gross-Pitaevskii equations for the numerical simulation of two-fluid quantum flows", *Journal of Computational Physics*, 488, 112193, 2023.
- Z. Zhang, I. Danaila, E. Lévéque, and L. Danaila, "Higher-order statistics and intermittency of a two-fluid Hall–Vinen–Bekarevich–Khalatnikov quantum turbulent flow", *Journal of Fluid Mechanics*, 962, A22, 2023.

The machine-readable version of these references is available in [`docs/references.bib`](docs/references.bib).
