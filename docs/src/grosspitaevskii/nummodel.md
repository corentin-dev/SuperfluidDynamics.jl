# Numerical schemes

## Gross-Pitaevskii

### Imaginary time

```@docs
NumModelBackwardEuler
NumModelBackwardEulerNoPrecond
NumModelCrankNicolson
NumModelCrankNicolsonQuasiNewton
```

### Real time

```@docs
NumModelSplit1
NumModelSplit2
NumModelCrankNicolsonT
NumModelCrankNicolsonQuasiNewtonT
NumModelGPRK
```

`NumModelGPRK` is the **explicit** Runge-Kutta integrator (RK1/RK2/RK4, port of
the reference `GP_RK4`), as opposed to the implicit/splitting schemes above.
It is unconditionally *accurate* but subject to the dispersive stability limit
`|coeffΔ| k_max² Δt ≲ 2` (RK1) / `≲ 2.8` (RK4), so it is intended for
benchmarks, short runs and as an accuracy reference; the implicit schemes are
preferred for long, high-resolution evolutions.

### External velocity

```@docs
NumModelExternalVelocity
```
