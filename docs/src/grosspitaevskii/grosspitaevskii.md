# Gross-Pitaevskii equation

The [Gross-Pitaevskii equation](https://en.wikipedia.org/wiki/Gross%E2%80%93Pitaevskii_equation) is a nonlinear Schrödinger equation used to model Bose-Einstein condensates at $0\,\mathrm{K}$:

$$i \dfrac{dϕ}{dt} = αΔϕ +  β |ϕ|²ϕ  + V(\vec{x})ϕ - i Ω L_z ϕ$$

with:

- $α$: `coeffΔ` (default $-\frac{1}{2}$, must be negative),
- $β$: `β`, the interaction coefficient,
- $V(\vec{x})$: `pot`, a potential (see [Potentials](potential.md)),
- $Ω$: `Ω`, the rotation coefficient about the $z$ axis.

Stationary states are obtained by evolving the equation in _imaginary_ time,
which amounts to a gradient descent on the energy. The solvers used for this,
as well as the real-time schemes, are described in the [Numerical schemes](nummodel.md) section.

## Parameters

### Constructor

```@docs
GrossPitaevskiiParameters(; :: Real, :: Real, :: AbstractPotential, :: Real)
```

### Structure

```@docs
SuperfluidDynamics.GrossPitaevskiiParameters
```

## Energy

```@docs
SuperfluidDynamics.energy
```

## Utilities

```@docs
SuperfluidDynamics.lapRot
```
