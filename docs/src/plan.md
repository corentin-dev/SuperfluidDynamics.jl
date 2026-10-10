# Plan

Plans describes the way derivatives can be computed. They also store temporary arrays distributed along each direction.

## Constructors

```@docs
Plan
```

## Structures

### FFT Plans

```@docs
SuperfluidDynamics.PlanFFT2D
```

```@docs
SuperfluidDynamics.PlanFFT3D
```

### Finite Difference Plans

```@docs
SuperfluidDynamics.PlanFD2D
```

```@docs
SuperfluidDynamics.PlanFD3D
```

### Compact Finite Difference Plans

The compact schemes resolve derivatives to 6th order at a cost comparable to
a second-order stencil, at the price of a tridiagonal line solve per
direction. The boundary condition on each axis is set through the `bcs`
keyword of [`SuperfluidDynamics.CompactPlan`](@ref): `0`/`:periodic` (default),
`1`/`:dirichlet` (homogeneous Dirichlet) or `2`/`:neumann` (homogeneous
Neumann, even-mirror closure). Axes may be mixed, e.g. `bcs = (2, 0)` bounds
`x` with Neumann and keeps `y` periodic.

```@docs
SuperfluidDynamics.CompactPlan
```

```@docs
SuperfluidDynamics.PlanCompact2D
```

```@docs
SuperfluidDynamics.PlanCompact3D
```
