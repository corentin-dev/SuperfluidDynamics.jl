# Gradient Field

Gradient field is used to contain derivatives of a field by the simulation. It depends on a [field](@ref Fields). It may or may not compute rotation along the ``z`` axis.

!!! info
    There exists two kinds of `GradientField` classes, with or without rotation.

## Constructors

```@docs
GradientField
```

## Structures

```@docs
SuperfluidDynamics.GradientField2D
```

```@docs
SuperfluidDynamics.GradientRotField2D
```

```@docs
SuperfluidDynamics.GradientField3D
```

```@docs
SuperfluidDynamics.GradientRotField3D
```
