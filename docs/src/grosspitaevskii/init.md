# Initialization

Initialization is not mandatory as you can use your own function to initialize a [field](@ref Fields). You can see them as a helper to setup a simulation.

If you want to use it, you first need to setup a `Grid` and a `Field`. Once you have it, you can run for example:
```julia-repl
julia> init = InitThomasFermi(field, β, γx = γx, γy = γy, γz = γz)
InitThomasFermi
  ├──────  parameters: γx 1 γy 1 γz 1 β 1000
  ├──────────────  μ = γz^1/4 / π^3/4 × exp(-1/2 × (x²+y²+z²))
  └──────────────  ϕ = (1-Ω) × s1 + Ω × s2
# or
julia> init = InitGauss(field, Ω = Ω)
InitGauss
  ├──────  parameters: γz 1.0 Ω 0.9
  ├─────────────  s1 = γz^1/4 / π^3/4 × exp(-1/2 × (x²+y²+γz z²))
  ├─────────────  s2 = γz^1/4 (x+iy) / π^3/4 × exp(-1/2 × (x²+y²+γz z²))
  └──────────────  ϕ = (1-Ω) × s1 + Ω × s2
```

!!! info
    For the moment, for Bose-Einstein Condensates, only Thomas-Fermi and Gauss type are implemented. For Quantum Turbulence, you have an external velocity, only implementing Taylor-Green type for the moment.

## Thomas-Fermi

### Initializer functions

```@docs
initField!(::SuperfluidDynamics.InitThomasFermi{F}) where {F<:SuperfluidDynamics.AbstractField2D}
initField!(::SuperfluidDynamics.InitThomasFermi{F}) where {F<:SuperfluidDynamics.AbstractField3D}
```

### Constructors

```@docs
InitThomasFermi(:: F,:: Real;:: Real, :: Real, :: Real) where {F<:SuperfluidDynamics.AbstractField}
```

### Structures

```@docs
SuperfluidDynamics.InitThomasFermi
```

## Gauss

### Initializer functions

```@docs
initField!(::SuperfluidDynamics.InitGauss{F}) where {F<:SuperfluidDynamics.AbstractField2D}
initField!(::SuperfluidDynamics.InitGauss{F}) where {F<:SuperfluidDynamics.AbstractField3D}
```

### Constructors

```@docs
InitGauss(::F; ::Real, :: Real) where {F<:SuperfluidDynamics.AbstractField}
```

### Structures

```@docs
SuperfluidDynamics.InitGauss
```

## External velocity

### Initializer functions

```@docs
initField!(::InitExternalVelocity{F}) where {F<:SuperfluidDynamics.AbstractField2D}
```

### Constructors

```@docs
InitExternalVelocity(:: F,:: Real,:: Real) where {F<:SuperfluidDynamics.AbstractField}
```

### Structures

```@docs
SuperfluidDynamics.InitExternalVelocity
```
