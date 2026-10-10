# Grids

## Type definition

[`Grid`](@ref) objects are used to describe the physical dimension and discretization of physical space.

Examples are given below. They can be used to construct a distributed [`Field`](@ref), but [`Grid`](@ref) represents a non-distributed geometry.

!!! info
    Only 2D and 3D rectilinear grids are supported at the moment.
    To have a more friendly type, some aliases has been declared: [`Grid2D`](@ref) and [`Grid3D`](@ref).

```@docs
Grid
```

## Constructors

To construct a [`Grid2D`](@ref), a more friendly constructor has been defined as follow:

```@docs
Grid(size::Tuple{Integer,Integer},
              bounds::Tuple{Tuple{Real,Real},Tuple{Real,Real}};
              FT=Float64, array_type)
```

To construct a [`Grid3D`](@ref), a more friendly constructor has been defined as follow:

```@docs
Grid(size::Tuple{Integer,Integer,Integer},
              bounds::Tuple{Tuple{Real,Real},Tuple{Real,Real},Tuple{Real,Real}};
              FT=Float64, array_type)
```

## Aliases

To have a more friendly type, some aliases has been declared. For the moment, there exists two kinds of fields: [`Grid2D`](@ref) and [`Grid3D`](@ref).

```@docs
Grid2D
```

```@docs
Grid3D
```
